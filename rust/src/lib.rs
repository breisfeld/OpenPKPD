//! openpkpd._core — compiled inner-loop extensions.
//!
//! Currently provides:
//!
//! * `neg2ll_obs_loop` — vectorised observation log-likelihood accumulator
//!   that replaces the pure-Python per-observation loop in
//!   `IndividualModel.log_likelihood`.
//!
//! BLQ methods implemented here mirror `openpkpd.data.blq`:
//!   0 / 1  = no BLQ / M1  → exclude BLQ observations
//!   2      = M2            → censored: log Φ((LLOQ−μ)/σ)
//!   3      = M3            → same formula as M2
//!   4      = M4            → truncated-normal: log [Φ(z_lloq)−Φ(z_0)] − log [1−Φ(z_0)]
//!   5      = M5            → impute DV = LLOQ/2
//!   6      = M6            → impute first BLQ with LLOQ/2, exclude rest
//!   7      = M7            → impute DV = 0

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use std::f64::consts::SQRT_2;
#[cfg(feature = "cvode-wrap-spike")]
use std::time::Instant;

#[cfg(feature = "cvode-wrap-spike")]
use cvode_wrap::{
    AbsTolerance, LinearMultistepMethod, RhsResult, SolverNoSensi, StepKind,
};

#[cfg(feature = "cvode-wrap-spike")]
struct WarfarinPkpdTheta {
    ktr: f64,
    ka: f64,
    cl: f64,
    v: f64,
    emax: f64,
    ec50: f64,
    kout: f64,
    e0: f64,
}

#[cfg(feature = "cvode-wrap-spike")]
fn cvode_wrap_warfarin_pkpd_rhs(
    _t: f64,
    y: &[f64; 4],
    dy: &mut [f64; 4],
    theta: &WarfarinPkpdTheta,
) -> RhsResult {
    let a1 = y[0];
    let a2 = y[1];
    let a3 = y[2];
    let a4 = y[3];
    let conc = a3 / theta.v;
    let pd = 1.0 - theta.emax * conc / (theta.ec50 + conc);

    *dy = [
        -theta.ktr * a1,
        theta.ktr * a1 - theta.ka * a2,
        theta.ka * a2 - (theta.cl / theta.v) * a3,
        theta.kout * theta.e0 * (pd - 1.0) - theta.kout * a4,
    ];
    RhsResult::Ok
}

#[cfg(feature = "cvode-wrap-spike")]
fn cvode_wrap_linear_rhs(
    _t: f64,
    y: &[f64; 2],
    dy: &mut [f64; 2],
    _data: &(),
) -> RhsResult {
    *dy = [-0.5 * y[0], 0.5 * y[0] - 0.25 * y[1]];
    RhsResult::Ok
}

// ln(2π)
const LOG2PI: f64 = 1.837_877_066_409_345_5_f64;

// ── math helpers ─────────────────────────────────────────────────────────────

/// Normal log-likelihood: −½ [ln(2π) + ln(σ²) + (y−μ)²/σ²]
///
/// Returns −1×10³⁰ when σ² ≤ 0 (matches Python fallback).
#[inline(always)]
fn normal_ll(y: f64, mu: f64, var: f64) -> f64 {
    if var <= 0.0 {
        return -1e30;
    }
    let r = y - mu;
    -0.5 * (LOG2PI + var.ln() + r * r / var)
}

/// Standard normal CDF:  Φ(z) = erfc(−z / √2) / 2
#[inline(always)]
fn norm_cdf(z: f64) -> f64 {
    libm::erfc(-z / SQRT_2) * 0.5
}

/// Standard normal log-CDF:  ln Φ(z)
///
/// Clamps to ln(1×10⁻³⁰⁰) to avoid −∞ in degenerate tails.
#[inline(always)]
fn norm_logcdf(z: f64) -> f64 {
    let v = libm::erfc(-z / SQRT_2) * 0.5;
    v.max(1e-300_f64).ln()
}

// ── public extension function ─────────────────────────────────────────────────

/// Compute −2 × Σ log p(yᵢ | μᵢ, σᵢ²) for all active observations of a
/// single subject.
///
/// All arrays must have the same length `n` (= number of observation rows,
/// including MDV=1 rows which are skipped via `obs_mask`).
///
/// Parameters
/// ----------
/// dv : float64[n]   — observed dependent variable (NaN for MDV rows)
/// pred : float64[n] — model predictions
/// var : float64[n]  — residual variances (must be > 0 for active obs)
/// obs_mask : bool[n]— True = active observation (MDV=0, not already excluded)
/// lloq : float64[n] — per-observation LLOQ; NaN means no BLQ for that obs
/// blq_method : u8   — 0/1=M1(exclude), 2=M2, 3=M3, 4=M4, 5=M5, 6=M6, 7=M7
///
/// Returns
/// -------
/// float : −2 × log-likelihood sum
#[pyfunction]
fn neg2ll_obs_loop(
    dv: PyReadonlyArray1<f64>,
    pred: PyReadonlyArray1<f64>,
    var: PyReadonlyArray1<f64>,
    obs_mask: PyReadonlyArray1<bool>,
    lloq: PyReadonlyArray1<f64>,
    blq_method: u8,
) -> f64 {
    let dv = dv.as_array();
    let pred = pred.as_array();
    let var = var.as_array();
    let mask = obs_mask.as_array();
    let lloq = lloq.as_array();

    let n = mask.len()
        .min(dv.len())
        .min(pred.len())
        .min(var.len())
        .min(lloq.len());

    let mut ll = 0.0_f64;
    let mut seen_blq_m6 = false;

    for i in 0..n {
        if !mask[i] {
            continue;
        }
        let y = dv[i];
        if y.is_nan() {
            continue;
        }
        let mu = pred[i];
        let v = var[i];
        let lloq_i = lloq[i];

        let is_blq = !lloq_i.is_nan() && y < lloq_i;

        if is_blq {
            match blq_method {
                // 0 = no BLQ active, 1 = M1: exclude BLQ observations
                0 | 1 => continue,

                // M2 / M3: censored likelihood  log Φ((LLOQ−μ)/σ)
                2 | 3 => {
                    let sigma = v.sqrt();
                    if sigma <= 0.0 {
                        ll += -1e30;
                    } else {
                        ll += norm_logcdf((lloq_i - mu) / sigma);
                    }
                }

                // M4: truncated normal  log[Φ(z_lloq)−Φ(z_0)] − log[1−Φ(z_0)]
                4 => {
                    let sigma = v.sqrt();
                    if sigma <= 0.0 {
                        ll += -1e30;
                    } else {
                        let z_lloq = (lloq_i - mu) / sigma;
                        let z_0 = -mu / sigma;
                        let prob_window = norm_cdf(z_lloq) - norm_cdf(z_0);
                        let prob_pos = 1.0 - norm_cdf(z_0);
                        if prob_pos <= 0.0 || prob_window <= 0.0 {
                            ll += -1e30;
                        } else {
                            ll += prob_window.ln() - prob_pos.ln();
                        }
                    }
                }

                // M5: impute DV = LLOQ/2
                5 => ll += normal_ll(lloq_i * 0.5, mu, v),

                // M6: impute first BLQ with LLOQ/2, exclude the rest
                6 => {
                    if !seen_blq_m6 {
                        seen_blq_m6 = true;
                        ll += normal_ll(lloq_i * 0.5, mu, v);
                    }
                    // subsequent BLQ in M6 → skip
                }

                // M7: impute DV = 0
                7 => ll += normal_ll(0.0, mu, v),

                // Unknown method: exclude (safe fallback)
                _ => continue,
            }
        } else {
            // Normal (non-BLQ) observation
            ll += normal_ll(y, mu, v);
        }
    }

    -2.0 * ll
}

#[cfg(feature = "cvode-wrap-spike")]
#[pyfunction]
fn cvode_wrap_linear_probe(tout: f64) -> PyResult<Vec<f64>> {
    if !tout.is_finite() || tout < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tout must be finite and non-negative",
        ));
    }

    let y0 = [1.0_f64, 0.0_f64];
    let mut solver = SolverNoSensi::new(
        LinearMultistepMethod::Bdf,
        cvode_wrap_linear_rhs,
        0.0,
        &y0,
        1e-8,
        AbsTolerance::scalar(1e-10),
        (),
    )
    .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(format!("{err:?}")))?;

    let (_tret, y) = solver
        .step(tout, StepKind::Normal)
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(format!("{err:?}")))?;

    Ok(y.to_vec())
}

#[cfg(feature = "cvode-wrap-spike")]
#[pyfunction]
fn cvode_wrap_warfarin_pkpd_probe(
    times: Vec<f64>,
    dose_amt: f64,
    theta: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    if theta.len() != 8 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "theta must have exactly 8 values: KTR, KA, CL, V, EMAX, EC50, KOUT, E0",
        ));
    }
    if !dose_amt.is_finite() || dose_amt < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dose_amt must be finite and non-negative",
        ));
    }
    if times.iter().any(|t| !t.is_finite() || *t < 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "times must be finite and non-negative",
        ));
    }
    if times.windows(2).any(|w| w[1] < w[0]) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "times must be sorted in non-decreasing order",
        ));
    }

    let theta = WarfarinPkpdTheta {
        ktr: theta[0],
        ka: theta[1],
        cl: theta[2],
        v: theta[3],
        emax: theta[4],
        ec50: theta[5],
        kout: theta[6],
        e0: theta[7],
    };
    let y0 = [dose_amt, 0.0_f64, 0.0_f64, 0.0_f64];

    let mut solver = SolverNoSensi::new(
        LinearMultistepMethod::Bdf,
        cvode_wrap_warfarin_pkpd_rhs,
        0.0,
        &y0,
        1e-8,
        AbsTolerance::scalar(1e-10),
        theta,
    )
    .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(format!("{err:?}")))?;

    let mut out = Vec::with_capacity(times.len());
    for &tout in &times {
        if tout == 0.0 {
            out.push(y0.to_vec());
            continue;
        }
        let (_tret, y) = solver
            .step(tout, StepKind::Normal)
            .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(format!("{err:?}")))?;
        out.push(y.to_vec());
    }
    Ok(out)
}

#[cfg(feature = "cvode-wrap-spike")]
#[pyfunction]
fn cvode_wrap_warfarin_pkpd_repeat_probe(
    times: Vec<f64>,
    dose_amt: f64,
    theta: Vec<f64>,
    n_repeats: usize,
) -> PyResult<(f64, Vec<f64>)> {
    if n_repeats == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_repeats must be >= 1",
        ));
    }
    if theta.len() != 8 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "theta must have exactly 8 values: KTR, KA, CL, V, EMAX, EC50, KOUT, E0",
        ));
    }
    if !dose_amt.is_finite() || dose_amt < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dose_amt must be finite and non-negative",
        ));
    }
    if times.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "times must not be empty",
        ));
    }
    if times.iter().any(|t| !t.is_finite() || *t < 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "times must be finite and non-negative",
        ));
    }
    if times.windows(2).any(|w| w[1] < w[0]) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "times must be sorted in non-decreasing order",
        ));
    }

    let base_theta = WarfarinPkpdTheta {
        ktr: theta[0],
        ka: theta[1],
        cl: theta[2],
        v: theta[3],
        emax: theta[4],
        ec50: theta[5],
        kout: theta[6],
        e0: theta[7],
    };
    let y0 = [dose_amt, 0.0_f64, 0.0_f64, 0.0_f64];
    let mut last_state = y0.to_vec();
    let t0 = Instant::now();

    for _ in 0..n_repeats {
        let mut solver = SolverNoSensi::new(
            LinearMultistepMethod::Bdf,
            cvode_wrap_warfarin_pkpd_rhs,
            0.0,
            &y0,
            1e-8,
            AbsTolerance::scalar(1e-10),
            WarfarinPkpdTheta {
                ktr: base_theta.ktr,
                ka: base_theta.ka,
                cl: base_theta.cl,
                v: base_theta.v,
                emax: base_theta.emax,
                ec50: base_theta.ec50,
                kout: base_theta.kout,
                e0: base_theta.e0,
            },
        )
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(format!("{err:?}")))?;

        for &tout in &times {
            let (_, y) = if tout == 0.0 {
                (0.0, &y0)
            } else {
                solver.step(tout, StepKind::Normal)
                    .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(format!("{err:?}")))?
            };
            last_state = y.to_vec();
        }
    }

    Ok((t0.elapsed().as_secs_f64(), last_state))
}

// ── module registration ───────────────────────────────────────────────────────

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(neg2ll_obs_loop, m)?)?;
    #[cfg(feature = "cvode-wrap-spike")]
    m.add_function(wrap_pyfunction!(cvode_wrap_linear_probe, m)?)?;
    #[cfg(feature = "cvode-wrap-spike")]
    m.add_function(wrap_pyfunction!(cvode_wrap_warfarin_pkpd_probe, m)?)?;
    #[cfg(feature = "cvode-wrap-spike")]
    m.add_function(wrap_pyfunction!(cvode_wrap_warfarin_pkpd_repeat_probe, m)?)?;
    Ok(())
}
