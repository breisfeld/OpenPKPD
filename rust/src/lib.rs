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
#[cfg(feature = "native-cvodes")]
use std::time::Instant;

#[cfg(feature = "native-cvodes")]
use cvode_wrap::{
    AbsTolerance, LinearMultistepMethod, RhsResult, SensiAbsTolerance, SolverNoSensi,
    SolverSensi, StepKind,
};

/// Number of ODE parameters for the ADVAN6 mixed PK/PD model
/// (KTR, KA, CL, V, EMAX, EC50, KOUT, E0).
#[cfg(feature = "native-cvodes")]
const N_ODE_PARAMS: usize = 8;

#[cfg(feature = "native-cvodes")]
struct Advan6MixedPkpdTheta {
    ktr: f64,
    ka: f64,
    cl: f64,
    v: f64,
    emax: f64,
    ec50: f64,
    kout: f64,
    e0: f64,
}

#[cfg(feature = "native-cvodes")]
fn cvode_wrap_advan6_mixed_pkpd_rhs(
    _t: f64,
    y: &[f64; 4],
    dy: &mut [f64; 4],
    theta: &Advan6MixedPkpdTheta,
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

/// Forward-sensitivity RHS for the ADVAN6 mixed PK/PD ODE.
///
/// For each ODE parameter θⱼ the sensitivity sⱼ = dA/dθⱼ (a 4-vector)
/// satisfies the variational equation:
///
///     dsⱼ/dt = J(y, θ) · sⱼ + ∂f/∂θⱼ
///
/// where J is the 4×4 state Jacobian and ∂f/∂θⱼ is the direct
/// parameter derivative of the base RHS.  Both are computed analytically
/// from the current state y and parameters θ.
///
/// Parameter ordering: [KTR, KA, CL, V, EMAX, EC50, KOUT, E0]
#[cfg(feature = "native-cvodes")]
fn cvode_wrap_advan6_mixed_pkpd_sens_rhs(
    _t: f64,
    y: &[f64; 4],
    _ydot: &[f64; 4],
    ys: [&[f64; 4]; N_ODE_PARAMS],
    ysdot: [&mut [f64; 4]; N_ODE_PARAMS],
    theta: &Advan6MixedPkpdTheta,
) -> RhsResult {
    let a1 = y[0];
    let a2 = y[1];
    let a3 = y[2];
    let a4 = y[3];
    let conc = a3 / theta.v;
    let denom = theta.ec50 + conc;           // ec50 + conc
    let denom2 = denom * denom;
    let v2 = theta.v * theta.v;
    let pd = 1.0 - theta.emax * conc / denom;

    // ── State Jacobian J (only non-zero entries) ─────────────────────────────
    // Row 0:  dA1' / dA_i
    let j00 = -theta.ktr;
    // Row 1:  dA2' / dA_i
    let j10 = theta.ktr;
    let j11 = -theta.ka;
    // Row 2:  dA3' / dA_i
    let j21 = theta.ka;
    let j22 = -theta.cl / theta.v;
    // Row 3:  dA4' / dA_i   (only A3 and A4 have non-zero entries)
    //   dA4'/dA3 = kout * e0 * d(pd)/d(A3)
    //            = kout * e0 * (−emax * ec50) / (v * denom²)
    let j32 = theta.kout * theta.e0 * (-theta.emax * theta.ec50) / (theta.v * denom2);
    let j33 = -theta.kout;

    // ── Direct parameter derivatives ∂f/∂θⱼ ────────────────────────────────
    // Layout: [d/dKTR, d/dKA, d/dCL, d/dV, d/dEMAX, d/dEC50, d/dKOUT, d/dE0]
    //
    // p_ktr:   [-A1,  A1,     0,                                  0]
    // p_ka:    [  0, -A2,    A2,                                  0]
    // p_cl:    [  0,   0, -A3/V,                                  0]
    // p_v:     [  0,   0, CL·A3/V², kout·e0·emax·ec50·A3/(V²·denom²)]
    // p_emax:  [  0,   0,     0, kout·e0·(−conc/denom)             ]
    // p_ec50:  [  0,   0,     0, kout·e0·emax·conc/denom²          ]
    // p_kout:  [  0,   0,     0, e0·(pd−1) − A4                   ]
    // p_e0:    [  0,   0,     0, kout·(pd−1)                       ]
    let direct: [[f64; 4]; N_ODE_PARAMS] = [
        [-a1, a1, 0.0, 0.0],
        [0.0, -a2, a2, 0.0],
        [0.0, 0.0, -a3 / theta.v, 0.0],
        [
            0.0,
            0.0,
            theta.cl * a3 / v2,
            theta.kout * theta.e0 * theta.emax * theta.ec50 * a3 / (v2 * denom2),
        ],
        [0.0, 0.0, 0.0, theta.kout * theta.e0 * (-conc / denom)],
        [0.0, 0.0, 0.0, theta.kout * theta.e0 * theta.emax * conc / denom2],
        [0.0, 0.0, 0.0, theta.e0 * (pd - 1.0) - a4],
        [0.0, 0.0, 0.0, theta.kout * (pd - 1.0)],
    ];

    // ── Assemble sensitivity derivatives ────────────────────────────────────
    // ysdot[j] = J · ys[j] + direct[j]
    for (j, (s, sd)) in ys.into_iter().zip(ysdot.into_iter()).enumerate() {
        let p = direct[j];
        *sd = [
            j00 * s[0]              + p[0],
            j10 * s[0] + j11 * s[1] + p[1],
            j21 * s[1] + j22 * s[2] + p[2],
            j32 * s[2] + j33 * s[3] + p[3],
        ];
    }
    RhsResult::Ok
}

#[cfg(feature = "native-cvodes")]
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

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_linear_probe(tout: f64) -> PyResult<Vec<f64>> {
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

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_advan6_mixed_pkpd_probe(
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

    let theta = Advan6MixedPkpdTheta {
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
        cvode_wrap_advan6_mixed_pkpd_rhs,
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

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_advan6_mixed_pkpd_repeat_probe(
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

    let base_theta = Advan6MixedPkpdTheta {
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
            cvode_wrap_advan6_mixed_pkpd_rhs,
            0.0,
            &y0,
            1e-8,
            AbsTolerance::scalar(1e-10),
            Advan6MixedPkpdTheta {
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

// ── multi-dose probe ─────────────────────────────────────────────────────────

/// Build an `Advan6MixedPkpdTheta` from a slice, returning `Err` on bad input.
#[cfg(feature = "native-cvodes")]
fn unpack_theta(theta: &[f64]) -> PyResult<Advan6MixedPkpdTheta> {
    if theta.len() != 8 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "theta must have exactly 8 values: KTR, KA, CL, V, EMAX, EC50, KOUT, E0",
        ));
    }
    Ok(Advan6MixedPkpdTheta {
        ktr: theta[0], ka: theta[1], cl: theta[2], v: theta[3],
        emax: theta[4], ec50: theta[5], kout: theta[6], e0: theta[7],
    })
}

/// Clone-like copy of `Advan6MixedPkpdTheta` (not derived to keep the struct simple).
#[cfg(feature = "native-cvodes")]
fn copy_theta(t: &Advan6MixedPkpdTheta) -> Advan6MixedPkpdTheta {
    Advan6MixedPkpdTheta {
        ktr: t.ktr, ka: t.ka, cl: t.cl, v: t.v,
        emax: t.emax, ec50: t.ec50, kout: t.kout, e0: t.e0,
    }
}

/// CVODES BDF integrator for the 4-state warfarin-shaped mixed PK/PD ODE
/// with multiple IV bolus doses.
///
/// Algorithm
/// ---------
/// Events (doses and observations) are merged into a single sorted timeline.
/// At each dose time a bolus is applied to A1 and a new CVODES solver is
/// created for the next integration segment.  Observations before the first
/// dose return the zero vector.  Observations at a dose time return the
/// post-dose state.
///
/// Parameters
/// ----------
/// obs_times  : sorted observation times (non-decreasing)
/// dose_times : bolus dose times (must be sorted non-decreasing)
/// dose_amts  : dose amounts corresponding to dose_times
/// theta      : [KTR, KA, CL, V, EMAX, EC50, KOUT, E0]
///
/// Returns
/// -------
/// Vec of length n_obs, each element a Vec<f64> of length 4 (A1..A4).
#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_advan6_mixed_pkpd_probe_multidose(
    obs_times: Vec<f64>,
    dose_times: Vec<f64>,
    dose_amts: Vec<f64>,
    theta: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    // ── input validation ─────────────────────────────────────────────────────
    if dose_times.len() != dose_amts.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dose_times and dose_amts must have the same length",
        ));
    }
    if dose_times.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dose_times must not be empty",
        ));
    }
    if obs_times.iter().any(|t| !t.is_finite() || *t < 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "obs_times must be finite and non-negative",
        ));
    }
    if obs_times.windows(2).any(|w| w[1] < w[0]) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "obs_times must be sorted in non-decreasing order",
        ));
    }
    if dose_times.windows(2).any(|w| w[1] < w[0]) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dose_times must be sorted in non-decreasing order",
        ));
    }
    if dose_times.iter().any(|t| !t.is_finite() || *t < 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dose_times must be finite and non-negative",
        ));
    }
    if dose_amts.iter().any(|a| !a.is_finite() || *a < 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dose_amts must be finite and non-negative",
        ));
    }

    let base_theta = unpack_theta(&theta)?;
    let n_obs = obs_times.len();
    let n_doses = dose_times.len();

    // ── segment-based integration ─────────────────────────────────────────────
    //
    // State vector and output allocation.
    let mut y: [f64; 4] = [0.0; 4];
    let mut out: Vec<Vec<f64>> = vec![vec![0.0; 4]; n_obs];
    let mut obs_i = 0usize;

    // Phase 1: observations strictly before the first dose → zero state.
    let first_dose_t = dose_times[0];
    while obs_i < n_obs && obs_times[obs_i] < first_dose_t {
        // out[obs_i] is already the zero vector from initialization.
        obs_i += 1;
    }

    // Phase 2: one solver per dose interval.
    //
    // For each dose we:
    //   a) apply the instantaneous bolus to y[0]
    //   b) record observations at exactly the dose time (post-dose state)
    //   c) create a new CVODES solver starting at t_dose with state y
    //   d) step through every observation in (t_dose, next_dose)
    //   e) advance the solver to the next dose time so the state y is ready
    //      for the next iteration's bolus application
    //
    // The solver is declared *inside* the loop body so Rust can infer its
    // concrete type without an explicit generic annotation.
    for dose_i in 0..n_doses {
        let t_dose = dose_times[dose_i];
        let next_dose_t = if dose_i + 1 < n_doses {
            dose_times[dose_i + 1]
        } else {
            f64::INFINITY
        };

        // (a) Apply instantaneous bolus.
        y[0] += dose_amts[dose_i];

        // (b) Observations at exactly the dose time get the post-dose state.
        while obs_i < n_obs && obs_times[obs_i] <= t_dose {
            out[obs_i] = y.to_vec();
            obs_i += 1;
        }

        // Determine whether this segment needs integration.
        let needs_integration = obs_i < n_obs  // observations remain
            || next_dose_t.is_finite();        // must advance state to next dose

        if !needs_integration {
            break;
        }

        // (c) Create a new solver for [t_dose, next_dose_t).
        let mut solver = SolverNoSensi::new(
            LinearMultistepMethod::Bdf,
            cvode_wrap_advan6_mixed_pkpd_rhs,
            t_dose,
            &y,
            1e-8,
            AbsTolerance::scalar(1e-10),
            copy_theta(&base_theta),
        )
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(format!("{err:?}")))?;

        // (d) Step through observations in this dose interval.
        while obs_i < n_obs
            && (next_dose_t.is_infinite() || obs_times[obs_i] < next_dose_t)
        {
            let tout = obs_times[obs_i];
            let (_, y_new) = solver
                .step(tout, StepKind::Normal)
                .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(format!("{err:?}")))?;
            out[obs_i] = y_new.to_vec();
            y = *y_new;
            obs_i += 1;
        }

        // (e) Advance solver to the next dose time to obtain the pre-dose state.
        if next_dose_t.is_finite() {
            let (_, y_new) = solver
                .step(next_dose_t, StepKind::Normal)
                .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(format!("{err:?}")))?;
            y = *y_new;
        }
        // solver is dropped here; a fresh one is created in the next iteration.
    }

    Ok(out)
}

// ── sensitivity probe ─────────────────────────────────────────────────────────

/// CVODES BDF forward-sensitivity probe for the 4-state mixed PK/PD ODE
/// with multiple IV bolus doses.
///
/// Computes the ODE states **and** their sensitivities with respect to all
/// 8 ODE parameters (KTR, KA, CL, V, EMAX, EC50, KOUT, E0) in a single
/// integration pass using `SolverSensi`.
///
/// Algorithm
/// ---------
/// Mirrors `native_cvodes_advan6_mixed_pkpd_probe_multidose` with a
/// `SolverSensi` solver replacing `SolverNoSensi`.  At each dose event
/// the bolus is applied to A1 (as before) and the sensitivity initial
/// conditions are carried over unchanged — because dose amounts are not
/// functions of the ODE parameters.
///
/// Parameters
/// ----------
/// obs_times  : sorted observation times (non-decreasing, finite, ≥ 0)
/// dose_times : bolus dose times (sorted non-decreasing, finite, ≥ 0)
/// dose_amts  : dose amounts corresponding to dose_times (finite, ≥ 0)
/// theta      : [KTR, KA, CL, V, EMAX, EC50, KOUT, E0]
///
/// Returns
/// -------
/// (states, sensitivities) where:
///   states         Vec of n_obs × [A1, A2, A3, A4]
///   sensitivities  Vec of n_obs × [dA0/dθ0, dA1/dθ0, dA2/dθ0, dA3/dθ0,
///                                   dA0/dθ1, …, dA3/dθ7]   (length 32)
#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_advan6_mixed_pkpd_sensitivity_probe_multidose(
    obs_times: Vec<f64>,
    dose_times: Vec<f64>,
    dose_amts: Vec<f64>,
    theta: Vec<f64>,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    // ── input validation ─────────────────────────────────────────────────────
    if dose_times.len() != dose_amts.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dose_times and dose_amts must have the same length",
        ));
    }
    if dose_times.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dose_times must not be empty",
        ));
    }
    if obs_times.iter().any(|t| !t.is_finite() || *t < 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "obs_times must be finite and non-negative",
        ));
    }
    if obs_times.windows(2).any(|w| w[1] < w[0]) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "obs_times must be sorted in non-decreasing order",
        ));
    }
    if dose_times.windows(2).any(|w| w[1] < w[0]) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dose_times must be sorted in non-decreasing order",
        ));
    }
    if dose_times.iter().any(|t| !t.is_finite() || *t < 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dose_times must be finite and non-negative",
        ));
    }
    if dose_amts.iter().any(|a| !a.is_finite() || *a < 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dose_amts must be finite and non-negative",
        ));
    }

    let base_theta = unpack_theta(&theta)?;
    let n_obs = obs_times.len();
    let n_doses = dose_times.len();

    // ── state and sensitivity arrays ─────────────────────────────────────────
    let mut y = [0.0f64; 4];
    // ys[j][i] = dA_i/dθ_j;  initial sensitivities are zero because A(0)
    // does not depend on the ODE parameters (dose amounts ≠ ODE params).
    let mut ys = [[0.0f64; 4]; N_ODE_PARAMS];

    // Pre-allocate outputs (zero-initialised: correct for pre-dose observations)
    let mut out_states: Vec<Vec<f64>> = vec![vec![0.0; 4]; n_obs];
    let mut out_sens:   Vec<Vec<f64>> = vec![vec![0.0; 4 * N_ODE_PARAMS]; n_obs];
    let mut obs_i = 0usize;

    // Phase 1: observations strictly before the first dose → zero state/sens
    let first_dose_t = dose_times[0];
    while obs_i < n_obs && obs_times[obs_i] < first_dose_t {
        obs_i += 1;
    }

    // ── helper: flatten sensitivity matrix into a 32-element Vec ────────────
    let flatten_sens = |ys: &[[f64; 4]; N_ODE_PARAMS]| -> Vec<f64> {
        let mut v = Vec::with_capacity(4 * N_ODE_PARAMS);
        for j in 0..N_ODE_PARAMS {
            v.extend_from_slice(&ys[j]);
        }
        v
    };

    // Phase 2: one SolverSensi segment per dose interval
    for dose_i in 0..n_doses {
        let t_dose = dose_times[dose_i];
        let next_dose_t = if dose_i + 1 < n_doses {
            dose_times[dose_i + 1]
        } else {
            f64::INFINITY
        };

        // (a) Apply instantaneous bolus to A1; sensitivities unchanged.
        y[0] += dose_amts[dose_i];

        // (b) Record obs at exactly the dose time (post-dose state).
        while obs_i < n_obs && obs_times[obs_i] <= t_dose {
            out_states[obs_i] = y.to_vec();
            out_sens[obs_i]   = flatten_sens(&ys);
            obs_i += 1;
        }

        let needs_integration = obs_i < n_obs || next_dose_t.is_finite();
        if !needs_integration {
            break;
        }

        // (c) Create SolverSensi for [t_dose, next_dose_t).
        //
        // Sensitivity tolerances match the state tolerances so that
        // gradient accuracy tracks ODE accuracy.
        let mut solver = SolverSensi::new(
            LinearMultistepMethod::Bdf,
            cvode_wrap_advan6_mixed_pkpd_rhs,
            cvode_wrap_advan6_mixed_pkpd_sens_rhs,
            t_dose,
            &y,
            &ys,
            1e-8,
            AbsTolerance::scalar(1e-10),
            SensiAbsTolerance::scalar([1e-10f64; N_ODE_PARAMS]),
            copy_theta(&base_theta),
        )
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(format!("{err:?}")))?;

        // (d) Step through observations in this dose interval.
        while obs_i < n_obs
            && (next_dose_t.is_infinite() || obs_times[obs_i] < next_dose_t)
        {
            let tout = obs_times[obs_i];
            let (_, y_new, ys_new) = solver
                .step(tout, StepKind::Normal)
                .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(format!("{err:?}")))?;
            y = *y_new;
            for (dst, src) in ys.iter_mut().zip(ys_new.into_iter()) {
                *dst = *src;
            }
            out_states[obs_i] = y.to_vec();
            out_sens[obs_i]   = flatten_sens(&ys);
            obs_i += 1;
        }

        // (e) Advance to next dose time to obtain the pre-dose carry-over state.
        if next_dose_t.is_finite() {
            let (_, y_new, ys_new) = solver
                .step(next_dose_t, StepKind::Normal)
                .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(format!("{err:?}")))?;
            y = *y_new;
            for (dst, src) in ys.iter_mut().zip(ys_new.into_iter()) {
                *dst = *src;
            }
        }
        // solver is dropped here; fresh one created next iteration.
    }

    Ok((out_states, out_sens))
}

// ── shared helpers for template ODE solvers ───────────────────────────────────

/// Validate inputs common to all multidose probe functions.
#[cfg(feature = "native-cvodes")]
fn validate_multidose_inputs(
    obs_times: &[f64],
    dose_times: &[f64],
    dose_amts: &[f64],
    n_theta: usize,
    theta: &[f64],
    theta_desc: &str,
) -> PyResult<()> {
    if theta.len() != n_theta {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "theta must have exactly {} values: {}",
            n_theta, theta_desc
        )));
    }
    if dose_times.len() != dose_amts.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dose_times and dose_amts must have the same length",
        ));
    }
    if dose_times.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("dose_times must not be empty"));
    }
    if obs_times.iter().any(|t| !t.is_finite() || *t < 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "obs_times must be finite and non-negative",
        ));
    }
    if obs_times.windows(2).any(|w| w[1] < w[0]) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "obs_times must be sorted in non-decreasing order",
        ));
    }
    if dose_times.windows(2).any(|w| w[1] < w[0]) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dose_times must be sorted in non-decreasing order",
        ));
    }
    if dose_times.iter().any(|t| !t.is_finite() || *t < 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dose_times must be finite and non-negative",
        ));
    }
    if dose_amts.iter().any(|a| !a.is_finite() || *a < 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dose_amts must be finite and non-negative",
        ));
    }
    Ok(())
}

/// Flatten a 2-D sensitivity array `[[f64; N_STATES]; N_PARAMS]` into
/// a contiguous `Vec<f64>` in parameter-major order.
#[cfg(feature = "native-cvodes")]
fn flatten_sens_2d<const N_PARAMS: usize, const N_STATES: usize>(
    ys: &[[f64; N_STATES]; N_PARAMS],
) -> Vec<f64> {
    let mut v = Vec::with_capacity(N_PARAMS * N_STATES);
    for j in 0..N_PARAMS {
        v.extend_from_slice(&ys[j]);
    }
    v
}

// ──────────────────────────────────────────────────────────────────────────────
// Template: 1-compartment IV  (CL, V)
// State:    A1 (central)
// IPRED:    A1 / V          dose → A1
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "native-cvodes")]
#[derive(Clone, Copy)]
struct Cmt1IvTheta { cl: f64, v: f64 }

#[cfg(feature = "native-cvodes")]
fn rhs_1cmt_iv(_t: f64, y: &[f64; 1], dy: &mut [f64; 1], p: &Cmt1IvTheta) -> RhsResult {
    *dy = [-(p.cl / p.v) * y[0]];
    RhsResult::Ok
}

/// Sensitivity RHS for 1-cmt IV.  Parameters ordered: [CL, V].
/// J = [[-CL/V]].
/// ∂f/∂CL = [-A1/V],  ∂f/∂V = [CL·A1/V²].
#[cfg(feature = "native-cvodes")]
fn sens_rhs_1cmt_iv(
    _t: f64, y: &[f64; 1], _ydot: &[f64; 1],
    ys: [&[f64; 1]; 2], ysdot: [&mut [f64; 1]; 2],
    p: &Cmt1IvTheta,
) -> RhsResult {
    let a1 = y[0];
    let ke = p.cl / p.v;
    let v2 = p.v * p.v;
    let direct: [[f64; 1]; 2] = [[-a1 / p.v], [p.cl * a1 / v2]];
    for (j, (s, sd)) in ys.into_iter().zip(ysdot.into_iter()).enumerate() {
        *sd = [-ke * s[0] + direct[j][0]];
    }
    RhsResult::Ok
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_1cmt_iv_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>, theta: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 2, &theta, "CL, V")?;
    let p = Cmt1IvTheta { cl: theta[0], v: theta[1] };
    let n_obs = obs_times.len();
    let n_doses = dose_times.len();
    let mut y = [0.0f64; 1];
    let mut out: Vec<Vec<f64>> = vec![vec![0.0; 1]; n_obs];
    let mut obs_i = 0usize;
    let first_t = dose_times[0];
    while obs_i < n_obs && obs_times[obs_i] < first_t { obs_i += 1; }
    for dose_i in 0..n_doses {
        let td = dose_times[dose_i];
        let next_t = if dose_i + 1 < n_doses { dose_times[dose_i + 1] } else { f64::INFINITY };
        y[0] += dose_amts[dose_i];
        while obs_i < n_obs && obs_times[obs_i] <= td { out[obs_i] = y.to_vec(); obs_i += 1; }
        if obs_i >= n_obs && next_t.is_infinite() { break; }
        let mut solver = SolverNoSensi::new(LinearMultistepMethod::Bdf, rhs_1cmt_iv, td, &y,
            1e-8, AbsTolerance::scalar(1e-10), p,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
        while obs_i < n_obs && (next_t.is_infinite() || obs_times[obs_i] < next_t) {
            let (_, y_new) = solver.step(obs_times[obs_i], StepKind::Normal)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new; out[obs_i] = y.to_vec(); obs_i += 1;
        }
        if next_t.is_finite() {
            let (_, y_new) = solver.step(next_t, StepKind::Normal)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new;
        }
    }
    Ok(out)
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_1cmt_iv_sensitivity_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>, theta: Vec<f64>,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 2, &theta, "CL, V")?;
    let p = Cmt1IvTheta { cl: theta[0], v: theta[1] };
    let n_obs = obs_times.len();
    let n_doses = dose_times.len();
    let mut y = [0.0f64; 1];
    let mut ys = [[0.0f64; 1]; 2];
    let mut out_s: Vec<Vec<f64>> = vec![vec![0.0; 1]; n_obs];
    let mut out_g: Vec<Vec<f64>> = vec![vec![0.0; 2]; n_obs];
    let mut obs_i = 0usize;
    let first_t = dose_times[0];
    while obs_i < n_obs && obs_times[obs_i] < first_t { obs_i += 1; }
    for dose_i in 0..n_doses {
        let td = dose_times[dose_i];
        let next_t = if dose_i + 1 < n_doses { dose_times[dose_i + 1] } else { f64::INFINITY };
        y[0] += dose_amts[dose_i];
        while obs_i < n_obs && obs_times[obs_i] <= td {
            out_s[obs_i] = y.to_vec(); out_g[obs_i] = flatten_sens_2d(&ys); obs_i += 1;
        }
        if obs_i >= n_obs && next_t.is_infinite() { break; }
        let mut solver = SolverSensi::new(LinearMultistepMethod::Bdf,
            rhs_1cmt_iv, sens_rhs_1cmt_iv, td, &y, &ys,
            1e-8, AbsTolerance::scalar(1e-10),
            SensiAbsTolerance::scalar([1e-10f64; 2]), p,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
        while obs_i < n_obs && (next_t.is_infinite() || obs_times[obs_i] < next_t) {
            let (_, y_new, ys_new) = solver.step(obs_times[obs_i], StepKind::Normal)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new;
            for (d, s) in ys.iter_mut().zip(ys_new.into_iter()) { *d = *s; }
            out_s[obs_i] = y.to_vec(); out_g[obs_i] = flatten_sens_2d(&ys); obs_i += 1;
        }
        if next_t.is_finite() {
            let (_, y_new, ys_new) = solver.step(next_t, StepKind::Normal)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new;
            for (d, s) in ys.iter_mut().zip(ys_new.into_iter()) { *d = *s; }
        }
    }
    Ok((out_s, out_g))
}

// ──────────────────────────────────────────────────────────────────────────────
// Template: 1-compartment oral  (KA, CL, V)
// State:    A1 (depot), A2 (central)
// IPRED:    A2 / V          dose → A1
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "native-cvodes")]
#[derive(Clone, Copy)]
struct Cmt1OralTheta { ka: f64, cl: f64, v: f64 }

#[cfg(feature = "native-cvodes")]
fn rhs_1cmt_oral(_t: f64, y: &[f64; 2], dy: &mut [f64; 2], p: &Cmt1OralTheta) -> RhsResult {
    *dy = [-p.ka * y[0], p.ka * y[0] - (p.cl / p.v) * y[1]];
    RhsResult::Ok
}

/// Sensitivity RHS for 1-cmt oral.  Parameters: [KA, CL, V].
/// J = [[-KA, 0], [KA, -CL/V]].
/// ∂f/∂KA=[-A1; A1], ∂f/∂CL=[0;-A2/V], ∂f/∂V=[0; CL·A2/V²].
#[cfg(feature = "native-cvodes")]
fn sens_rhs_1cmt_oral(
    _t: f64, y: &[f64; 2], _ydot: &[f64; 2],
    ys: [&[f64; 2]; 3], ysdot: [&mut [f64; 2]; 3],
    p: &Cmt1OralTheta,
) -> RhsResult {
    let (a1, a2) = (y[0], y[1]);
    let ke = p.cl / p.v;
    let v2 = p.v * p.v;
    let direct: [[f64; 2]; 3] = [[-a1, a1], [0.0, -a2 / p.v], [0.0, p.cl * a2 / v2]];
    for (j, (s, sd)) in ys.into_iter().zip(ysdot.into_iter()).enumerate() {
        let d = direct[j];
        *sd = [-p.ka * s[0] + d[0], p.ka * s[0] - ke * s[1] + d[1]];
    }
    RhsResult::Ok
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_1cmt_oral_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>, theta: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 3, &theta, "KA, CL, V")?;
    let p = Cmt1OralTheta { ka: theta[0], cl: theta[1], v: theta[2] };
    let n_obs = obs_times.len();
    let n_doses = dose_times.len();
    let mut y = [0.0f64; 2];
    let mut out: Vec<Vec<f64>> = vec![vec![0.0; 2]; n_obs];
    let mut obs_i = 0usize;
    let first_t = dose_times[0];
    while obs_i < n_obs && obs_times[obs_i] < first_t { obs_i += 1; }
    for dose_i in 0..n_doses {
        let td = dose_times[dose_i];
        let next_t = if dose_i + 1 < n_doses { dose_times[dose_i + 1] } else { f64::INFINITY };
        y[0] += dose_amts[dose_i];
        while obs_i < n_obs && obs_times[obs_i] <= td { out[obs_i] = y.to_vec(); obs_i += 1; }
        if obs_i >= n_obs && next_t.is_infinite() { break; }
        let mut solver = SolverNoSensi::new(LinearMultistepMethod::Bdf, rhs_1cmt_oral, td, &y,
            1e-8, AbsTolerance::scalar(1e-10), p,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
        while obs_i < n_obs && (next_t.is_infinite() || obs_times[obs_i] < next_t) {
            let (_, y_new) = solver.step(obs_times[obs_i], StepKind::Normal)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new; out[obs_i] = y.to_vec(); obs_i += 1;
        }
        if next_t.is_finite() {
            let (_, y_new) = solver.step(next_t, StepKind::Normal)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new;
        }
    }
    Ok(out)
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_1cmt_oral_sensitivity_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>, theta: Vec<f64>,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 3, &theta, "KA, CL, V")?;
    let p = Cmt1OralTheta { ka: theta[0], cl: theta[1], v: theta[2] };
    let n_obs = obs_times.len();
    let n_doses = dose_times.len();
    let mut y = [0.0f64; 2];
    let mut ys = [[0.0f64; 2]; 3];
    let mut out_s: Vec<Vec<f64>> = vec![vec![0.0; 2]; n_obs];
    let mut out_g: Vec<Vec<f64>> = vec![vec![0.0; 6]; n_obs];
    let mut obs_i = 0usize;
    let first_t = dose_times[0];
    while obs_i < n_obs && obs_times[obs_i] < first_t { obs_i += 1; }
    for dose_i in 0..n_doses {
        let td = dose_times[dose_i];
        let next_t = if dose_i + 1 < n_doses { dose_times[dose_i + 1] } else { f64::INFINITY };
        y[0] += dose_amts[dose_i];
        while obs_i < n_obs && obs_times[obs_i] <= td {
            out_s[obs_i] = y.to_vec(); out_g[obs_i] = flatten_sens_2d(&ys); obs_i += 1;
        }
        if obs_i >= n_obs && next_t.is_infinite() { break; }
        let mut solver = SolverSensi::new(LinearMultistepMethod::Bdf,
            rhs_1cmt_oral, sens_rhs_1cmt_oral, td, &y, &ys,
            1e-8, AbsTolerance::scalar(1e-10),
            SensiAbsTolerance::scalar([1e-10f64; 3]), p,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
        while obs_i < n_obs && (next_t.is_infinite() || obs_times[obs_i] < next_t) {
            let (_, y_new, ys_new) = solver.step(obs_times[obs_i], StepKind::Normal)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new;
            for (d, s) in ys.iter_mut().zip(ys_new.into_iter()) { *d = *s; }
            out_s[obs_i] = y.to_vec(); out_g[obs_i] = flatten_sens_2d(&ys); obs_i += 1;
        }
        if next_t.is_finite() {
            let (_, y_new, ys_new) = solver.step(next_t, StepKind::Normal)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new;
            for (d, s) in ys.iter_mut().zip(ys_new.into_iter()) { *d = *s; }
        }
    }
    Ok((out_s, out_g))
}

// ──────────────────────────────────────────────────────────────────────────────
// Template: 2-compartment IV  (CL, V1, Q, V2)
// State:    A1 (central), A2 (peripheral)
// IPRED:    A1 / V1         dose → A1
// k10=CL/V1, k12=Q/V1, k21=Q/V2
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "native-cvodes")]
#[derive(Clone, Copy)]
struct Cmt2IvTheta { cl: f64, v1: f64, q: f64, v2: f64 }

#[cfg(feature = "native-cvodes")]
fn rhs_2cmt_iv(_t: f64, y: &[f64; 2], dy: &mut [f64; 2], p: &Cmt2IvTheta) -> RhsResult {
    let k10 = p.cl / p.v1;
    let k12 = p.q / p.v1;
    let k21 = p.q / p.v2;
    *dy = [-(k10 + k12) * y[0] + k21 * y[1], k12 * y[0] - k21 * y[1]];
    RhsResult::Ok
}

/// Sensitivity RHS for 2-cmt IV.  Parameters: [CL, V1, Q, V2].
/// k10=CL/V1, k12=Q/V1, k21=Q/V2.
/// J=[[-(k10+k12),k21],[k12,-k21]].
/// ∂f/∂CL=[-A1/V1;0], ∂f/∂V1=[(CL+Q)A1/V1²;-QA1/V1²],
/// ∂f/∂Q=[-A1/V1+A2/V2;A1/V1-A2/V2], ∂f/∂V2=[-QA2/V2²;QA2/V2²].
#[cfg(feature = "native-cvodes")]
fn sens_rhs_2cmt_iv(
    _t: f64, y: &[f64; 2], _ydot: &[f64; 2],
    ys: [&[f64; 2]; 4], ysdot: [&mut [f64; 2]; 4],
    p: &Cmt2IvTheta,
) -> RhsResult {
    let (a1, a2) = (y[0], y[1]);
    let k10 = p.cl / p.v1; let k12 = p.q / p.v1; let k21 = p.q / p.v2;
    let v1s = p.v1 * p.v1; let v2s = p.v2 * p.v2;
    let direct: [[f64; 2]; 4] = [
        [-a1 / p.v1, 0.0],
        [(p.cl + p.q) * a1 / v1s, -p.q * a1 / v1s],
        [-a1 / p.v1 + a2 / p.v2, a1 / p.v1 - a2 / p.v2],
        [-p.q * a2 / v2s, p.q * a2 / v2s],
    ];
    let j00 = -(k10 + k12); let j11 = -k21;
    for (j, (s, sd)) in ys.into_iter().zip(ysdot.into_iter()).enumerate() {
        let d = direct[j];
        *sd = [j00 * s[0] + k21 * s[1] + d[0], k12 * s[0] + j11 * s[1] + d[1]];
    }
    RhsResult::Ok
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_2cmt_iv_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>, theta: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 4, &theta, "CL, V1, Q, V2")?;
    let p = Cmt2IvTheta { cl: theta[0], v1: theta[1], q: theta[2], v2: theta[3] };
    let n_obs = obs_times.len();
    let n_doses = dose_times.len();
    let mut y = [0.0f64; 2];
    let mut out: Vec<Vec<f64>> = vec![vec![0.0; 2]; n_obs];
    let mut obs_i = 0usize;
    let first_t = dose_times[0];
    while obs_i < n_obs && obs_times[obs_i] < first_t { obs_i += 1; }
    for dose_i in 0..n_doses {
        let td = dose_times[dose_i];
        let next_t = if dose_i + 1 < n_doses { dose_times[dose_i + 1] } else { f64::INFINITY };
        y[0] += dose_amts[dose_i];
        while obs_i < n_obs && obs_times[obs_i] <= td { out[obs_i] = y.to_vec(); obs_i += 1; }
        if obs_i >= n_obs && next_t.is_infinite() { break; }
        let mut solver = SolverNoSensi::new(LinearMultistepMethod::Bdf, rhs_2cmt_iv, td, &y,
            1e-8, AbsTolerance::scalar(1e-10), p,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
        while obs_i < n_obs && (next_t.is_infinite() || obs_times[obs_i] < next_t) {
            let (_, y_new) = solver.step(obs_times[obs_i], StepKind::Normal)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new; out[obs_i] = y.to_vec(); obs_i += 1;
        }
        if next_t.is_finite() {
            let (_, y_new) = solver.step(next_t, StepKind::Normal)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new;
        }
    }
    Ok(out)
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_2cmt_iv_sensitivity_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>, theta: Vec<f64>,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 4, &theta, "CL, V1, Q, V2")?;
    let p = Cmt2IvTheta { cl: theta[0], v1: theta[1], q: theta[2], v2: theta[3] };
    let n_obs = obs_times.len();
    let n_doses = dose_times.len();
    let mut y = [0.0f64; 2];
    let mut ys = [[0.0f64; 2]; 4];
    let mut out_s: Vec<Vec<f64>> = vec![vec![0.0; 2]; n_obs];
    let mut out_g: Vec<Vec<f64>> = vec![vec![0.0; 8]; n_obs];
    let mut obs_i = 0usize;
    let first_t = dose_times[0];
    while obs_i < n_obs && obs_times[obs_i] < first_t { obs_i += 1; }
    for dose_i in 0..n_doses {
        let td = dose_times[dose_i];
        let next_t = if dose_i + 1 < n_doses { dose_times[dose_i + 1] } else { f64::INFINITY };
        y[0] += dose_amts[dose_i];
        while obs_i < n_obs && obs_times[obs_i] <= td {
            out_s[obs_i] = y.to_vec(); out_g[obs_i] = flatten_sens_2d(&ys); obs_i += 1;
        }
        if obs_i >= n_obs && next_t.is_infinite() { break; }
        let mut solver = SolverSensi::new(LinearMultistepMethod::Bdf,
            rhs_2cmt_iv, sens_rhs_2cmt_iv, td, &y, &ys,
            1e-8, AbsTolerance::scalar(1e-10),
            SensiAbsTolerance::scalar([1e-10f64; 4]), p,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
        while obs_i < n_obs && (next_t.is_infinite() || obs_times[obs_i] < next_t) {
            let (_, y_new, ys_new) = solver.step(obs_times[obs_i], StepKind::Normal)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new;
            for (d, s) in ys.iter_mut().zip(ys_new.into_iter()) { *d = *s; }
            out_s[obs_i] = y.to_vec(); out_g[obs_i] = flatten_sens_2d(&ys); obs_i += 1;
        }
        if next_t.is_finite() {
            let (_, y_new, ys_new) = solver.step(next_t, StepKind::Normal)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new;
            for (d, s) in ys.iter_mut().zip(ys_new.into_iter()) { *d = *s; }
        }
    }
    Ok((out_s, out_g))
}

// ──────────────────────────────────────────────────────────────────────────────
// Template: 2-compartment oral  (KA, CL, V2, Q, V3)
// State:    A1 (depot), A2 (central), A3 (peripheral)
// IPRED:    A2 / V2         dose → A1
// k10=CL/V2, k12=Q/V2, k21=Q/V3
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "native-cvodes")]
#[derive(Clone, Copy)]
struct Cmt2OralTheta { ka: f64, cl: f64, v2: f64, q: f64, v3: f64 }

#[cfg(feature = "native-cvodes")]
fn rhs_2cmt_oral(_t: f64, y: &[f64; 3], dy: &mut [f64; 3], p: &Cmt2OralTheta) -> RhsResult {
    let k10 = p.cl / p.v2; let k12 = p.q / p.v2; let k21 = p.q / p.v3;
    *dy = [
        -p.ka * y[0],
        p.ka * y[0] - (k10 + k12) * y[1] + k21 * y[2],
        k12 * y[1] - k21 * y[2],
    ];
    RhsResult::Ok
}

/// Sensitivity RHS for 2-cmt oral.  Parameters: [KA, CL, V2, Q, V3].
#[cfg(feature = "native-cvodes")]
fn sens_rhs_2cmt_oral(
    _t: f64, y: &[f64; 3], _ydot: &[f64; 3],
    ys: [&[f64; 3]; 5], ysdot: [&mut [f64; 3]; 5],
    p: &Cmt2OralTheta,
) -> RhsResult {
    let (a1, a2, a3) = (y[0], y[1], y[2]);
    let k10 = p.cl / p.v2; let k12 = p.q / p.v2; let k21 = p.q / p.v3;
    let v2s = p.v2 * p.v2; let v3s = p.v3 * p.v3;
    // direct[j] = ∂f/∂θⱼ for each parameter
    let direct: [[f64; 3]; 5] = [
        [-a1, a1, 0.0],                                          // KA
        [0.0, -a2 / p.v2, 0.0],                                 // CL
        [0.0, (p.cl + p.q) * a2 / v2s, -p.q * a2 / v2s],       // V2
        [0.0, -a2 / p.v2 + a3 / p.v3, a2 / p.v2 - a3 / p.v3], // Q
        [0.0, -p.q * a3 / v3s, p.q * a3 / v3s],                 // V3
    ];
    let j11 = -(k10 + k12); let j22 = -k21;
    for (j, (s, sd)) in ys.into_iter().zip(ysdot.into_iter()).enumerate() {
        let d = direct[j];
        *sd = [
            -p.ka * s[0] + d[0],
            p.ka * s[0] + j11 * s[1] + k21 * s[2] + d[1],
            k12 * s[1] + j22 * s[2] + d[2],
        ];
    }
    RhsResult::Ok
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_2cmt_oral_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>, theta: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 5, &theta, "KA, CL, V2, Q, V3")?;
    let p = Cmt2OralTheta { ka: theta[0], cl: theta[1], v2: theta[2], q: theta[3], v3: theta[4] };
    let n_obs = obs_times.len();
    let n_doses = dose_times.len();
    let mut y = [0.0f64; 3];
    let mut out: Vec<Vec<f64>> = vec![vec![0.0; 3]; n_obs];
    let mut obs_i = 0usize;
    let first_t = dose_times[0];
    while obs_i < n_obs && obs_times[obs_i] < first_t { obs_i += 1; }
    for dose_i in 0..n_doses {
        let td = dose_times[dose_i];
        let next_t = if dose_i + 1 < n_doses { dose_times[dose_i + 1] } else { f64::INFINITY };
        y[0] += dose_amts[dose_i];
        while obs_i < n_obs && obs_times[obs_i] <= td { out[obs_i] = y.to_vec(); obs_i += 1; }
        if obs_i >= n_obs && next_t.is_infinite() { break; }
        let mut solver = SolverNoSensi::new(LinearMultistepMethod::Bdf, rhs_2cmt_oral, td, &y,
            1e-8, AbsTolerance::scalar(1e-10), p,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
        while obs_i < n_obs && (next_t.is_infinite() || obs_times[obs_i] < next_t) {
            let (_, y_new) = solver.step(obs_times[obs_i], StepKind::Normal)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new; out[obs_i] = y.to_vec(); obs_i += 1;
        }
        if next_t.is_finite() {
            let (_, y_new) = solver.step(next_t, StepKind::Normal)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new;
        }
    }
    Ok(out)
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_2cmt_oral_sensitivity_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>, theta: Vec<f64>,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 5, &theta, "KA, CL, V2, Q, V3")?;
    let p = Cmt2OralTheta { ka: theta[0], cl: theta[1], v2: theta[2], q: theta[3], v3: theta[4] };
    let n_obs = obs_times.len();
    let n_doses = dose_times.len();
    let mut y = [0.0f64; 3];
    let mut ys = [[0.0f64; 3]; 5];
    let mut out_s: Vec<Vec<f64>> = vec![vec![0.0; 3]; n_obs];
    let mut out_g: Vec<Vec<f64>> = vec![vec![0.0; 15]; n_obs];
    let mut obs_i = 0usize;
    let first_t = dose_times[0];
    while obs_i < n_obs && obs_times[obs_i] < first_t { obs_i += 1; }
    for dose_i in 0..n_doses {
        let td = dose_times[dose_i];
        let next_t = if dose_i + 1 < n_doses { dose_times[dose_i + 1] } else { f64::INFINITY };
        y[0] += dose_amts[dose_i];
        while obs_i < n_obs && obs_times[obs_i] <= td {
            out_s[obs_i] = y.to_vec(); out_g[obs_i] = flatten_sens_2d(&ys); obs_i += 1;
        }
        if obs_i >= n_obs && next_t.is_infinite() { break; }
        let mut solver = SolverSensi::new(LinearMultistepMethod::Bdf,
            rhs_2cmt_oral, sens_rhs_2cmt_oral, td, &y, &ys,
            1e-8, AbsTolerance::scalar(1e-10),
            SensiAbsTolerance::scalar([1e-10f64; 5]), p,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
        while obs_i < n_obs && (next_t.is_infinite() || obs_times[obs_i] < next_t) {
            let (_, y_new, ys_new) = solver.step(obs_times[obs_i], StepKind::Normal)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new;
            for (d, s) in ys.iter_mut().zip(ys_new.into_iter()) { *d = *s; }
            out_s[obs_i] = y.to_vec(); out_g[obs_i] = flatten_sens_2d(&ys); obs_i += 1;
        }
        if next_t.is_finite() {
            let (_, y_new, ys_new) = solver.step(next_t, StepKind::Normal)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new;
            for (d, s) in ys.iter_mut().zip(ys_new.into_iter()) { *d = *s; }
        }
    }
    Ok((out_s, out_g))
}

// ──────────────────────────────────────────────────────────────────────────────
// Template: 3-compartment IV  (CL, V1, Q2, V2, Q3, V3)
// State:    A1 (central), A2 (periph-1), A3 (periph-2)
// IPRED:    A1 / V1         dose → A1
// k10=CL/V1, k12=Q2/V1, k21=Q2/V2, k13=Q3/V1, k31=Q3/V3
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "native-cvodes")]
#[derive(Clone, Copy)]
struct Cmt3IvTheta { cl: f64, v1: f64, q2: f64, v2: f64, q3: f64, v3: f64 }

#[cfg(feature = "native-cvodes")]
fn rhs_3cmt_iv(_t: f64, y: &[f64; 3], dy: &mut [f64; 3], p: &Cmt3IvTheta) -> RhsResult {
    let k10 = p.cl / p.v1; let k12 = p.q2 / p.v1; let k21 = p.q2 / p.v2;
    let k13 = p.q3 / p.v1; let k31 = p.q3 / p.v3;
    *dy = [
        -(k10 + k12 + k13) * y[0] + k21 * y[1] + k31 * y[2],
        k12 * y[0] - k21 * y[1],
        k13 * y[0] - k31 * y[2],
    ];
    RhsResult::Ok
}

/// Sensitivity RHS for 3-cmt IV.  Parameters: [CL, V1, Q2, V2, Q3, V3].
#[cfg(feature = "native-cvodes")]
fn sens_rhs_3cmt_iv(
    _t: f64, y: &[f64; 3], _ydot: &[f64; 3],
    ys: [&[f64; 3]; 6], ysdot: [&mut [f64; 3]; 6],
    p: &Cmt3IvTheta,
) -> RhsResult {
    let (a1, a2, a3) = (y[0], y[1], y[2]);
    let k10 = p.cl / p.v1; let k12 = p.q2 / p.v1; let k21 = p.q2 / p.v2;
    let k13 = p.q3 / p.v1; let k31 = p.q3 / p.v3;
    let v1s = p.v1 * p.v1; let v2s = p.v2 * p.v2; let v3s = p.v3 * p.v3;
    let direct: [[f64; 3]; 6] = [
        [-a1 / p.v1, 0.0, 0.0],                                              // CL
        [(p.cl + p.q2 + p.q3) * a1 / v1s, -p.q2 * a1 / v1s, -p.q3 * a1 / v1s], // V1
        [-a1 / p.v1 + a2 / p.v2, a1 / p.v1 - a2 / p.v2, 0.0],              // Q2
        [-p.q2 * a2 / v2s, p.q2 * a2 / v2s, 0.0],                          // V2
        [-a1 / p.v1 + a3 / p.v3, 0.0, a1 / p.v1 - a3 / p.v3],              // Q3
        [-p.q3 * a3 / v3s, 0.0, p.q3 * a3 / v3s],                          // V3
    ];
    let j00 = -(k10 + k12 + k13); let j11 = -k21; let j22 = -k31;
    for (j, (s, sd)) in ys.into_iter().zip(ysdot.into_iter()).enumerate() {
        let d = direct[j];
        *sd = [
            j00 * s[0] + k21 * s[1] + k31 * s[2] + d[0],
            k12 * s[0] + j11 * s[1]               + d[1],
            k13 * s[0]               + j22 * s[2] + d[2],
        ];
    }
    RhsResult::Ok
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_3cmt_iv_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>, theta: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 6, &theta, "CL, V1, Q2, V2, Q3, V3")?;
    let p = Cmt3IvTheta { cl:theta[0], v1:theta[1], q2:theta[2], v2:theta[3], q3:theta[4], v3:theta[5] };
    let n_obs = obs_times.len();
    let n_doses = dose_times.len();
    let mut y = [0.0f64; 3];
    let mut out: Vec<Vec<f64>> = vec![vec![0.0; 3]; n_obs];
    let mut obs_i = 0usize;
    let first_t = dose_times[0];
    while obs_i < n_obs && obs_times[obs_i] < first_t { obs_i += 1; }
    for dose_i in 0..n_doses {
        let td = dose_times[dose_i];
        let next_t = if dose_i + 1 < n_doses { dose_times[dose_i + 1] } else { f64::INFINITY };
        y[0] += dose_amts[dose_i];
        while obs_i < n_obs && obs_times[obs_i] <= td { out[obs_i] = y.to_vec(); obs_i += 1; }
        if obs_i >= n_obs && next_t.is_infinite() { break; }
        let mut solver = SolverNoSensi::new(LinearMultistepMethod::Bdf, rhs_3cmt_iv, td, &y,
            1e-8, AbsTolerance::scalar(1e-10), p,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
        while obs_i < n_obs && (next_t.is_infinite() || obs_times[obs_i] < next_t) {
            let (_, y_new) = solver.step(obs_times[obs_i], StepKind::Normal)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new; out[obs_i] = y.to_vec(); obs_i += 1;
        }
        if next_t.is_finite() {
            let (_, y_new) = solver.step(next_t, StepKind::Normal)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new;
        }
    }
    Ok(out)
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_3cmt_iv_sensitivity_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>, theta: Vec<f64>,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 6, &theta, "CL, V1, Q2, V2, Q3, V3")?;
    let p = Cmt3IvTheta { cl:theta[0], v1:theta[1], q2:theta[2], v2:theta[3], q3:theta[4], v3:theta[5] };
    let n_obs = obs_times.len();
    let n_doses = dose_times.len();
    let mut y = [0.0f64; 3];
    let mut ys = [[0.0f64; 3]; 6];
    let mut out_s: Vec<Vec<f64>> = vec![vec![0.0; 3]; n_obs];
    let mut out_g: Vec<Vec<f64>> = vec![vec![0.0; 18]; n_obs];
    let mut obs_i = 0usize;
    let first_t = dose_times[0];
    while obs_i < n_obs && obs_times[obs_i] < first_t { obs_i += 1; }
    for dose_i in 0..n_doses {
        let td = dose_times[dose_i];
        let next_t = if dose_i + 1 < n_doses { dose_times[dose_i + 1] } else { f64::INFINITY };
        y[0] += dose_amts[dose_i];
        while obs_i < n_obs && obs_times[obs_i] <= td {
            out_s[obs_i] = y.to_vec(); out_g[obs_i] = flatten_sens_2d(&ys); obs_i += 1;
        }
        if obs_i >= n_obs && next_t.is_infinite() { break; }
        let mut solver = SolverSensi::new(LinearMultistepMethod::Bdf,
            rhs_3cmt_iv, sens_rhs_3cmt_iv, td, &y, &ys,
            1e-8, AbsTolerance::scalar(1e-10),
            SensiAbsTolerance::scalar([1e-10f64; 6]), p,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
        while obs_i < n_obs && (next_t.is_infinite() || obs_times[obs_i] < next_t) {
            let (_, y_new, ys_new) = solver.step(obs_times[obs_i], StepKind::Normal)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new;
            for (d, s) in ys.iter_mut().zip(ys_new.into_iter()) { *d = *s; }
            out_s[obs_i] = y.to_vec(); out_g[obs_i] = flatten_sens_2d(&ys); obs_i += 1;
        }
        if next_t.is_finite() {
            let (_, y_new, ys_new) = solver.step(next_t, StepKind::Normal)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new;
            for (d, s) in ys.iter_mut().zip(ys_new.into_iter()) { *d = *s; }
        }
    }
    Ok((out_s, out_g))
}


// ──────────────────────────────────────────────────────────────────────────────
// Template: 3-compartment oral  (KA, CL, V2, Q3, V3, Q4, V4)
// State:    A1 (depot), A2 (central), A3 (periph-1), A4 (periph-2)
// IPRED:    A2 / V2         dose → A1
// k10=CL/V2, k23=Q3/V2, k32=Q3/V3, k24=Q4/V2, k42=Q4/V4  [NONMEM ADVAN12]
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "native-cvodes")]
#[derive(Clone, Copy)]
struct Cmt3OralTheta { ka: f64, cl: f64, v2: f64, q3: f64, v3: f64, q4: f64, v4: f64 }

#[cfg(feature = "native-cvodes")]
fn rhs_3cmt_oral(_t: f64, y: &[f64; 4], dy: &mut [f64; 4], p: &Cmt3OralTheta) -> RhsResult {
    let k10 = p.cl / p.v2; let k23 = p.q3 / p.v2; let k32 = p.q3 / p.v3;
    let k24 = p.q4 / p.v2; let k42 = p.q4 / p.v4;
    *dy = [
        -p.ka * y[0],
        p.ka * y[0] - (k10 + k23 + k24) * y[1] + k32 * y[2] + k42 * y[3],
        k23 * y[1] - k32 * y[2],
        k24 * y[1] - k42 * y[3],
    ];
    RhsResult::Ok
}

#[cfg(feature = "native-cvodes")]
fn sens_rhs_3cmt_oral(
    _t: f64, y: &[f64; 4], _ydot: &[f64; 4],
    ys: [&[f64; 4]; 7], ysdot: [&mut [f64; 4]; 7], p: &Cmt3OralTheta,
) -> RhsResult {
    let (a1, a2, a3, a4) = (y[0], y[1], y[2], y[3]);
    let k10 = p.cl / p.v2; let k23 = p.q3 / p.v2; let k32 = p.q3 / p.v3;
    let k24 = p.q4 / p.v2; let k42 = p.q4 / p.v4;
    let v2s = p.v2 * p.v2; let v3s = p.v3 * p.v3; let v4s = p.v4 * p.v4;
    // direct[j] = ∂f/∂θⱼ  (params: KA, CL, V2, Q3, V3, Q4, V4)
    let direct: [[f64; 4]; 7] = [
        [-a1, a1, 0.0, 0.0],                                                            // KA
        [0.0, -a2 / p.v2, 0.0, 0.0],                                                   // CL
        [0.0, (p.cl + p.q3 + p.q4) * a2 / v2s, -p.q3 * a2 / v2s, -p.q4 * a2 / v2s],  // V2
        [0.0, -a2 / p.v2 + a3 / p.v3, a2 / p.v2 - a3 / p.v3, 0.0],                   // Q3
        [0.0, -p.q3 * a3 / v3s, p.q3 * a3 / v3s, 0.0],                                // V3
        [0.0, -a2 / p.v2 + a4 / p.v4, 0.0, a2 / p.v2 - a4 / p.v4],                   // Q4
        [0.0, -p.q4 * a4 / v4s, 0.0, p.q4 * a4 / v4s],                                // V4
    ];
    let j00 = -p.ka; let j11 = -(k10 + k23 + k24); let j22 = -k32; let j33 = -k42;
    for (j, (s, sd)) in ys.into_iter().zip(ysdot.into_iter()).enumerate() {
        let d = direct[j];
        *sd = [
            j00 * s[0] + d[0],
            p.ka * s[0] + j11 * s[1] + k32 * s[2] + k42 * s[3] + d[1],
            k23 * s[1] + j22 * s[2] + d[2],
            k24 * s[1] + j33 * s[3] + d[3],
        ];
    }
    RhsResult::Ok
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_3cmt_oral_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>, theta: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 7, &theta, "KA, CL, V2, Q3, V3, Q4, V4")?;
    let p = Cmt3OralTheta { ka:theta[0], cl:theta[1], v2:theta[2], q3:theta[3], v3:theta[4], q4:theta[5], v4:theta[6] };
    let n_obs = obs_times.len(); let n_doses = dose_times.len();
    let mut y = [0.0f64; 4];
    let mut out: Vec<Vec<f64>> = vec![vec![0.0; 4]; n_obs];
    let mut obs_i = 0usize;
    let first_t = dose_times[0];
    while obs_i < n_obs && obs_times[obs_i] < first_t { obs_i += 1; }
    for dose_i in 0..n_doses {
        let td = dose_times[dose_i]; let next_t = if dose_i + 1 < n_doses { dose_times[dose_i + 1] } else { f64::INFINITY };
        y[0] += dose_amts[dose_i];
        while obs_i < n_obs && obs_times[obs_i] <= td { out[obs_i] = y.to_vec(); obs_i += 1; }
        if obs_i >= n_obs && next_t.is_infinite() { break; }
        let mut solver = SolverNoSensi::new(LinearMultistepMethod::Bdf, rhs_3cmt_oral, td, &y,
            1e-8, AbsTolerance::scalar(1e-10), p,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
        while obs_i < n_obs && (next_t.is_infinite() || obs_times[obs_i] < next_t) {
            let (_, y_new) = solver.step(obs_times[obs_i], StepKind::Normal)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new; out[obs_i] = y.to_vec(); obs_i += 1;
        }
        if next_t.is_finite() { let (_, y_new) = solver.step(next_t, StepKind::Normal)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?; y = *y_new; }
    }
    Ok(out)
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_3cmt_oral_sensitivity_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>, theta: Vec<f64>,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 7, &theta, "KA, CL, V2, Q3, V3, Q4, V4")?;
    let p = Cmt3OralTheta { ka:theta[0], cl:theta[1], v2:theta[2], q3:theta[3], v3:theta[4], q4:theta[5], v4:theta[6] };
    let n_obs = obs_times.len(); let n_doses = dose_times.len();
    let mut y = [0.0f64; 4]; let mut ys = [[0.0f64; 4]; 7];
    let mut out_s: Vec<Vec<f64>> = vec![vec![0.0; 4]; n_obs];
    let mut out_g: Vec<Vec<f64>> = vec![vec![0.0; 28]; n_obs];   // 7 params × 4 states
    let mut obs_i = 0usize;
    let first_t = dose_times[0];
    while obs_i < n_obs && obs_times[obs_i] < first_t { obs_i += 1; }
    for dose_i in 0..n_doses {
        let td = dose_times[dose_i]; let next_t = if dose_i + 1 < n_doses { dose_times[dose_i + 1] } else { f64::INFINITY };
        y[0] += dose_amts[dose_i];
        while obs_i < n_obs && obs_times[obs_i] <= td {
            out_s[obs_i] = y.to_vec(); out_g[obs_i] = flatten_sens_2d(&ys); obs_i += 1;
        }
        if obs_i >= n_obs && next_t.is_infinite() { break; }
        let mut solver = SolverSensi::new(LinearMultistepMethod::Bdf,
            rhs_3cmt_oral, sens_rhs_3cmt_oral, td, &y, &ys,
            1e-8, AbsTolerance::scalar(1e-10), SensiAbsTolerance::scalar([1e-10f64; 7]), p,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
        while obs_i < n_obs && (next_t.is_infinite() || obs_times[obs_i] < next_t) {
            let (_, y_new, ys_new) = solver.step(obs_times[obs_i], StepKind::Normal)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new; for (d, s) in ys.iter_mut().zip(ys_new.into_iter()) { *d = *s; }
            out_s[obs_i] = y.to_vec(); out_g[obs_i] = flatten_sens_2d(&ys); obs_i += 1;
        }
        if next_t.is_finite() { let (_, y_new, ys_new) = solver.step(next_t, StepKind::Normal)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new; for (d, s) in ys.iter_mut().zip(ys_new.into_iter()) { *d = *s; } }
    }
    Ok((out_s, out_g))
}

// ──────────────────────────────────────────────────────────────────────────────
// Template: 4-compartment IV  (CL, V1, Q2, V2, Q3, V3, Q4, V4)
// State:    A1 (central), A2 (periph-1), A3 (periph-2), A4 (periph-3)
// IPRED:    A1 / V1         dose → A1
// k10=CL/V1, k12=Q2/V1, k21=Q2/V2, k13=Q3/V1, k31=Q3/V3, k14=Q4/V1, k41=Q4/V4
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "native-cvodes")]
#[derive(Clone, Copy)]
struct Cmt4IvTheta { cl: f64, v1: f64, q2: f64, v2: f64, q3: f64, v3: f64, q4: f64, v4: f64 }

#[cfg(feature = "native-cvodes")]
fn rhs_4cmt_iv(_t: f64, y: &[f64; 4], dy: &mut [f64; 4], p: &Cmt4IvTheta) -> RhsResult {
    let k10 = p.cl / p.v1; let k12 = p.q2 / p.v1; let k21 = p.q2 / p.v2;
    let k13 = p.q3 / p.v1; let k31 = p.q3 / p.v3;
    let k14 = p.q4 / p.v1; let k41 = p.q4 / p.v4;
    *dy = [
        -(k10 + k12 + k13 + k14) * y[0] + k21 * y[1] + k31 * y[2] + k41 * y[3],
        k12 * y[0] - k21 * y[1],
        k13 * y[0] - k31 * y[2],
        k14 * y[0] - k41 * y[3],
    ];
    RhsResult::Ok
}

#[cfg(feature = "native-cvodes")]
fn sens_rhs_4cmt_iv(
    _t: f64, y: &[f64; 4], _ydot: &[f64; 4],
    ys: [&[f64; 4]; 8], ysdot: [&mut [f64; 4]; 8], p: &Cmt4IvTheta,
) -> RhsResult {
    let (a1, a2, a3, a4) = (y[0], y[1], y[2], y[3]);
    let k10 = p.cl / p.v1; let k12 = p.q2 / p.v1; let k21 = p.q2 / p.v2;
    let k13 = p.q3 / p.v1; let k31 = p.q3 / p.v3;
    let k14 = p.q4 / p.v1; let k41 = p.q4 / p.v4;
    let v1s = p.v1 * p.v1; let v2s = p.v2 * p.v2; let v3s = p.v3 * p.v3; let v4s = p.v4 * p.v4;
    // direct[j] = ∂f/∂θⱼ  (params: CL, V1, Q2, V2, Q3, V3, Q4, V4)
    let direct: [[f64; 4]; 8] = [
        [-a1 / p.v1, 0.0, 0.0, 0.0],                                                                  // CL
        [(p.cl + p.q2 + p.q3 + p.q4) * a1 / v1s, -p.q2 * a1 / v1s, -p.q3 * a1 / v1s, -p.q4 * a1 / v1s], // V1
        [-a1 / p.v1 + a2 / p.v2, a1 / p.v1 - a2 / p.v2, 0.0, 0.0],                                  // Q2
        [-p.q2 * a2 / v2s, p.q2 * a2 / v2s, 0.0, 0.0],                                               // V2
        [-a1 / p.v1 + a3 / p.v3, 0.0, a1 / p.v1 - a3 / p.v3, 0.0],                                  // Q3
        [-p.q3 * a3 / v3s, 0.0, p.q3 * a3 / v3s, 0.0],                                               // V3
        [-a1 / p.v1 + a4 / p.v4, 0.0, 0.0, a1 / p.v1 - a4 / p.v4],                                  // Q4
        [-p.q4 * a4 / v4s, 0.0, 0.0, p.q4 * a4 / v4s],                                               // V4
    ];
    let j00 = -(k10 + k12 + k13 + k14); let j11 = -k21; let j22 = -k31; let j33 = -k41;
    for (j, (s, sd)) in ys.into_iter().zip(ysdot.into_iter()).enumerate() {
        let d = direct[j];
        *sd = [
            j00 * s[0] + k21 * s[1] + k31 * s[2] + k41 * s[3] + d[0],
            k12 * s[0] + j11 * s[1] + d[1],
            k13 * s[0] + j22 * s[2] + d[2],
            k14 * s[0] + j33 * s[3] + d[3],
        ];
    }
    RhsResult::Ok
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_4cmt_iv_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>, theta: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 8, &theta, "CL, V1, Q2, V2, Q3, V3, Q4, V4")?;
    let p = Cmt4IvTheta { cl:theta[0], v1:theta[1], q2:theta[2], v2:theta[3], q3:theta[4], v3:theta[5], q4:theta[6], v4:theta[7] };
    let n_obs = obs_times.len(); let n_doses = dose_times.len();
    let mut y = [0.0f64; 4]; let mut out: Vec<Vec<f64>> = vec![vec![0.0; 4]; n_obs];
    let mut obs_i = 0usize; let first_t = dose_times[0];
    while obs_i < n_obs && obs_times[obs_i] < first_t { obs_i += 1; }
    for dose_i in 0..n_doses {
        let td = dose_times[dose_i]; let next_t = if dose_i + 1 < n_doses { dose_times[dose_i + 1] } else { f64::INFINITY };
        y[0] += dose_amts[dose_i];
        while obs_i < n_obs && obs_times[obs_i] <= td { out[obs_i] = y.to_vec(); obs_i += 1; }
        if obs_i >= n_obs && next_t.is_infinite() { break; }
        let mut solver = SolverNoSensi::new(LinearMultistepMethod::Bdf, rhs_4cmt_iv, td, &y,
            1e-8, AbsTolerance::scalar(1e-10), p,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
        while obs_i < n_obs && (next_t.is_infinite() || obs_times[obs_i] < next_t) {
            let (_, y_new) = solver.step(obs_times[obs_i], StepKind::Normal)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new; out[obs_i] = y.to_vec(); obs_i += 1;
        }
        if next_t.is_finite() { let (_, y_new) = solver.step(next_t, StepKind::Normal)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?; y = *y_new; }
    }
    Ok(out)
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_4cmt_iv_sensitivity_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>, theta: Vec<f64>,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 8, &theta, "CL, V1, Q2, V2, Q3, V3, Q4, V4")?;
    let p = Cmt4IvTheta { cl:theta[0], v1:theta[1], q2:theta[2], v2:theta[3], q3:theta[4], v3:theta[5], q4:theta[6], v4:theta[7] };
    let n_obs = obs_times.len(); let n_doses = dose_times.len();
    let mut y = [0.0f64; 4]; let mut ys = [[0.0f64; 4]; 8];
    let mut out_s: Vec<Vec<f64>> = vec![vec![0.0; 4]; n_obs];
    let mut out_g: Vec<Vec<f64>> = vec![vec![0.0; 32]; n_obs];   // 8 params × 4 states
    let mut obs_i = 0usize; let first_t = dose_times[0];
    while obs_i < n_obs && obs_times[obs_i] < first_t { obs_i += 1; }
    for dose_i in 0..n_doses {
        let td = dose_times[dose_i]; let next_t = if dose_i + 1 < n_doses { dose_times[dose_i + 1] } else { f64::INFINITY };
        y[0] += dose_amts[dose_i];
        while obs_i < n_obs && obs_times[obs_i] <= td {
            out_s[obs_i] = y.to_vec(); out_g[obs_i] = flatten_sens_2d(&ys); obs_i += 1;
        }
        if obs_i >= n_obs && next_t.is_infinite() { break; }
        let mut solver = SolverSensi::new(LinearMultistepMethod::Bdf,
            rhs_4cmt_iv, sens_rhs_4cmt_iv, td, &y, &ys,
            1e-8, AbsTolerance::scalar(1e-10), SensiAbsTolerance::scalar([1e-10f64; 8]), p,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
        while obs_i < n_obs && (next_t.is_infinite() || obs_times[obs_i] < next_t) {
            let (_, y_new, ys_new) = solver.step(obs_times[obs_i], StepKind::Normal)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new; for (d, s) in ys.iter_mut().zip(ys_new.into_iter()) { *d = *s; }
            out_s[obs_i] = y.to_vec(); out_g[obs_i] = flatten_sens_2d(&ys); obs_i += 1;
        }
        if next_t.is_finite() { let (_, y_new, ys_new) = solver.step(next_t, StepKind::Normal)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new; for (d, s) in ys.iter_mut().zip(ys_new.into_iter()) { *d = *s; } }
    }
    Ok((out_s, out_g))
}

// ──────────────────────────────────────────────────────────────────────────────
// Template: 4-compartment oral  (KA, CL, V2, Q3, V3, Q4, V4, Q5, V5)
// State:    A1 (depot), A2 (central), A3 (periph-1), A4 (periph-2), A5 (periph-3)
// IPRED:    A2 / V2         dose → A1
// k10=CL/V2, k23=Q3/V2, k32=Q3/V3, k24=Q4/V2, k42=Q4/V4, k25=Q5/V2, k52=Q5/V5
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "native-cvodes")]
#[derive(Clone, Copy)]
struct Cmt4OralTheta { ka: f64, cl: f64, v2: f64, q3: f64, v3: f64, q4: f64, v4: f64, q5: f64, v5: f64 }

#[cfg(feature = "native-cvodes")]
fn rhs_4cmt_oral(_t: f64, y: &[f64; 5], dy: &mut [f64; 5], p: &Cmt4OralTheta) -> RhsResult {
    let k10 = p.cl / p.v2; let k23 = p.q3 / p.v2; let k32 = p.q3 / p.v3;
    let k24 = p.q4 / p.v2; let k42 = p.q4 / p.v4;
    let k25 = p.q5 / p.v2; let k52 = p.q5 / p.v5;
    *dy = [
        -p.ka * y[0],
        p.ka * y[0] - (k10 + k23 + k24 + k25) * y[1] + k32 * y[2] + k42 * y[3] + k52 * y[4],
        k23 * y[1] - k32 * y[2],
        k24 * y[1] - k42 * y[3],
        k25 * y[1] - k52 * y[4],
    ];
    RhsResult::Ok
}

#[cfg(feature = "native-cvodes")]
fn sens_rhs_4cmt_oral(
    _t: f64, y: &[f64; 5], _ydot: &[f64; 5],
    ys: [&[f64; 5]; 9], ysdot: [&mut [f64; 5]; 9], p: &Cmt4OralTheta,
) -> RhsResult {
    let (a1, a2, a3, a4, a5) = (y[0], y[1], y[2], y[3], y[4]);
    let k10 = p.cl / p.v2; let k23 = p.q3 / p.v2; let k32 = p.q3 / p.v3;
    let k24 = p.q4 / p.v2; let k42 = p.q4 / p.v4;
    let k25 = p.q5 / p.v2; let k52 = p.q5 / p.v5;
    let v2s = p.v2 * p.v2; let v3s = p.v3 * p.v3; let v4s = p.v4 * p.v4; let v5s = p.v5 * p.v5;
    // direct[j] = ∂f/∂θⱼ  (params: KA, CL, V2, Q3, V3, Q4, V4, Q5, V5)
    let direct: [[f64; 5]; 9] = [
        [-a1, a1, 0.0, 0.0, 0.0],                                                                                    // KA
        [0.0, -a2 / p.v2, 0.0, 0.0, 0.0],                                                                            // CL
        [0.0, (p.cl + p.q3 + p.q4 + p.q5) * a2 / v2s, -p.q3 * a2 / v2s, -p.q4 * a2 / v2s, -p.q5 * a2 / v2s],     // V2
        [0.0, -a2 / p.v2 + a3 / p.v3, a2 / p.v2 - a3 / p.v3, 0.0, 0.0],                                            // Q3
        [0.0, -p.q3 * a3 / v3s, p.q3 * a3 / v3s, 0.0, 0.0],                                                         // V3
        [0.0, -a2 / p.v2 + a4 / p.v4, 0.0, a2 / p.v2 - a4 / p.v4, 0.0],                                            // Q4
        [0.0, -p.q4 * a4 / v4s, 0.0, p.q4 * a4 / v4s, 0.0],                                                         // V4
        [0.0, -a2 / p.v2 + a5 / p.v5, 0.0, 0.0, a2 / p.v2 - a5 / p.v5],                                            // Q5
        [0.0, -p.q5 * a5 / v5s, 0.0, 0.0, p.q5 * a5 / v5s],                                                         // V5
    ];
    let j00 = -p.ka; let j11 = -(k10 + k23 + k24 + k25);
    let j22 = -k32; let j33 = -k42; let j44 = -k52;
    for (j, (s, sd)) in ys.into_iter().zip(ysdot.into_iter()).enumerate() {
        let d = direct[j];
        *sd = [
            j00 * s[0] + d[0],
            p.ka * s[0] + j11 * s[1] + k32 * s[2] + k42 * s[3] + k52 * s[4] + d[1],
            k23 * s[1] + j22 * s[2] + d[2],
            k24 * s[1] + j33 * s[3] + d[3],
            k25 * s[1] + j44 * s[4] + d[4],
        ];
    }
    RhsResult::Ok
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_4cmt_oral_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>, theta: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 9, &theta, "KA, CL, V2, Q3, V3, Q4, V4, Q5, V5")?;
    let p = Cmt4OralTheta { ka:theta[0], cl:theta[1], v2:theta[2], q3:theta[3], v3:theta[4], q4:theta[5], v4:theta[6], q5:theta[7], v5:theta[8] };
    let n_obs = obs_times.len(); let n_doses = dose_times.len();
    let mut y = [0.0f64; 5]; let mut out: Vec<Vec<f64>> = vec![vec![0.0; 5]; n_obs];
    let mut obs_i = 0usize; let first_t = dose_times[0];
    while obs_i < n_obs && obs_times[obs_i] < first_t { obs_i += 1; }
    for dose_i in 0..n_doses {
        let td = dose_times[dose_i]; let next_t = if dose_i + 1 < n_doses { dose_times[dose_i + 1] } else { f64::INFINITY };
        y[0] += dose_amts[dose_i];
        while obs_i < n_obs && obs_times[obs_i] <= td { out[obs_i] = y.to_vec(); obs_i += 1; }
        if obs_i >= n_obs && next_t.is_infinite() { break; }
        let mut solver = SolverNoSensi::new(LinearMultistepMethod::Bdf, rhs_4cmt_oral, td, &y,
            1e-8, AbsTolerance::scalar(1e-10), p,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
        while obs_i < n_obs && (next_t.is_infinite() || obs_times[obs_i] < next_t) {
            let (_, y_new) = solver.step(obs_times[obs_i], StepKind::Normal)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new; out[obs_i] = y.to_vec(); obs_i += 1;
        }
        if next_t.is_finite() { let (_, y_new) = solver.step(next_t, StepKind::Normal)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?; y = *y_new; }
    }
    Ok(out)
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_4cmt_oral_sensitivity_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>, theta: Vec<f64>,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 9, &theta, "KA, CL, V2, Q3, V3, Q4, V4, Q5, V5")?;
    let p = Cmt4OralTheta { ka:theta[0], cl:theta[1], v2:theta[2], q3:theta[3], v3:theta[4], q4:theta[5], v4:theta[6], q5:theta[7], v5:theta[8] };
    let n_obs = obs_times.len(); let n_doses = dose_times.len();
    let mut y = [0.0f64; 5]; let mut ys = [[0.0f64; 5]; 9];
    let mut out_s: Vec<Vec<f64>> = vec![vec![0.0; 5]; n_obs];
    let mut out_g: Vec<Vec<f64>> = vec![vec![0.0; 45]; n_obs];   // 9 params × 5 states
    let mut obs_i = 0usize; let first_t = dose_times[0];
    while obs_i < n_obs && obs_times[obs_i] < first_t { obs_i += 1; }
    for dose_i in 0..n_doses {
        let td = dose_times[dose_i]; let next_t = if dose_i + 1 < n_doses { dose_times[dose_i + 1] } else { f64::INFINITY };
        y[0] += dose_amts[dose_i];
        while obs_i < n_obs && obs_times[obs_i] <= td {
            out_s[obs_i] = y.to_vec(); out_g[obs_i] = flatten_sens_2d(&ys); obs_i += 1;
        }
        if obs_i >= n_obs && next_t.is_infinite() { break; }
        let mut solver = SolverSensi::new(LinearMultistepMethod::Bdf,
            rhs_4cmt_oral, sens_rhs_4cmt_oral, td, &y, &ys,
            1e-8, AbsTolerance::scalar(1e-10), SensiAbsTolerance::scalar([1e-10f64; 9]), p,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
        while obs_i < n_obs && (next_t.is_infinite() || obs_times[obs_i] < next_t) {
            let (_, y_new, ys_new) = solver.step(obs_times[obs_i], StepKind::Normal)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new; for (d, s) in ys.iter_mut().zip(ys_new.into_iter()) { *d = *s; }
            out_s[obs_i] = y.to_vec(); out_g[obs_i] = flatten_sens_2d(&ys); obs_i += 1;
        }
        if next_t.is_finite() { let (_, y_new, ys_new) = solver.step(next_t, StepKind::Normal)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new; for (d, s) in ys.iter_mut().zip(ys_new.into_iter()) { *d = *s; } }
    }
    Ok((out_s, out_g))
}

// ── module registration ───────────────────────────────────────────────────────

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(neg2ll_obs_loop, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_linear_probe, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_advan6_mixed_pkpd_probe, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_advan6_mixed_pkpd_repeat_probe, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_advan6_mixed_pkpd_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(
        native_cvodes_advan6_mixed_pkpd_sensitivity_probe_multidose, m
    )?)?;
    // ── new ODE template functions ────────────────────────────────────────────
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_1cmt_iv_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_1cmt_iv_sensitivity_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_1cmt_oral_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_1cmt_oral_sensitivity_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_2cmt_iv_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_2cmt_iv_sensitivity_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_2cmt_oral_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_2cmt_oral_sensitivity_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_3cmt_iv_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_3cmt_iv_sensitivity_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_3cmt_oral_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_3cmt_oral_sensitivity_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_4cmt_iv_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_4cmt_iv_sensitivity_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_4cmt_oral_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_4cmt_oral_sensitivity_probe_multidose, m)?)?;
    Ok(())
}
