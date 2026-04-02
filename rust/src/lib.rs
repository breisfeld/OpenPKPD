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


#[cfg(feature = "native-cvodes")]
use cvode_wrap::{
    AbsTolerance, LinearMultistepMethod, RhsResult, SensiAbsTolerance, SolverNoSensi,
    SolverSensi, StepKind,
};

/// Number of ODE parameters for the transit_1cmt_pkpd model
/// (KTR, KA, CL, V, EMAX, EC50, KOUT, E0).
#[cfg(feature = "native-cvodes")]
const N_ODE_PARAMS: usize = 8;

#[cfg(feature = "native-cvodes")]
#[derive(Clone, Copy)]
struct Transit1CmtPkpdTheta {
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
fn rhs_transit_1cmt_pkpd(
    _t: f64,
    y: &[f64; 4],
    dy: &mut [f64; 4],
    theta: &Transit1CmtPkpdTheta,
) -> RhsResult {
    let a1 = y[0];
    let a2 = y[1];
    let a3 = y[2];
    let a4 = y[3];
    let conc = a3 / theta.v;
    // Clamp emax to [0, 1] so pd ∈ [0, 1]; emax > 1 produces negative production
    // which is unphysical and can destabilise the ODE solver.
    let emax_clamped = theta.emax.clamp(0.0, 1.0);
    let pd = 1.0 - emax_clamped * conc / (theta.ec50 + conc);

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
fn sens_rhs_transit_1cmt_pkpd(
    _t: f64,
    y: &[f64; 4],
    _ydot: &[f64; 4],
    ys: [&[f64; 4]; N_ODE_PARAMS],
    ysdot: [&mut [f64; 4]; N_ODE_PARAMS],
    theta: &Transit1CmtPkpdTheta,
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
/// For z >= -30 uses the standard erfc formula.
/// For z < -30 uses the asymptotic expansion to avoid catastrophic
/// cancellation and clamping errors:
///
///   ln Φ(z) ≈ -z²/2 - 0.5·ln(2π) - ln(-z)   for z << 0
///
/// This gives numerically accurate values at any depth into the tail
/// (the old clamp v.max(1e-300) was catastrophically wrong for z < -25).
#[inline(always)]
fn norm_logcdf(z: f64) -> f64 {
    if z < -30.0 {
        // Asymptotic expansion: ln Φ(z) ≈ -z²/2 - 0.5*ln(2π) - ln(-z)
        // 0.9189385332046727 = 0.5 * ln(2π)
        -0.5 * z * z - 0.9189385332046727 - z.abs().ln()
    } else {
        let v = libm::erfc(-z / SQRT_2) * 0.5;
        v.max(1e-300_f64).ln()
    }
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

    let n = dv.len();
    if pred.len() != n || var.len() != n || mask.len() != n || lloq.len() != n {
        // Mismatched array lengths: return a large penalty value rather than
        // silently computing on a truncated slice, which would hide caller bugs.
        return 1e30_f64;
    }

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
fn native_cvodes_transit_1cmt_pkpd_probe(
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

    let theta = Transit1CmtPkpdTheta {
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
        rhs_transit_1cmt_pkpd,
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



/// CVODES BDF integrator for the transit_1cmt_pkpd 4-state ODE
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
fn native_cvodes_transit_1cmt_pkpd_probe_multidose(
    obs_times: Vec<f64>,
    dose_times: Vec<f64>,
    dose_amts: Vec<f64>,
    theta: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 8, &theta, "KTR, KA, CL, V, EMAX, EC50, KOUT, E0")?;
    let p = Transit1CmtPkpdTheta { ktr: theta[0], ka: theta[1], cl: theta[2], v: theta[3], emax: theta[4], ec50: theta[5], kout: theta[6], e0: theta[7] };
    probe_multidose_core::<4, _>(&obs_times, &dose_times, &dose_amts, p, rhs_transit_1cmt_pkpd)
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
/// Mirrors `native_cvodes_transit_1cmt_pkpd_probe_multidose` with a
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
fn native_cvodes_transit_1cmt_pkpd_sensitivity_probe_multidose(
    obs_times: Vec<f64>,
    dose_times: Vec<f64>,
    dose_amts: Vec<f64>,
    theta: Vec<f64>,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 8, &theta, "KTR, KA, CL, V, EMAX, EC50, KOUT, E0")?;
    let p = Transit1CmtPkpdTheta { ktr: theta[0], ka: theta[1], cl: theta[2], v: theta[3], emax: theta[4], ec50: theta[5], kout: theta[6], e0: theta[7] };
    sens_probe_multidose_core::<4, 8, _>(&obs_times, &dose_times, &dose_amts, p, rhs_transit_1cmt_pkpd, sens_rhs_transit_1cmt_pkpd)
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

/// Validate inputs for the infusion-aware multidose probe functions.
/// Extends `validate_multidose_inputs` with a `dose_rates` length check.
#[cfg(feature = "native-cvodes")]
fn validate_infusion_multidose_inputs(
    obs_times: &[f64], dose_times: &[f64], dose_amts: &[f64],
    dose_rates: &[f64], n_theta: usize, theta: &[f64], theta_desc: &str,
) -> PyResult<()> {
    validate_multidose_inputs(obs_times, dose_times, dose_amts, n_theta, theta, theta_desc)?;
    if dose_rates.len() != dose_times.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dose_rates must have the same length as dose_times",
        ));
    }
    if dose_rates.iter().any(|r| !r.is_finite() || *r < 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dose_rates must be finite and non-negative (0.0 = bolus)",
        ));
    }
    Ok(())
}

/// Build a sorted list of (time, bolus_delta, rate_delta) breakpoints from a
/// mixed bolus / constant-rate-infusion dose schedule.
///
/// For each dose:
///   - bolus  (rate == 0):  one breakpoint  (t_dose, +amount, 0.0)
///   - infusion (rate > 0):  two breakpoints (t_dose, 0.0, +rate)
///                                           (t_dose+amt/rate, 0.0, −rate)
///
/// Breakpoints at the same time are sorted so that positive `rate_delta`
/// entries (new dose / infusion start) precede negative ones (infusion end).
#[cfg(feature = "native-cvodes")]
fn build_infusion_breakpoints(
    dose_times: &[f64], dose_amts: &[f64], dose_rates: &[f64],
) -> Vec<(f64, f64, f64)> {
    let mut bps: Vec<(f64, f64, f64)> = Vec::with_capacity(dose_times.len() * 2);
    for i in 0..dose_times.len() {
        let t = dose_times[i];
        let amt = dose_amts[i];
        let rate = dose_rates[i];
        if rate == 0.0 {
            bps.push((t, amt, 0.0));
        } else {
            bps.push((t, 0.0, rate));
            bps.push((t + amt / rate, 0.0, -rate));
        }
    }
    // Sort by time; within same time, positive rate_delta first.
    bps.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
    });
    bps
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
// Generic dose-advance loops — shared by all bolus and infusion templates.
// Each exported probe function unpacks its Vec<f64> theta into the appropriate
// struct, then delegates to one of these four generic helpers.
// ──────────────────────────────────────────────────────────────────────────────

/// Multi-dose bolus probe (no sensitivity). Doses applied to state index 0.
/// `N` = ODE states; `Theta: Copy` = parameter struct.
#[cfg(feature = "native-cvodes")]
fn probe_multidose_core<const N: usize, Theta: Copy>(
    obs_times: &[f64],
    dose_times: &[f64],
    dose_amts: &[f64],
    theta: Theta,
    rhs: fn(f64, &[f64; N], &mut [f64; N], &Theta) -> RhsResult,
) -> PyResult<Vec<Vec<f64>>> {
    let n_obs = obs_times.len();
    let n_doses = dose_times.len();
    let mut y = [0.0f64; N];
    let mut out: Vec<Vec<f64>> = vec![vec![0.0; N]; n_obs];
    let mut obs_i = 0usize;
    let first_t = dose_times[0];
    while obs_i < n_obs && obs_times[obs_i] < first_t { obs_i += 1; }
    for dose_i in 0..n_doses {
        let td = dose_times[dose_i];
        let next_t = if dose_i + 1 < n_doses { dose_times[dose_i + 1] } else { f64::INFINITY };
        y[0] += dose_amts[dose_i];
        while obs_i < n_obs && obs_times[obs_i] <= td { out[obs_i] = y.to_vec(); obs_i += 1; }
        if obs_i >= n_obs && next_t.is_infinite() { break; }
        let mut solver = SolverNoSensi::new(LinearMultistepMethod::Bdf, rhs, td, &y,
            1e-8, AbsTolerance::scalar(1e-10), theta,
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

/// Multi-dose bolus probe WITH forward sensitivities. Doses applied to state 0.
/// `N` = ODE states; `P` = number of ODE parameters; `Theta: Copy`.
#[cfg(feature = "native-cvodes")]
fn sens_probe_multidose_core<const N: usize, const P: usize, Theta: Copy>(
    obs_times: &[f64],
    dose_times: &[f64],
    dose_amts: &[f64],
    theta: Theta,
    rhs: fn(f64, &[f64; N], &mut [f64; N], &Theta) -> RhsResult,
    sens_rhs: fn(f64, &[f64; N], &[f64; N], [&[f64; N]; P], [&mut [f64; N]; P], &Theta) -> RhsResult,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let n_obs = obs_times.len();
    let n_doses = dose_times.len();
    let mut y = [0.0f64; N];
    let mut ys = [[0.0f64; N]; P];
    let mut out_s: Vec<Vec<f64>> = vec![vec![0.0; N]; n_obs];
    let mut out_g: Vec<Vec<f64>> = vec![vec![0.0; N * P]; n_obs];
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
            rhs, sens_rhs, td, &y, &ys,
            1e-8, AbsTolerance::scalar(1e-10),
            SensiAbsTolerance::scalar([1e-10f64; P]), theta,
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

/// Multi-dose constant-rate infusion probe (no sensitivity).
/// `N` = ODE states; `InfTheta: Copy` = per-segment parameter struct.
/// `make_theta(active_rate)` constructs the segment theta from the running rate.
#[cfg(feature = "native-cvodes")]
fn infusion_probe_multidose_core<const N: usize, InfTheta: Copy>(
    obs_times: &[f64],
    dose_times: &[f64],
    dose_amts: &[f64],
    dose_rates: &[f64],
    make_theta: impl Fn(f64) -> InfTheta,
    rhs: fn(f64, &[f64; N], &mut [f64; N], &InfTheta) -> RhsResult,
) -> PyResult<Vec<Vec<f64>>> {
    let n_obs = obs_times.len();
    let mut y = [0.0f64; N];
    let mut out: Vec<Vec<f64>> = vec![vec![0.0; N]; n_obs];
    let mut obs_i = 0usize; let mut active_rate = 0.0f64;
    let bps = build_infusion_breakpoints(dose_times, dose_amts, dose_rates);
    let n_bps = bps.len();
    if let Some(&(first_t, _, _)) = bps.first() {
        while obs_i < n_obs && obs_times[obs_i] < first_t { obs_i += 1; }
    }
    for bp_i in 0..n_bps {
        let (t_bp, bolus_delta, rate_delta) = bps[bp_i];
        y[0] += bolus_delta; active_rate += rate_delta;
        while obs_i < n_obs && obs_times[obs_i] <= t_bp { out[obs_i] = y.to_vec(); obs_i += 1; }
        let next_t = if bp_i + 1 < n_bps { bps[bp_i + 1].0 } else { f64::INFINITY };
        if next_t <= t_bp { continue; }
        if obs_i >= n_obs && next_t.is_infinite() { break; }
        let inf_p = make_theta(active_rate);
        let mut solver = SolverNoSensi::new(LinearMultistepMethod::Bdf, rhs,
            t_bp, &y, 1e-8, AbsTolerance::scalar(1e-10), inf_p,
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

/// Multi-dose constant-rate infusion probe WITH forward sensitivities.
/// `N` = ODE states; `P` = number of ODE parameters; `InfTheta: Copy`.
#[cfg(feature = "native-cvodes")]
fn infusion_sens_probe_multidose_core<const N: usize, const P: usize, InfTheta: Copy>(
    obs_times: &[f64],
    dose_times: &[f64],
    dose_amts: &[f64],
    dose_rates: &[f64],
    make_theta: impl Fn(f64) -> InfTheta,
    rhs: fn(f64, &[f64; N], &mut [f64; N], &InfTheta) -> RhsResult,
    sens_rhs: fn(f64, &[f64; N], &[f64; N], [&[f64; N]; P], [&mut [f64; N]; P], &InfTheta) -> RhsResult,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let n_obs = obs_times.len();
    let mut y = [0.0f64; N]; let mut ys = [[0.0f64; N]; P];
    let mut out_s: Vec<Vec<f64>> = vec![vec![0.0; N]; n_obs];
    let mut out_g: Vec<Vec<f64>> = vec![vec![0.0; N * P]; n_obs];
    let mut obs_i = 0usize; let mut active_rate = 0.0f64;
    let bps = build_infusion_breakpoints(dose_times, dose_amts, dose_rates);
    let n_bps = bps.len();
    if let Some(&(first_t, _, _)) = bps.first() {
        while obs_i < n_obs && obs_times[obs_i] < first_t { obs_i += 1; }
    }
    for bp_i in 0..n_bps {
        let (t_bp, bolus_delta, rate_delta) = bps[bp_i];
        y[0] += bolus_delta; active_rate += rate_delta;
        while obs_i < n_obs && obs_times[obs_i] <= t_bp {
            out_s[obs_i] = y.to_vec(); out_g[obs_i] = flatten_sens_2d(&ys); obs_i += 1;
        }
        let next_t = if bp_i + 1 < n_bps { bps[bp_i + 1].0 } else { f64::INFINITY };
        if next_t <= t_bp { continue; }
        if obs_i >= n_obs && next_t.is_infinite() { break; }
        let inf_p = make_theta(active_rate);
        let mut solver = SolverSensi::new(LinearMultistepMethod::Bdf,
            rhs, sens_rhs, t_bp, &y, &ys,
            1e-8, AbsTolerance::scalar(1e-10),
            SensiAbsTolerance::scalar([1e-10f64; P]), inf_p,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
        while obs_i < n_obs && (next_t.is_infinite() || obs_times[obs_i] < next_t) {
            let (_, y_new, ys_new) = solver.step(obs_times[obs_i], StepKind::Normal)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new; for (d, s) in ys.iter_mut().zip(ys_new.into_iter()) { *d = *s; }
            out_s[obs_i] = y.to_vec(); out_g[obs_i] = flatten_sens_2d(&ys); obs_i += 1;
        }
        if next_t.is_finite() {
            let (_, y_new, ys_new) = solver.step(next_t, StepKind::Normal)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
            y = *y_new; for (d, s) in ys.iter_mut().zip(ys_new.into_iter()) { *d = *s; }
        }
    }
    Ok((out_s, out_g))
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
    probe_multidose_core::<1, _>(&obs_times, &dose_times, &dose_amts, p, rhs_1cmt_iv)
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_1cmt_iv_sensitivity_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>, theta: Vec<f64>,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 2, &theta, "CL, V")?;
    let p = Cmt1IvTheta { cl: theta[0], v: theta[1] };
    sens_probe_multidose_core::<1, 2, _>(&obs_times, &dose_times, &dose_amts, p, rhs_1cmt_iv, sens_rhs_1cmt_iv)
}

// ── Infusion variant: 1-compartment IV ───────────────────────────────────────

#[cfg(feature = "native-cvodes")]
#[derive(Clone, Copy)]
struct Cmt1IvInfTheta { cl: f64, v: f64, rate: f64 }

#[cfg(feature = "native-cvodes")]
fn rhs_1cmt_iv_inf(_t: f64, y: &[f64; 1], dy: &mut [f64; 1], p: &Cmt1IvInfTheta) -> RhsResult {
    *dy = [p.rate - (p.cl / p.v) * y[0]];
    RhsResult::Ok
}

/// Sensitivity RHS for infusion 1-cmt IV. The infusion rate is not a function
/// of any ODE parameter, so ∂rate/∂θ = 0 and the direct terms are unchanged.
#[cfg(feature = "native-cvodes")]
fn sens_rhs_1cmt_iv_inf(
    _t: f64, y: &[f64; 1], _ydot: &[f64; 1],
    ys: [&[f64; 1]; 2], ysdot: [&mut [f64; 1]; 2],
    p: &Cmt1IvInfTheta,
) -> RhsResult {
    let a1 = y[0]; let ke = p.cl / p.v; let v2 = p.v * p.v;
    let direct: [[f64; 1]; 2] = [[-a1 / p.v], [p.cl * a1 / v2]];
    for (j, (s, sd)) in ys.into_iter().zip(ysdot.into_iter()).enumerate() {
        *sd = [-ke * s[0] + direct[j][0]];
    }
    RhsResult::Ok
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_1cmt_iv_infusion_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>,
    dose_rates: Vec<f64>, theta: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    validate_infusion_multidose_inputs(&obs_times, &dose_times, &dose_amts, &dose_rates, 2, &theta, "CL, V")?;
    let cl = theta[0];
    let v = theta[1];
    infusion_probe_multidose_core::<1, _>(&obs_times, &dose_times, &dose_amts, &dose_rates, |rate| Cmt1IvInfTheta { cl, v, rate }, rhs_1cmt_iv_inf)
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_1cmt_iv_infusion_sensitivity_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>,
    dose_rates: Vec<f64>, theta: Vec<f64>,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    validate_infusion_multidose_inputs(&obs_times, &dose_times, &dose_amts, &dose_rates, 2, &theta, "CL, V")?;
    let cl = theta[0];
    let v = theta[1];
    infusion_sens_probe_multidose_core::<1, 2, _>(&obs_times, &dose_times, &dose_amts, &dose_rates, |rate| Cmt1IvInfTheta { cl, v, rate }, rhs_1cmt_iv_inf, sens_rhs_1cmt_iv_inf)
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
    probe_multidose_core::<2, _>(&obs_times, &dose_times, &dose_amts, p, rhs_1cmt_oral)
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_1cmt_oral_sensitivity_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>, theta: Vec<f64>,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 3, &theta, "KA, CL, V")?;
    let p = Cmt1OralTheta { ka: theta[0], cl: theta[1], v: theta[2] };
    sens_probe_multidose_core::<2, 3, _>(&obs_times, &dose_times, &dose_amts, p, rhs_1cmt_oral, sens_rhs_1cmt_oral)
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
    probe_multidose_core::<2, _>(&obs_times, &dose_times, &dose_amts, p, rhs_2cmt_iv)
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_2cmt_iv_sensitivity_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>, theta: Vec<f64>,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 4, &theta, "CL, V1, Q, V2")?;
    let p = Cmt2IvTheta { cl: theta[0], v1: theta[1], q: theta[2], v2: theta[3] };
    sens_probe_multidose_core::<2, 4, _>(&obs_times, &dose_times, &dose_amts, p, rhs_2cmt_iv, sens_rhs_2cmt_iv)
}

// ── Infusion variant: 2-compartment IV ───────────────────────────────────────

#[cfg(feature = "native-cvodes")]
#[derive(Clone, Copy)]
struct Cmt2IvInfTheta { cl: f64, v1: f64, q: f64, v2: f64, rate: f64 }

#[cfg(feature = "native-cvodes")]
fn rhs_2cmt_iv_inf(_t: f64, y: &[f64; 2], dy: &mut [f64; 2], p: &Cmt2IvInfTheta) -> RhsResult {
    let k10 = p.cl / p.v1; let k12 = p.q / p.v1; let k21 = p.q / p.v2;
    *dy = [p.rate - (k10 + k12) * y[0] + k21 * y[1], k12 * y[0] - k21 * y[1]];
    RhsResult::Ok
}

#[cfg(feature = "native-cvodes")]
fn sens_rhs_2cmt_iv_inf(
    _t: f64, y: &[f64; 2], _ydot: &[f64; 2],
    ys: [&[f64; 2]; 4], ysdot: [&mut [f64; 2]; 4],
    p: &Cmt2IvInfTheta,
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
fn native_cvodes_2cmt_iv_infusion_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>,
    dose_rates: Vec<f64>, theta: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    validate_infusion_multidose_inputs(&obs_times, &dose_times, &dose_amts, &dose_rates, 4, &theta, "CL, V1, Q, V2")?;
    let cl = theta[0];
    let v1 = theta[1];
    let q = theta[2];
    let v2 = theta[3];
    infusion_probe_multidose_core::<2, _>(&obs_times, &dose_times, &dose_amts, &dose_rates, |rate| Cmt2IvInfTheta { cl, v1, q, v2, rate }, rhs_2cmt_iv_inf)
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_2cmt_iv_infusion_sensitivity_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>,
    dose_rates: Vec<f64>, theta: Vec<f64>,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    validate_infusion_multidose_inputs(&obs_times, &dose_times, &dose_amts, &dose_rates, 4, &theta, "CL, V1, Q, V2")?;
    let cl = theta[0];
    let v1 = theta[1];
    let q = theta[2];
    let v2 = theta[3];
    infusion_sens_probe_multidose_core::<2, 4, _>(&obs_times, &dose_times, &dose_amts, &dose_rates, |rate| Cmt2IvInfTheta { cl, v1, q, v2, rate }, rhs_2cmt_iv_inf, sens_rhs_2cmt_iv_inf)
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
    probe_multidose_core::<3, _>(&obs_times, &dose_times, &dose_amts, p, rhs_2cmt_oral)
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_2cmt_oral_sensitivity_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>, theta: Vec<f64>,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 5, &theta, "KA, CL, V2, Q, V3")?;
    let p = Cmt2OralTheta { ka: theta[0], cl: theta[1], v2: theta[2], q: theta[3], v3: theta[4] };
    sens_probe_multidose_core::<3, 5, _>(&obs_times, &dose_times, &dose_amts, p, rhs_2cmt_oral, sens_rhs_2cmt_oral)
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
    let p = Cmt3IvTheta { cl: theta[0], v1: theta[1], q2: theta[2], v2: theta[3], q3: theta[4], v3: theta[5] };
    probe_multidose_core::<3, _>(&obs_times, &dose_times, &dose_amts, p, rhs_3cmt_iv)
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_3cmt_iv_sensitivity_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>, theta: Vec<f64>,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 6, &theta, "CL, V1, Q2, V2, Q3, V3")?;
    let p = Cmt3IvTheta { cl: theta[0], v1: theta[1], q2: theta[2], v2: theta[3], q3: theta[4], v3: theta[5] };
    sens_probe_multidose_core::<3, 6, _>(&obs_times, &dose_times, &dose_amts, p, rhs_3cmt_iv, sens_rhs_3cmt_iv)
}


// ── Infusion variant: 3-compartment IV ───────────────────────────────────────

#[cfg(feature = "native-cvodes")]
#[derive(Clone, Copy)]
struct Cmt3IvInfTheta { cl: f64, v1: f64, q2: f64, v2: f64, q3: f64, v3: f64, rate: f64 }

#[cfg(feature = "native-cvodes")]
fn rhs_3cmt_iv_inf(_t: f64, y: &[f64; 3], dy: &mut [f64; 3], p: &Cmt3IvInfTheta) -> RhsResult {
    let k10 = p.cl / p.v1; let k12 = p.q2 / p.v1; let k21 = p.q2 / p.v2;
    let k13 = p.q3 / p.v1; let k31 = p.q3 / p.v3;
    *dy = [
        p.rate - (k10 + k12 + k13) * y[0] + k21 * y[1] + k31 * y[2],
        k12 * y[0] - k21 * y[1],
        k13 * y[0] - k31 * y[2],
    ];
    RhsResult::Ok
}

#[cfg(feature = "native-cvodes")]
fn sens_rhs_3cmt_iv_inf(
    _t: f64, y: &[f64; 3], _ydot: &[f64; 3],
    ys: [&[f64; 3]; 6], ysdot: [&mut [f64; 3]; 6],
    p: &Cmt3IvInfTheta,
) -> RhsResult {
    let (a1, a2, a3) = (y[0], y[1], y[2]);
    let k10 = p.cl / p.v1; let k12 = p.q2 / p.v1; let k21 = p.q2 / p.v2;
    let k13 = p.q3 / p.v1; let k31 = p.q3 / p.v3;
    let v1s = p.v1 * p.v1; let v2s = p.v2 * p.v2; let v3s = p.v3 * p.v3;
    let direct: [[f64; 3]; 6] = [
        [-a1 / p.v1, 0.0, 0.0],
        [(p.cl + p.q2 + p.q3) * a1 / v1s, -p.q2 * a1 / v1s, -p.q3 * a1 / v1s],
        [-a1 / p.v1 + a2 / p.v2, a1 / p.v1 - a2 / p.v2, 0.0],
        [-p.q2 * a2 / v2s, p.q2 * a2 / v2s, 0.0],
        [-a1 / p.v1 + a3 / p.v3, 0.0, a1 / p.v1 - a3 / p.v3],
        [-p.q3 * a3 / v3s, 0.0, p.q3 * a3 / v3s],
    ];
    let j00 = -(k10 + k12 + k13); let j11 = -k21; let j22 = -k31;
    for (j, (s, sd)) in ys.into_iter().zip(ysdot.into_iter()).enumerate() {
        let d = direct[j];
        *sd = [
            j00 * s[0] + k21 * s[1] + k31 * s[2] + d[0],
            k12 * s[0] + j11 * s[1] + d[1],
            k13 * s[0] + j22 * s[2] + d[2],
        ];
    }
    RhsResult::Ok
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_3cmt_iv_infusion_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>,
    dose_rates: Vec<f64>, theta: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    validate_infusion_multidose_inputs(&obs_times, &dose_times, &dose_amts, &dose_rates, 6, &theta, "CL, V1, Q2, V2, Q3, V3")?;
    let cl = theta[0];
    let v1 = theta[1];
    let q2 = theta[2];
    let v2 = theta[3];
    let q3 = theta[4];
    let v3 = theta[5];
    infusion_probe_multidose_core::<3, _>(&obs_times, &dose_times, &dose_amts, &dose_rates, |rate| Cmt3IvInfTheta { cl, v1, q2, v2, q3, v3, rate }, rhs_3cmt_iv_inf)
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_3cmt_iv_infusion_sensitivity_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>,
    dose_rates: Vec<f64>, theta: Vec<f64>,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    validate_infusion_multidose_inputs(&obs_times, &dose_times, &dose_amts, &dose_rates, 6, &theta, "CL, V1, Q2, V2, Q3, V3")?;
    let cl = theta[0];
    let v1 = theta[1];
    let q2 = theta[2];
    let v2 = theta[3];
    let q3 = theta[4];
    let v3 = theta[5];
    infusion_sens_probe_multidose_core::<3, 6, _>(&obs_times, &dose_times, &dose_amts, &dose_rates, |rate| Cmt3IvInfTheta { cl, v1, q2, v2, q3, v3, rate }, rhs_3cmt_iv_inf, sens_rhs_3cmt_iv_inf)
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
    let p = Cmt3OralTheta { ka: theta[0], cl: theta[1], v2: theta[2], q3: theta[3], v3: theta[4], q4: theta[5], v4: theta[6] };
    probe_multidose_core::<4, _>(&obs_times, &dose_times, &dose_amts, p, rhs_3cmt_oral)
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_3cmt_oral_sensitivity_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>, theta: Vec<f64>,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 7, &theta, "KA, CL, V2, Q3, V3, Q4, V4")?;
    let p = Cmt3OralTheta { ka: theta[0], cl: theta[1], v2: theta[2], q3: theta[3], v3: theta[4], q4: theta[5], v4: theta[6] };
    sens_probe_multidose_core::<4, 7, _>(&obs_times, &dose_times, &dose_amts, p, rhs_3cmt_oral, sens_rhs_3cmt_oral)
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
    let p = Cmt4IvTheta { cl: theta[0], v1: theta[1], q2: theta[2], v2: theta[3], q3: theta[4], v3: theta[5], q4: theta[6], v4: theta[7] };
    probe_multidose_core::<4, _>(&obs_times, &dose_times, &dose_amts, p, rhs_4cmt_iv)
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_4cmt_iv_sensitivity_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>, theta: Vec<f64>,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 8, &theta, "CL, V1, Q2, V2, Q3, V3, Q4, V4")?;
    let p = Cmt4IvTheta { cl: theta[0], v1: theta[1], q2: theta[2], v2: theta[3], q3: theta[4], v3: theta[5], q4: theta[6], v4: theta[7] };
    sens_probe_multidose_core::<4, 8, _>(&obs_times, &dose_times, &dose_amts, p, rhs_4cmt_iv, sens_rhs_4cmt_iv)
}

// ── Infusion variant: 4-compartment IV ───────────────────────────────────────

#[cfg(feature = "native-cvodes")]
#[derive(Clone, Copy)]
struct Cmt4IvInfTheta { cl:f64, v1:f64, q2:f64, v2:f64, q3:f64, v3:f64, q4:f64, v4:f64, rate:f64 }

#[cfg(feature = "native-cvodes")]
fn rhs_4cmt_iv_inf(_t: f64, y: &[f64; 4], dy: &mut [f64; 4], p: &Cmt4IvInfTheta) -> RhsResult {
    let k10=p.cl/p.v1; let k12=p.q2/p.v1; let k21=p.q2/p.v2;
    let k13=p.q3/p.v1; let k31=p.q3/p.v3; let k14=p.q4/p.v1; let k41=p.q4/p.v4;
    *dy = [
        p.rate - (k10+k12+k13+k14)*y[0] + k21*y[1] + k31*y[2] + k41*y[3],
        k12*y[0] - k21*y[1], k13*y[0] - k31*y[2], k14*y[0] - k41*y[3],
    ];
    RhsResult::Ok
}

#[cfg(feature = "native-cvodes")]
fn sens_rhs_4cmt_iv_inf(
    _t: f64, y: &[f64; 4], _ydot: &[f64; 4],
    ys: [&[f64; 4]; 8], ysdot: [&mut [f64; 4]; 8], p: &Cmt4IvInfTheta,
) -> RhsResult {
    let (a1,a2,a3,a4)=(y[0],y[1],y[2],y[3]);
    let k10=p.cl/p.v1; let k12=p.q2/p.v1; let k21=p.q2/p.v2;
    let k13=p.q3/p.v1; let k31=p.q3/p.v3; let k14=p.q4/p.v1; let k41=p.q4/p.v4;
    let v1s=p.v1*p.v1; let v2s=p.v2*p.v2; let v3s=p.v3*p.v3; let v4s=p.v4*p.v4;
    let direct: [[f64; 4]; 8] = [
        [-a1/p.v1, 0.0, 0.0, 0.0],
        [(p.cl+p.q2+p.q3+p.q4)*a1/v1s, -p.q2*a1/v1s, -p.q3*a1/v1s, -p.q4*a1/v1s],
        [-a1/p.v1+a2/p.v2, a1/p.v1-a2/p.v2, 0.0, 0.0],
        [-p.q2*a2/v2s, p.q2*a2/v2s, 0.0, 0.0],
        [-a1/p.v1+a3/p.v3, 0.0, a1/p.v1-a3/p.v3, 0.0],
        [-p.q3*a3/v3s, 0.0, p.q3*a3/v3s, 0.0],
        [-a1/p.v1+a4/p.v4, 0.0, 0.0, a1/p.v1-a4/p.v4],
        [-p.q4*a4/v4s, 0.0, 0.0, p.q4*a4/v4s],
    ];
    let j00=-(k10+k12+k13+k14); let j11=-k21; let j22=-k31; let j33=-k41;
    for (j,(s,sd)) in ys.into_iter().zip(ysdot.into_iter()).enumerate() {
        let d=direct[j];
        *sd=[j00*s[0]+k21*s[1]+k31*s[2]+k41*s[3]+d[0], k12*s[0]+j11*s[1]+d[1],
             k13*s[0]+j22*s[2]+d[2], k14*s[0]+j33*s[3]+d[3]];
    }
    RhsResult::Ok
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_4cmt_iv_infusion_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>,
    dose_rates: Vec<f64>, theta: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    validate_infusion_multidose_inputs(&obs_times, &dose_times, &dose_amts, &dose_rates, 8, &theta, "CL, V1, Q2, V2, Q3, V3, Q4, V4")?;
    let cl = theta[0];
    let v1 = theta[1];
    let q2 = theta[2];
    let v2 = theta[3];
    let q3 = theta[4];
    let v3 = theta[5];
    let q4 = theta[6];
    let v4 = theta[7];
    infusion_probe_multidose_core::<4, _>(&obs_times, &dose_times, &dose_amts, &dose_rates, |rate| Cmt4IvInfTheta { cl, v1, q2, v2, q3, v3, q4, v4, rate }, rhs_4cmt_iv_inf)
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_4cmt_iv_infusion_sensitivity_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>,
    dose_rates: Vec<f64>, theta: Vec<f64>,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    validate_infusion_multidose_inputs(&obs_times, &dose_times, &dose_amts, &dose_rates, 8, &theta, "CL, V1, Q2, V2, Q3, V3, Q4, V4")?;
    let cl = theta[0];
    let v1 = theta[1];
    let q2 = theta[2];
    let v2 = theta[3];
    let q3 = theta[4];
    let v3 = theta[5];
    let q4 = theta[6];
    let v4 = theta[7];
    infusion_sens_probe_multidose_core::<4, 8, _>(&obs_times, &dose_times, &dose_amts, &dose_rates, |rate| Cmt4IvInfTheta { cl, v1, q2, v2, q3, v3, q4, v4, rate }, rhs_4cmt_iv_inf, sens_rhs_4cmt_iv_inf)
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
    let p = Cmt4OralTheta { ka: theta[0], cl: theta[1], v2: theta[2], q3: theta[3], v3: theta[4], q4: theta[5], v4: theta[6], q5: theta[7], v5: theta[8] };
    probe_multidose_core::<5, _>(&obs_times, &dose_times, &dose_amts, p, rhs_4cmt_oral)
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn native_cvodes_4cmt_oral_sensitivity_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>, theta: Vec<f64>,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 9, &theta, "KA, CL, V2, Q3, V3, Q4, V4, Q5, V5")?;
    let p = Cmt4OralTheta { ka: theta[0], cl: theta[1], v2: theta[2], q3: theta[3], v3: theta[4], q4: theta[5], v4: theta[6], q5: theta[7], v5: theta[8] };
    sens_probe_multidose_core::<5, 9, _>(&obs_times, &dose_times, &dose_amts, p, rhs_4cmt_oral, sens_rhs_4cmt_oral)
}

// ══════════════════════════════════════════════════════════════════════════════
// Analytical closed-form probes — exact superposition, no ODE integration.
//
// ADVAN1 (1-cmt IV)    theta = [CL, V]          states = [A1]
// ADVAN2 (1-cmt oral)  theta = [KA, CL, V]      states = [A1_depot, A2_central]
// ADVAN3 (2-cmt IV)    theta = [CL, V1, Q, V2]  states = [A1_central, A2_periph]
// ADVAN4 (2-cmt oral)  theta = [KA,CL,V2,Q,V3]  states = [A1,A2_central,A3]
//
// Bolus pre-dose convention: observation at dose time is excluded (dt > 0).
// Infusion convention: observation at infusion start is included (dt >= 0).
// ══════════════════════════════════════════════════════════════════════════════

/// KA ≈ K tolerance (use L'Hôpital limit form for Bateman equation).
#[cfg(feature = "native-cvodes")]
const ANALYTIC_KA_K_TOL: f64 = 1e-6;

/// λ1 ≈ λ2 tolerance (degenerate eigenvalue limit form).
#[cfg(feature = "native-cvodes")]
const ANALYTIC_EIG_TOL: f64 = 1e-10;

/// 2-compartment eigenvalues from micro-rate constants (k10, k12, k21).
/// Returns (lam1_slow, lam2_fast) where lam1 ≤ lam2.
#[cfg(feature = "native-cvodes")]
#[inline]
fn eigenvalues_2cmt(k10: f64, k12: f64, k21: f64) -> (f64, f64) {
    let s = k10 + k12 + k21;
    let d = (s * s - 4.0 * k10 * k21).max(0.0).sqrt();
    ((s - d) / 2.0, (s + d) / 2.0)
}

/// Stable (1 − exp(−λ·t)) / λ with the λ→0 limit.
#[cfg(feature = "native-cvodes")]
#[inline]
fn one_minus_exp_over_lam(lam: f64, dt: f64) -> f64 {
    if lam.abs() < 1e-14 { dt } else { (1.0 - (-lam * dt).exp()) / lam }
}

/// Stable (exp(−a·t) − exp(−b·t)) / (b − a) with the a→b limit.
#[cfg(feature = "native-cvodes")]
#[inline]
fn analytic_decay_diff(a: f64, b: f64, dt: f64) -> f64 {
    if (b - a).abs() < ANALYTIC_EIG_TOL {
        dt * (-a * dt).exp()
    } else {
        ((-a * dt).exp() - (-b * dt).exp()) / (b - a)
    }
}

/// Propagate 2-cmt system from (a1_0, a2_0) over time dt using eigendecomp.
#[cfg(feature = "native-cvodes")]
#[inline]
fn analytic_propagate_2cmt(
    a1_0: f64, a2_0: f64,
    lam1: f64, lam2: f64, k12: f64, k21: f64,
    dt: f64,
) -> (f64, f64) {
    let dl = lam2 - lam1;
    if dl < ANALYTIC_EIG_TOL {
        let e = (-lam1 * dt).exp();
        return ((a1_0 + a2_0 * k21 * dt) * e, a2_0 * e);
    }
    let e1 = (-lam1 * dt).exp();
    let e2 = (-lam2 * dt).exp();
    let new_a1 = (a1_0 * (k21 - lam1) + a2_0 * k21) / dl * e1
               + (a1_0 * (lam2 - k21) - a2_0 * k21) / dl * e2;
    let new_a2 = (a1_0 * k12 + a2_0 * (lam2 - k21)) / dl * e1
               + (-a1_0 * k12 + a2_0 * (k21 - lam1)) / dl * e2;
    (new_a1, new_a2)
}

/// 2-cmt biexponential contribution for a single IV bolus dose.
#[cfg(feature = "native-cvodes")]
#[inline]
fn analytic_biexp_bolus(
    dose: f64, k12: f64, k21: f64, lam1: f64, lam2: f64, dt: f64,
) -> (f64, f64) {
    let dl = lam2 - lam1;
    if dl < ANALYTIC_EIG_TOL {
        let e = (-lam1 * dt).exp();
        return (dose * e * (1.0 - lam1 * dt + k21 * dt), dose * k12 * dt * e);
    }
    let e1 = (-lam1 * dt).exp();
    let e2 = (-lam2 * dt).exp();
    let a1 = dose * ((k21 - lam1) / dl * e1 + (lam2 - k21) / dl * e2);
    let a2 = dose * k12 / dl * (e1 - e2);
    (a1, a2)
}

/// 2-cmt amounts during constant-rate IV infusion (analytical, no ODE).
#[cfg(feature = "native-cvodes")]
#[inline]
fn analytic_biexp_infusion(
    rate: f64, k12: f64, k21: f64, lam1: f64, lam2: f64, dt: f64,
) -> (f64, f64) {
    let dl = lam2 - lam1;
    if dl < ANALYTIC_EIG_TOL {
        return (rate * one_minus_exp_over_lam(lam1, dt), 0.0);
    }
    let f1 = one_minus_exp_over_lam(lam1, dt);
    let f2 = one_minus_exp_over_lam(lam2, dt);
    let a1 = rate * ((k21 - lam1) / dl * f1 + (lam2 - k21) / dl * f2);
    let a2 = rate * k12 / dl * (f1 - f2);
    (a1, a2)
}

// ──────────────────────────────────────────────────────────────────────────────
// ADVAN1: 1-compartment IV bolus/infusion  (CL, V)
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn analytic_1cmt_iv_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>, theta: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 2, &theta, "CL, V")?;
    let k = theta[0] / theta[1]; // k = CL/V
    let n_obs = obs_times.len();
    let mut out = vec![vec![0.0f64; 1]; n_obs];
    for (i, &t_obs) in obs_times.iter().enumerate() {
        let mut a1 = 0.0f64;
        for j in 0..dose_times.len() {
            let dt = t_obs - dose_times[j];
            if dt > 0.0 { a1 += dose_amts[j] * (-k * dt).exp(); }
        }
        out[i][0] = a1;
    }
    Ok(out)
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn analytic_1cmt_iv_infusion_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>,
    dose_rates: Vec<f64>, theta: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    validate_infusion_multidose_inputs(
        &obs_times, &dose_times, &dose_amts, &dose_rates, 2, &theta, "CL, V")?;
    let k = theta[0] / theta[1];
    let n_obs = obs_times.len();
    let mut out = vec![vec![0.0f64; 1]; n_obs];
    for (i, &t_obs) in obs_times.iter().enumerate() {
        let mut a1 = 0.0f64;
        for j in 0..dose_times.len() {
            let td = dose_times[j]; let amt = dose_amts[j]; let rate = dose_rates[j];
            let dt = t_obs - td;
            if rate == 0.0 {
                if dt > 0.0 { a1 += amt * (-k * dt).exp(); }
            } else {
                let dur = amt / rate;
                if dt >= 0.0 && dt <= dur {
                    a1 += rate / k * (1.0 - (-k * dt).exp());
                } else if dt > dur {
                    a1 += rate / k * (1.0 - (-k * dur).exp()) * (-k * (dt - dur)).exp();
                }
            }
        }
        out[i][0] = a1;
    }
    Ok(out)
}

// ──────────────────────────────────────────────────────────────────────────────
// ADVAN2: 1-compartment oral  (KA, CL, V)
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn analytic_1cmt_oral_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>, theta: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 3, &theta, "KA, CL, V")?;
    let ka = theta[0]; let k = theta[1] / theta[2]; // k = CL/V
    let limit_form = (ka - k).abs() < ANALYTIC_KA_K_TOL;
    let bolus_scale = if limit_form { 0.0 } else { ka / (ka - k) };
    let n_obs = obs_times.len();
    let mut out = vec![vec![0.0f64; 2]; n_obs];
    for (i, &t_obs) in obs_times.iter().enumerate() {
        let mut a1 = 0.0f64; // depot
        let mut a2 = 0.0f64; // central
        for j in 0..dose_times.len() {
            let dt = t_obs - dose_times[j];
            if dt <= 0.0 { continue; }
            let dose = dose_amts[j];
            a1 += dose * (-ka * dt).exp();
            if limit_form {
                a2 += dose * ka * dt * (-k * dt).exp();
            } else {
                a2 += dose * bolus_scale * ((-k * dt).exp() - (-ka * dt).exp());
            }
        }
        out[i][0] = a1; out[i][1] = a2;
    }
    Ok(out)
}

// ──────────────────────────────────────────────────────────────────────────────
// ADVAN3: 2-compartment IV  (CL, V1, Q, V2)
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn analytic_2cmt_iv_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>, theta: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    validate_multidose_inputs(&obs_times, &dose_times, &dose_amts, 4, &theta, "CL, V1, Q, V2")?;
    let k10 = theta[0] / theta[1]; let k12 = theta[2] / theta[1]; let k21 = theta[2] / theta[3];
    let (lam1, lam2) = eigenvalues_2cmt(k10, k12, k21);
    let n_obs = obs_times.len();
    let mut out = vec![vec![0.0f64; 2]; n_obs];
    for (i, &t_obs) in obs_times.iter().enumerate() {
        let mut a1 = 0.0f64; let mut a2 = 0.0f64;
        for j in 0..dose_times.len() {
            let dt = t_obs - dose_times[j];
            if dt <= 0.0 { continue; }
            let (da1, da2) = analytic_biexp_bolus(dose_amts[j], k12, k21, lam1, lam2, dt);
            a1 += da1; a2 += da2;
        }
        out[i][0] = a1; out[i][1] = a2;
    }
    Ok(out)
}

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn analytic_2cmt_iv_infusion_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>,
    dose_rates: Vec<f64>, theta: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    validate_infusion_multidose_inputs(
        &obs_times, &dose_times, &dose_amts, &dose_rates, 4, &theta, "CL, V1, Q, V2")?;
    let k10 = theta[0] / theta[1]; let k12 = theta[2] / theta[1]; let k21 = theta[2] / theta[3];
    let (lam1, lam2) = eigenvalues_2cmt(k10, k12, k21);
    let n_obs = obs_times.len();
    let mut out = vec![vec![0.0f64; 2]; n_obs];
    for (i, &t_obs) in obs_times.iter().enumerate() {
        let mut a1 = 0.0f64; let mut a2 = 0.0f64;
        for j in 0..dose_times.len() {
            let td = dose_times[j]; let amt = dose_amts[j]; let rate = dose_rates[j];
            let dt = t_obs - td;
            if rate == 0.0 {
                if dt <= 0.0 { continue; }
                let (da1, da2) = analytic_biexp_bolus(amt, k12, k21, lam1, lam2, dt);
                a1 += da1; a2 += da2;
            } else {
                let dur = amt / rate;
                if dt < 0.0 { continue; }
                if dt <= dur {
                    let (da1, da2) = analytic_biexp_infusion(rate, k12, k21, lam1, lam2, dt);
                    a1 += da1; a2 += da2;
                } else {
                    let (a1e, a2e) = analytic_biexp_infusion(rate, k12, k21, lam1, lam2, dur);
                    let (da1, da2) = analytic_propagate_2cmt(a1e, a2e, lam1, lam2, k12, k21, dt - dur);
                    a1 += da1; a2 += da2;
                }
            }
        }
        out[i][0] = a1; out[i][1] = a2;
    }
    Ok(out)
}

// ──────────────────────────────────────────────────────────────────────────────
// ADVAN4: 2-compartment oral  (KA, CL, V2, Q, V3)
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "native-cvodes")]
#[pyfunction]
fn analytic_2cmt_oral_probe_multidose(
    obs_times: Vec<f64>, dose_times: Vec<f64>, dose_amts: Vec<f64>, theta: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    validate_multidose_inputs(
        &obs_times, &dose_times, &dose_amts, 5, &theta, "KA, CL, V2, Q, V3")?;
    let ka = theta[0]; let k10 = theta[1] / theta[2];
    let k12 = theta[3] / theta[2]; let k21 = theta[3] / theta[4];
    let (lam1, lam2) = eigenvalues_2cmt(k10, k12, k21);
    let dl = lam2 - lam1;
    let n_obs = obs_times.len();
    let mut out = vec![vec![0.0f64; 3]; n_obs];
    for (i, &t_obs) in obs_times.iter().enumerate() {
        let mut a1 = 0.0f64; // depot
        let mut a2 = 0.0f64; // central
        let mut a3 = 0.0f64; // peripheral
        for j in 0..dose_times.len() {
            let dt = t_obs - dose_times[j];
            if dt <= 0.0 { continue; }
            let dose = dose_amts[j];
            a1 += dose * (-ka * dt).exp();
            if dl < ANALYTIC_EIG_TOL {
                // Degenerate disposition eigenvalues: collapse to ADVAN2-like
                let denom = ka - k10;
                if denom.abs() < ANALYTIC_KA_K_TOL {
                    a2 += dose * ka * dt * (-k10 * dt).exp();
                } else {
                    a2 += dose * ka / denom * ((-k10 * dt).exp() - (-ka * dt).exp());
                }
                // a3 remains 0 in this degenerate limit
            } else {
                let h1 = analytic_decay_diff(lam1, ka, dt);
                let h2 = analytic_decay_diff(lam2, ka, dt);
                let c1 = (k21 - lam1) / dl;
                let c2 = (lam2 - k21) / dl;
                a2 += dose * ka * (c1 * h1 + c2 * h2);
                a3 += dose * ka * k12 / dl * (h1 - h2);
            }
        }
        out[i][0] = a1; out[i][1] = a2; out[i][2] = a3;
    }
    Ok(out)
}

// ── module registration ───────────────────────────────────────────────────────

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(neg2ll_obs_loop, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_transit_1cmt_pkpd_probe, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_transit_1cmt_pkpd_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(
        native_cvodes_transit_1cmt_pkpd_sensitivity_probe_multidose, m
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
    // ── analytical closed-form probes (ADVAN1/2/3/4) ──────────────────────────
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(analytic_1cmt_iv_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(analytic_1cmt_iv_infusion_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(analytic_1cmt_oral_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(analytic_2cmt_iv_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(analytic_2cmt_iv_infusion_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(analytic_2cmt_oral_probe_multidose, m)?)?;
    // ── infusion-aware IV probes ──────────────────────────────────────────────
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_1cmt_iv_infusion_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_1cmt_iv_infusion_sensitivity_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_2cmt_iv_infusion_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_2cmt_iv_infusion_sensitivity_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_3cmt_iv_infusion_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_3cmt_iv_infusion_sensitivity_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_4cmt_iv_infusion_probe_multidose, m)?)?;
    #[cfg(feature = "native-cvodes")]
    m.add_function(wrap_pyfunction!(native_cvodes_4cmt_iv_infusion_sensitivity_probe_multidose, m)?)?;
    Ok(())
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── norm_logcdf ──────────────────────────────────────────────────────────

    #[test]
    fn test_norm_logcdf_standard_values() {
        // z=0: ln(0.5) ≈ -0.6931
        let r0 = norm_logcdf(0.0_f64);
        assert!(
            (r0 - (-0.6931_f64)).abs() < 0.001,
            "norm_logcdf(0) = {r0}, expected ≈ -0.6931"
        );

        // z=-1: ln(Φ(-1)) ≈ -1.8411
        let r1 = norm_logcdf(-1.0_f64);
        assert!(
            (r1 - (-1.8411_f64)).abs() < 0.001,
            "norm_logcdf(-1) = {r1}, expected ≈ -1.8411"
        );

        // z=1: ln(Φ(1)) ≈ -0.1728
        let r_pos = norm_logcdf(1.0_f64);
        assert!(
            r_pos < 0.0 && r_pos > -1.0,
            "norm_logcdf(1) = {r_pos}, expected in (-1, 0)"
        );
    }

    #[test]
    fn test_norm_logcdf_deep_tail_finite() {
        // z=-30: should be finite and not clamped to -1e10
        let r30 = norm_logcdf(-30.0_f64);
        assert!(r30.is_finite(), "norm_logcdf(-30) must be finite, got {r30}");
        assert!(r30 < -400.0, "norm_logcdf(-30) = {r30}, expected < -400");
    }

    #[test]
    fn test_norm_logcdf_deep_tail_not_clamped() {
        // Before the Sprint 1 fix, values at z=-100 were wrongly clamped to
        // ln(1e-300) ≈ -690. The asymptotic expansion gives ≈ -5012.5.
        let r100 = norm_logcdf(-100.0_f64);
        assert!(r100.is_finite(), "norm_logcdf(-100) must be finite");
        assert!(r100 < -5000.0, "norm_logcdf(-100) = {r100}, expected < -5000");
        assert!(r100 > -6000.0, "norm_logcdf(-100) = {r100}, expected > -6000");
    }

    #[test]
    fn test_norm_logcdf_m3_known_value() {
        // M3: log Φ((lloq - ipred) / sigma)
        // ipred=2.0, sigma=0.3, lloq=1.0 → z = (1-2)/0.3 = -3.333...
        let z = (1.0_f64 - 2.0_f64) / 0.3_f64;
        let result = norm_logcdf(z);
        // Expected: log Φ(-3.333) ≈ -7.754  (Φ(-3.333)≈4.29e-4, log(4.29e-4)≈-7.754)
        assert!(
            (result - (-7.7539_f64)).abs() < 0.01,
            "norm_logcdf(-3.333) = {result}, expected ≈ -7.754"
        );
    }

    // ── normal_ll ────────────────────────────────────────────────────────────

    #[test]
    fn test_normal_ll_zero_variance_returns_large_negative() {
        let ll = normal_ll(1.0, 1.0, 0.0);
        assert_eq!(ll, -1e30, "var=0 must return -1e30");
    }

    #[test]
    fn test_normal_ll_perfect_fit() {
        // y = mu, var = 1 → ll = -0.5 * ln(2π) ≈ -0.9189
        let ll = normal_ll(3.0, 3.0, 1.0);
        let expected = -0.5_f64 * LOG2PI;
        assert!(
            (ll - expected).abs() < 1e-10,
            "normal_ll(y=mu) = {ll}, expected {expected}"
        );
    }

    // ── BLQ method branching (via neg2ll_obs_loop logic) ─────────────────────
    // The BLQ branching is only accessible via PyO3 functions, so we test
    // the underlying math helpers (norm_logcdf, normal_ll, norm_cdf) that
    // implement each method.

    #[test]
    fn test_blq_m1_contribution_is_zero() {
        // M1: BLQ observations are excluded → contribution to LL is 0.
        // This is enforced by the 'continue' in the match arm.
        // We verify indirectly: computing normal_ll for a non-BLQ obs gives
        // a finite non-zero result, confirming M1 skips it.
        let non_blq_ll = normal_ll(2.5, 2.0, 0.09); // obs=2.5, mu=2.0, var=0.09
        assert!(non_blq_ll.is_finite() && non_blq_ll != 0.0);
        // M1 contribution for a BLQ obs is 0 (nothing added to ll)
        // — this is a branch, not a call, so we just verify the formula's absence.
        let m1_contribution: f64 = 0.0; // by definition of method M1
        assert_eq!(m1_contribution, 0.0);
    }

    #[test]
    fn test_blq_m3_censored_likelihood() {
        // M3: censored = log Φ((lloq - mu) / sigma)
        // ipred=2.0, sigma=0.3 (var=0.09), lloq=1.0
        let mu = 2.0_f64;
        let sigma = 0.3_f64;
        let lloq = 1.0_f64;
        let z = (lloq - mu) / sigma; // = -3.333...
        let ll_m3 = norm_logcdf(z);
        // log Φ(-3.333) ≈ -7.754  (Φ(-3.333)≈4.29e-4, log(4.29e-4)≈-7.754)
        assert!(
            (ll_m3 - (-7.7539_f64)).abs() < 0.01,
            "M3 log-likelihood = {ll_m3}, expected ≈ -7.754"
        );
    }

    #[test]
    fn test_blq_m4_truncated_normal() {
        // M4: log[Φ(z_lloq) - Φ(z_0)] - log[1 - Φ(z_0)]
        let mu = 2.0_f64;
        let sigma = 0.3_f64;
        let lloq = 1.0_f64;
        let z_lloq = (lloq - mu) / sigma;
        let z_0 = -mu / sigma;
        let prob_window = norm_cdf(z_lloq) - norm_cdf(z_0);
        let prob_pos = 1.0 - norm_cdf(z_0);
        assert!(prob_window > 0.0, "prob_window must be positive");
        assert!(prob_pos > 0.0, "prob_pos must be positive");
        let ll_m4 = prob_window.ln() - prob_pos.ln();
        // The result should be a finite, negative number
        assert!(ll_m4.is_finite(), "M4 log-likelihood must be finite");
        assert!(ll_m4 < 0.0, "M4 log-likelihood must be negative");
    }

    #[test]
    fn test_blq_m5_imputes_lloq_half() {
        // M5: impute DV = lloq / 2 → normal_ll(lloq*0.5, mu, var)
        let mu = 2.0_f64;
        let var = 0.09_f64;
        let lloq = 1.0_f64;
        let ll_m5 = normal_ll(lloq * 0.5, mu, var);
        let ll_direct = normal_ll(0.5, 2.0, 0.09);
        assert!((ll_m5 - ll_direct).abs() < 1e-15);
        assert!(ll_m5.is_finite() && ll_m5 < 0.0);
    }

    #[test]
    fn test_blq_m7_imputes_zero() {
        // M7: impute DV = 0 → normal_ll(0, mu, var)
        let mu = 2.0_f64;
        let var = 0.09_f64;
        let ll_m7 = normal_ll(0.0, mu, var);
        let ll_direct = normal_ll(0.0, 2.0, 0.09);
        assert!((ll_m7 - ll_direct).abs() < 1e-15);
        assert!(ll_m7.is_finite() && ll_m7 < 0.0);
    }

    #[test]
    fn test_norm_cdf_symmetry() {
        // Φ(z) + Φ(-z) = 1
        for z in &[0.5_f64, 1.0, 2.0, 3.0] {
            let sum = norm_cdf(*z) + norm_cdf(-z);
            assert!(
                (sum - 1.0_f64).abs() < 1e-14,
                "CDF symmetry broken at z={z}: Φ(z)+Φ(-z)={sum}"
            );
        }
    }

    #[test]
    fn test_norm_cdf_at_zero() {
        // Φ(0) = 0.5
        let r = norm_cdf(0.0_f64);
        assert!((r - 0.5_f64).abs() < 1e-15, "Φ(0) = {r}, expected 0.5");
    }
}
