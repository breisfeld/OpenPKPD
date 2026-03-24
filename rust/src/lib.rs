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

// ── module registration ───────────────────────────────────────────────────────

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(neg2ll_obs_loop, m)?)?;
    Ok(())
}
