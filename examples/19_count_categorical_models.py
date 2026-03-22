"""
Example 19: Count and categorical PD models.

Demonstrates:
  - PoissonModel: modelling count-type endpoints (e.g., number of seizures)
  - NegativeBinomialModel: over-dispersed counts
  - ZeroInflatedPoissonModel: excess zeros in count data
  - ProportionalOddsModel: ordered categorical response (pain, toxicity grades)
  - DiscreteTimeMarkovModel: longitudinal categorical states
  - ContinuousTimeMarkovModel: semi-Markov state transitions

Background
----------
Many PD endpoints in clinical trials are not continuous:
  - Count endpoints: number of lesions, seizures, adverse events per day
  - Ordered categorical: CTCAE toxicity grade, NRS pain score 0–10
  - Binary/Markov: responder/non-responder, disease state transitions

OpenPKPD provides a hierarchy of models under openpkpd.models for each type.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_count_subjects(counts: np.ndarray, concentrations: np.ndarray) -> list:
    from openpkpd.models.count import CountData

    return [
        CountData(
            subject_id=i + 1,
            counts=np.array([int(count)]),
            times=np.array([0.0]),
            covariates={"conc": np.array([float(conc)])},
        )
        for i, (count, conc) in enumerate(zip(counts, concentrations))
    ]


def _encode_po_params(thresholds: np.ndarray, coef: np.ndarray) -> np.ndarray:
    raw = np.empty_like(thresholds)
    raw[0] = thresholds[0]
    if len(thresholds) > 1:
        raw[1:] = np.log(np.maximum(np.diff(thresholds), 1e-12))
    return np.concatenate([raw, coef])


# ===========================================================================
# PART 1: COUNT DATA MODELS
# ===========================================================================

def demo_count_models():
    from openpkpd.models.count import (
        PoissonModel, NegativeBinomialModel, ZeroInflatedPoissonModel, CountData,
    )

    print("=" * 60)
    print("PART 1: Count PD models")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Generate synthetic count data: seizure count per week
    # Assume mean seizure rate = 3 + drug effect (negative)
    # ------------------------------------------------------------------
    np.random.seed(42)
    n_subjects = 30
    concentrations = np.random.exponential(scale=2.0, size=n_subjects)  # drug exposure

    # True model: log(rate) = log(3) - 0.15 * concentration
    true_log_rate = np.log(3.0) - 0.15 * concentrations
    true_rate = np.exp(true_log_rate)
    counts = np.random.poisson(true_rate)

    print(f"\nData summary: {n_subjects} subjects")
    print(f"  Mean count: {counts.mean():.2f}  (expected ~{true_rate.mean():.2f})")
    print(f"  Proportion zeros: {(counts == 0).mean():.1%}")
    print(f"  Max count: {counts.max()}")

    data = _make_count_subjects(counts, concentrations)
    poisson_init = np.array([np.log(max(counts.mean(), 1e-6)), 0.0])

    # ------------------------------------------------------------------
    # 1a. Poisson model
    # ------------------------------------------------------------------
    print("\n--- Poisson model ---")
    poisson_model = PoissonModel()
    result_poisson = poisson_model.fit(data, init_params=poisson_init)
    if result_poisson:
        print(f"  Fitted rate params: {np.round(result_poisson.rate_params, 3)}")
        print(f"  AIC: {result_poisson.aic:.2f}")

    # Predict
    test_conc = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
    pred_rates = None
    if result_poisson is not None:
        pred_rates = poisson_model.mean_rate(
            result_poisson.rate_params,
            covariates={"conc": test_conc},
        )
    if pred_rates is not None:
        print(f"  Predicted rates at C=[0,1,2,5,10]: {np.round(pred_rates, 3)}")

    # PMF at count=3 for different concentrations
    test_params = np.array([np.log(3.0), -0.15])
    pmf_at_3 = [
        np.exp(
            poisson_model.log_pmf(
                3,
                float(poisson_model.mean_rate(test_params, {"conc": np.array([c])})[0]),
            )
        )
        for c in [0.0, 2.0, 5.0]
    ]
    print(f"  P(count=3) at C=[0,2,5]: {[f'{p:.3f}' for p in pmf_at_3]}")

    # ------------------------------------------------------------------
    # 1b. Negative Binomial model (over-dispersion)
    # ------------------------------------------------------------------
    print("\n--- Negative Binomial model (over-dispersed counts) ---")
    # Generate over-dispersed data
    nb_counts = np.random.negative_binomial(n=2, p=0.4, size=n_subjects)
    nb_data = _make_count_subjects(nb_counts, concentrations)
    nb_init = np.array([np.log(max(nb_counts.mean(), 1e-6)), 0.0, 0.0])

    nb_model = NegativeBinomialModel()
    result_nb = nb_model.fit(nb_data, init_params=nb_init)
    if result_nb:
        print(f"  Fitted rate params: {np.round(result_nb.rate_params, 3)}")
        print(f"  Dispersion r: {result_nb.dispersion:.3f}")
        print(f"  AIC: {result_nb.aic:.2f}")
        # Compare AIC to Poisson on the same data
        pois_result = PoissonModel().fit(nb_data, init_params=poisson_init)
        if pois_result:
            print(f"  Poisson AIC: {pois_result.aic:.2f}  (NB AIC should be better for over-dispersed data)")

    # ------------------------------------------------------------------
    # 1c. Zero-inflated Poisson model
    # ------------------------------------------------------------------
    print("\n--- Zero-Inflated Poisson model (excess zeros) ---")
    # Mix of zeros (structural non-events) and Poisson counts
    struct_zero = np.random.binomial(1, 0.4, size=n_subjects)  # 40% structural zeros
    zip_counts = np.where(struct_zero == 1, 0, np.random.poisson(2.0, size=n_subjects))
    zip_data = _make_count_subjects(zip_counts, concentrations)
    zip_init = np.array([np.log(max(zip_counts.mean(), 1e-6)), 0.0, 0.0])

    print(f"  Proportion zeros (ZIP data): {(zip_counts == 0).mean():.1%}  (expected ~{1 - 0.6*np.exp(-2):.1%})")

    zip_model = ZeroInflatedPoissonModel()
    result_zip = zip_model.fit(zip_data, init_params=zip_init)
    if result_zip:
        print(f"  Fitted rate params: {np.round(result_zip.rate_params, 3)}")
        print(f"  AIC: {result_zip.aic:.2f}")


# ===========================================================================
# PART 2: CATEGORICAL DATA MODELS
# ===========================================================================

def demo_categorical_models():
    from openpkpd.models.categorical import (
        ProportionalOddsModel, DiscreteTimeMarkovModel,
        ContinuousTimeMarkovModel, CategoricalData,
    )

    print("\n" + "=" * 60)
    print("PART 2: Categorical PD models")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 2a. Proportional Odds Model
    #     Use case: pain NRS 0-10 collapsed to 3 categories (mild/mod/severe)
    # ------------------------------------------------------------------
    print("\n--- Proportional Odds model ---")
    np.random.seed(123)
    n = 80
    concentrations = np.random.uniform(0, 5, size=n)
    # True model: cumulative log-odds proportional to -0.3 * C
    # Thresholds: alpha1 = 0.5, alpha2 = 2.0  (3 categories: 0,1,2)
    alpha = [0.5, 2.0]
    beta_c = -0.5

    categories = np.zeros(n, dtype=int)
    for i in range(n):
        lp1 = alpha[0] + beta_c * concentrations[i]
        lp2 = alpha[1] + beta_c * concentrations[i]
        p_leq0 = 1 / (1 + np.exp(-lp1))
        p_leq1 = 1 / (1 + np.exp(-lp2))
        u = np.random.uniform()
        if u < p_leq0:
            categories[i] = 0
        elif u < p_leq1:
            categories[i] = 1
        else:
            categories[i] = 2

    print(f"  Category distribution: "
          f"0={( categories==0).sum()}  1={(categories==1).sum()}  2={(categories==2).sum()}")

    data_po = [
        CategoricalData(
            subject_id=1,
            categories=categories,
            times=np.arange(n, dtype=float),
            covariates={"conc": concentrations},
        )
    ]
    po_model = ProportionalOddsModel(n_categories=3)
    result_po = po_model.fit(data_po)
    if result_po:
        print(f"  Fitted thresholds: {np.round(result_po.thresholds, 3)}")
        print(f"  Fitted coefficients: {np.round(result_po.coef, 3)}")
        po_params = _encode_po_params(result_po.thresholds, result_po.coef)
        # Predict probabilities at C=0, 2.5, 5
        for c in [0.0, 2.5, 5.0]:
            probs = po_model.predict_probs(np.array([[c]]), po_params)
            print(f"  P(cat) at C={c}: {np.round(probs[0], 3)}")

    # ------------------------------------------------------------------
    # 2b. Discrete-Time Markov model
    #     Use case: longitudinal responder/non-responder status
    # ------------------------------------------------------------------
    print("\n--- Discrete-Time Markov model ---")
    # 2-state model: state 0 (non-responder), state 1 (responder)
    n_subjects = 20
    n_times = 6
    # Transition matrix P: P[i,j] = prob from state i to state j
    # P = [[0.7, 0.3],   (non-resp stays non-resp 70%, becomes resp 30%)
    #       [0.1, 0.9]]   (resp stays resp 90%, becomes non-resp 10%)
    true_P = np.array([[0.7, 0.3], [0.1, 0.9]])

    sequences = []
    initial_states = np.random.choice([0, 1], size=n_subjects, p=[0.6, 0.4])
    for s0 in initial_states:
        seq = [s0]
        for _ in range(n_times - 1):
            seq.append(np.random.choice(2, p=true_P[seq[-1]]))
        sequences.append(seq)

    obs_states = np.array(sequences)
    data_dtm = [
        CategoricalData(
            subject_id=subject_id + 1,
            categories=obs_states[subject_id],
            times=np.arange(n_times, dtype=float),
        )
        for subject_id in range(n_subjects)
    ]

    dtm_model = DiscreteTimeMarkovModel(n_states=2)
    result_dtm = dtm_model.fit(data_dtm)
    if result_dtm:
        print(f"  Fitted transition matrix:")
        P_fitted = dtm_model.transition_matrix(result_dtm.thresholds)
        for row in P_fitted:
            print(f"    {np.round(row, 3)}")
        print(f"  True matrix: [[0.7, 0.3], [0.1, 0.9]]")

    # ------------------------------------------------------------------
    # 2c. Continuous-Time Markov model
    #     Use case: time-varying disease state transitions
    # ------------------------------------------------------------------
    print("\n--- Continuous-Time Markov model ---")
    # 2-state: remission (0) / relapse (1)
    # Rate matrix Q: off-diagonal = instantaneous rates
    # Q = [[-0.2, 0.2],
    #       [0.5,  -0.5]]
    true_Q = np.array([[-0.2, 0.2], [0.5, -0.5]])

    n_subjects = 15
    obs_times = np.array([0.0, 1.0, 2.0, 4.0, 8.0])
    n_times = len(obs_times)
    sequences_ctm = []
    for _ in range(n_subjects):
        seq = [np.random.choice(2, p=[0.7, 0.3])]
        for t_idx in range(1, n_times):
            dt = obs_times[t_idx] - obs_times[t_idx - 1]
            from scipy.linalg import expm
            P_dt = expm(true_Q * dt)
            seq.append(np.random.choice(2, p=P_dt[seq[-1]]))
        sequences_ctm.append(seq)

    obs_states_ctm = np.array(sequences_ctm)
    data_ctm = [
        CategoricalData(
            subject_id=subject_id + 1,
            categories=obs_states_ctm[subject_id],
            times=obs_times,
        )
        for subject_id in range(n_subjects)
    ]

    ctm_model = ContinuousTimeMarkovModel(n_states=2)
    result_ctm = ctm_model.fit(data_ctm)
    if result_ctm:
        print(f"  Fitted rate matrix Q:")
        Q_fitted = ctm_model.rate_matrix(result_ctm.thresholds)
        for row in Q_fitted:
            print(f"    {np.round(row, 3)}")
        print(f"  True Q: [[-0.2, 0.2], [0.5, -0.5]]")


# ===========================================================================
# main
# ===========================================================================

def main():
    demo_count_models()
    demo_categorical_models()

    print("\n" + "=" * 60)
    print("Example 19 complete.")
    print()
    print("Key takeaways:")
    print("  PoissonModel        — count data with log-linear rate")
    print("  NegativeBinomialModel — over-dispersed counts")
    print("  ZeroInflatedPoisson — mixture of structural zeros + Poisson")
    print("  ProportionalOdds    — ordered categorical with covariates")
    print("  DiscreteTimeMarkov  — longitudinal state sequences")
    print("  ContinuousTimeMarkov — time-continuous state transitions")


if __name__ == "__main__":
    main()
