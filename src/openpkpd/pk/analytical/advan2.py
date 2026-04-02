"""
ADVAN2 — 1-compartment oral absorption model.

Two compartments: depot (cmt 1, absorption) → central (cmt 2, observation).

Analytical solution (Bateman function):
    A2(t) = F * DOSE * KA / (KA - K) * (exp(-K*t) - exp(-KA*t))

For multiple doses, superposition is used.
When KA ≈ K, L'Hôpital limit: A2(t) = F * DOSE * KA * t * exp(-K*t)
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.base import PKSolution, PKSubroutine
from openpkpd.utils.errors import PKError

_KA_K_TOL = 1e-6  # tolerance for KA ≈ K (use limit form)


@dataclass(frozen=True)
class _BolusDoseDesign:
    amount: float
    positive_idx: np.ndarray
    dt_pos: np.ndarray
    ss: bool = False
    ii: float = 0.0


@dataclass(frozen=True)
class _InfusionDoseDesign:
    amount: float
    rate: float
    positive_idx: np.ndarray
    dt_pos: np.ndarray


@dataclass(frozen=True)
class _ADVAN2Design:
    n_times: int
    bolus_doses: tuple[_BolusDoseDesign, ...]
    infusion_doses: tuple[_InfusionDoseDesign, ...]
    single_bolus: _BolusDoseDesign | None = None


@dataclass
class _ADVAN2CacheEntry:
    obs_times_ref: np.ndarray
    dose_events_ref: list[DoseEvent]
    design: _ADVAN2Design


class ADVAN2(PKSubroutine):
    """1-compartment oral model (ADVAN2)."""

    n_compartments = 2  # depot + central
    advan = 2
    output_compartment = 2  # observe central compartment

    _design_cache_max_entries = 256

    def _get_design(
        self,
        dose_events: list[DoseEvent],
        obs_times: np.ndarray,
    ) -> _ADVAN2Design:
        cache = getattr(self, "_design_cache", None)
        if cache is None:
            cache = {}
            self._design_cache = cache

        cache_key = (id(obs_times), id(dose_events))
        entry = cache.get(cache_key)
        if (
            entry is not None
            and entry.obs_times_ref is obs_times
            and entry.dose_events_ref is dose_events
        ):
            return entry.design

        bolus_doses: list[_BolusDoseDesign] = []
        infusion_doses: list[_InfusionDoseDesign] = []
        for dose in dose_events:
            if dose.reset:
                continue
            dt = obs_times - float(dose.time)
            positive_idx = np.flatnonzero(dt > 0.0)
            if len(positive_idx) == 0:
                continue
            dt_pos = dt[positive_idx]
            if dose.is_bolus:
                bolus_doses.append(
                    _BolusDoseDesign(
                        amount=float(dose.amount),
                        positive_idx=positive_idx,
                        dt_pos=dt_pos,
                        ss=dose.ss,
                        ii=float(dose.ii),
                    )
                )
                continue

            rate = float(dose.rate)
            infusion_doses.append(
                _InfusionDoseDesign(
                    amount=float(dose.amount),
                    rate=rate,
                    positive_idx=positive_idx,
                    dt_pos=dt_pos,
                )
            )

        single_bolus = bolus_doses[0] if len(bolus_doses) == 1 and not infusion_doses else None
        design = _ADVAN2Design(
            n_times=len(obs_times),
            bolus_doses=tuple(bolus_doses),
            infusion_doses=tuple(infusion_doses),
            single_bolus=single_bolus,
        )
        cache[cache_key] = _ADVAN2CacheEntry(
            obs_times_ref=obs_times, dose_events_ref=dose_events, design=design
        )
        if len(cache) > self._design_cache_max_entries:
            cache.pop(next(iter(cache)))
        return design

    def solve(
        self,
        pk_params: dict[str, float],
        dose_events: list[DoseEvent],
        obs_times: np.ndarray,
        pk_callable: Callable | None = None,
        des_callable: Callable | None = None,
        *,
        return_amounts: bool = True,
    ) -> PKSolution:
        obs_times = np.asarray(obs_times, dtype=float)
        design = self._get_design(dose_events, obs_times)
        ka = pk_params.get("KA")
        k = pk_params.get("K")
        v = pk_params.get("V")
        f1 = pk_params.get("F1", 1.0)  # Bioavailability fraction

        if ka is None or ka <= 0:
            raise PKError(f"ADVAN2 requires KA > 0, got KA={ka}")
        if k is None or k <= 0:
            raise PKError(f"ADVAN2 requires K > 0, got K={k}")
        if v is None or v <= 0:
            raise PKError(f"ADVAN2 requires V > 0, got V={v}")

        n_times = design.n_times
        if not design.bolus_doses and not design.infusion_doses:
            ipred = np.zeros(n_times, dtype=float)
            amounts = (
                np.zeros((n_times, 2), dtype=float)
                if return_amounts
                else np.empty((n_times, 0), dtype=float)
            )
            return PKSolution(times=obs_times.copy(), amounts=amounts, ipred=ipred)

        a1 = np.zeros(n_times, dtype=float) if return_amounts else None
        a2 = np.zeros(n_times, dtype=float)

        denom = ka - k
        limit_form = abs(denom) < _KA_K_TOL
        bolus_scale: float = 0.0 if limit_form else ka / denom

        if design.single_bolus is not None:
            dose = design.single_bolus
            amt = dose.amount * f1
            if len(dose.positive_idx) > 0:
                exp_ka = np.exp(-ka * dose.dt_pos)
                if dose.ss and dose.ii > 0:
                    tau = dose.ii
                    ss_k = 1.0 / (1.0 - np.exp(-k * tau))
                    ss_ka = 1.0 / (1.0 - np.exp(-ka * tau))
                    if return_amounts and a1 is not None:
                        a1[dose.positive_idx] = amt * ss_ka * exp_ka
                    if limit_form:
                        a2[dose.positive_idx] = amt * ka * dose.dt_pos * np.exp(-k * dose.dt_pos) * ss_k
                    else:
                        a2[dose.positive_idx] = amt * bolus_scale * (
                            ss_k * np.exp(-k * dose.dt_pos) - ss_ka * exp_ka
                        )
                else:
                    if return_amounts and a1 is not None:
                        a1[dose.positive_idx] = amt * exp_ka
                    if limit_form:
                        a2[dose.positive_idx] = amt * ka * dose.dt_pos * np.exp(-k * dose.dt_pos)
                    else:
                        a2[dose.positive_idx] = amt * bolus_scale * (np.exp(-k * dose.dt_pos) - exp_ka)
            ipred = a2 / v
            if return_amounts and a1 is not None:
                amounts = np.empty((n_times, 2), dtype=float)
                amounts[:, 0] = a1
                amounts[:, 1] = a2
            else:
                amounts = np.empty((n_times, 0), dtype=float)
            return PKSolution(times=obs_times.copy(), amounts=amounts, ipred=ipred)

        ss_infusion_warned = False
        for dose in design.bolus_doses:
            amt = dose.amount * f1
            if len(dose.positive_idx) == 0:
                continue
            exp_ka = np.exp(-ka * dose.dt_pos)
            if dose.ss and dose.ii > 0:
                # Steady-state oral bolus: per-pole SS accumulation factors.
                # A2_ss(t) = F*D*KA/(KA-K) * [exp(-K*t)/(1-exp(-K*tau))
                #                              - exp(-KA*t)/(1-exp(-KA*tau))]
                # Reference: Rowland & Tozer, Clinical PK & PD, Ch. 17.
                tau = dose.ii
                ss_k = 1.0 / (1.0 - np.exp(-k * tau))
                ss_ka = 1.0 / (1.0 - np.exp(-ka * tau))
                if return_amounts and a1 is not None:
                    a1[dose.positive_idx] += amt * ss_ka * exp_ka
                if limit_form:
                    a2[dose.positive_idx] += amt * ka * dose.dt_pos * np.exp(-k * dose.dt_pos) * ss_k
                else:
                    a2[dose.positive_idx] += amt * bolus_scale * (
                        ss_k * np.exp(-k * dose.dt_pos) - ss_ka * exp_ka
                    )
            else:
                if return_amounts and a1 is not None:
                    a1[dose.positive_idx] += amt * exp_ka
                if limit_form:
                    a2[dose.positive_idx] += amt * ka * dose.dt_pos * np.exp(-k * dose.dt_pos)
                else:
                    a2[dose.positive_idx] += amt * bolus_scale * (np.exp(-k * dose.dt_pos) - exp_ka)

        for inf_dose in design.infusion_doses:
            if getattr(inf_dose, "ss", False) and not ss_infusion_warned:
                warnings.warn(
                    "ADVAN2: SS=1 with infusion dosing is not yet implemented; "
                    "predictions are computed from a single-dose infusion.",
                    UserWarning,
                    stacklevel=3,
                )
                ss_infusion_warned = True
            amt = inf_dose.amount * f1
            r = inf_dose.rate
            d = amt / r
            dt_pos = inf_dose.dt_pos
            during_mask = dt_pos <= d
            during_idx = inf_dose.positive_idx[during_mask]
            dt_during = dt_pos[during_mask]
            after_idx = inf_dose.positive_idx[~during_mask]
            t_post = dt_pos[~during_mask] - d

            if len(during_idx) > 0:
                if return_amounts and a1 is not None:
                    a1[during_idx] += r / ka * (1 - np.exp(-ka * dt_during))
                if limit_form:
                    a2[during_idx] += _infusion_central_limit(r, k, dt_during)
                else:
                    a2[during_idx] += r * bolus_scale / k * (
                        1 - np.exp(-k * dt_during)
                    ) - r / denom * (1 - np.exp(-ka * dt_during))

            if len(after_idx) > 0:
                a1_end = r / ka * (1 - np.exp(-ka * d))
                if return_amounts and a1 is not None:
                    a1[after_idx] += a1_end * np.exp(-ka * t_post)
                if limit_form:
                    a2_end = _infusion_central_limit(r, k, np.array([d]))[0]
                    a2[after_idx] += a2_end * np.exp(-k * t_post) + a1_end * ka * t_post * np.exp(
                        -k * t_post
                    )
                else:
                    a2_end = r * bolus_scale / k * (1 - np.exp(-k * d)) - r / denom * (
                        1 - np.exp(-ka * d)
                    )
                    a2[after_idx] += a2_end * np.exp(-k * t_post) + a1_end * bolus_scale * (
                        np.exp(-k * t_post) - np.exp(-ka * t_post)
                    )

        ipred = a2 / v
        if return_amounts and a1 is not None:
            amounts = np.empty((n_times, 2), dtype=float)
            amounts[:, 0] = a1
            amounts[:, 1] = a2
        else:
            amounts = np.empty((n_times, 0), dtype=float)

        return PKSolution(times=obs_times.copy(), amounts=amounts, ipred=ipred)


def _infusion_central_limit(r: float, k: float, dt: np.ndarray) -> np.ndarray:
    """Central compartment during infusion in KA≈K limit form (L'Hôpital)."""
    return r * ((1.0 - np.exp(-k * dt)) / k - dt * np.exp(-k * dt))
