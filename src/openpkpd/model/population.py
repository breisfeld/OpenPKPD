"""
PopulationModel: assembles all components and evaluates the population OFV.

The PopulationModel holds:
  - The NONMEMDataset (processed into per-subject SubjectEvents)
  - The PKSubroutine
  - Compiled $PK and $ERROR callables
  - A ParameterSet (current theta/omega/sigma)
  - The EstimationRecord (method, options)

It provides:
  - evaluate_individual(subject_id, eta) → IPREDs, LL
  - ofv(params) → total OFV for use by outer-loop optimizer
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from openpkpd.data.dataset import NONMEMDataset
from openpkpd.data.event_processor import EventProcessor, SubjectEvents
from openpkpd.model.individual import IndividualModel
from openpkpd.model.parameters import ParameterSet
from openpkpd.pk.base import PKSubroutine
from openpkpd.utils.constants import BLQMethod
from openpkpd.utils.errors import ModelError


@dataclass
class PopulationModel:
    """
    Top-level assembled population PK/PD model.

    Created by the runner/CLI after parsing the control stream.
    """

    dataset: NONMEMDataset
    pk_subroutine: PKSubroutine
    params: ParameterSet

    pk_callable: Callable | None = None
    error_callable: Callable | None = None
    des_callable: Callable | None = None

    trans: int = 2
    advan: int = 2
    covariate_columns: list[str] = field(default_factory=list)
    blq_method: str = BLQMethod.M1  # A3: BLQ method propagated from $ESTIMATION

    # Cached per-subject data
    _subject_events: dict[int, SubjectEvents] = field(default_factory=dict, repr=False)
    _individual_models: dict[int, IndividualModel] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._setup_subjects()

    def _setup_subjects(self) -> None:
        """Pre-process dataset into per-subject event structures."""
        processor = EventProcessor(covariate_columns=self.covariate_columns)
        self._subject_events = processor.process(self.dataset.df)
        # A3: resolve LLOQ per subject if dataset has LLOQ column
        has_lloq = self.dataset.has_lloq
        self._individual_models = {
            subj_id: IndividualModel(
                subject_events=events,
                pk_subroutine=self.pk_subroutine,
                pk_callable=self.pk_callable,
                error_callable=self.error_callable,
                n_eps=self.params.n_eps(),
                des_callable=self.des_callable,
                blq_method=self.blq_method,
                lloq=self.dataset.lloq_values(subj_id) if has_lloq else None,
                occasion_indices=events.occasion_indices,
            )
            for subj_id, events in self._subject_events.items()
        }

    def subject_ids(self) -> list[int]:
        return sorted(self._subject_events.keys())

    def n_subjects(self) -> int:
        return len(self._subject_events)

    def individual_model(self, subject_id: int) -> IndividualModel:
        if subject_id not in self._individual_models:
            raise ModelError(f"Subject {subject_id} not found in dataset")
        return self._individual_models[subject_id]

    def evaluate_individual(
        self,
        subject_id: int,
        eta: np.ndarray,
        params: ParameterSet | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate predictions for a single subject.

        Returns:
            (ipred, obs_mask, f) arrays.
        """
        p = params or self.params
        indiv = self.individual_model(subject_id)
        return indiv.evaluate(p.theta, eta, p.sigma, trans=self.trans)

    def ofv_fo(self, params: ParameterSet) -> float:
        """
        First-Order OFV: evaluate at eta=0 for all subjects.

        OFV_FO = Σ_i { -2 * log p(y_i | eta=0, theta) + log|C_i(0)| }

        where C_i(0) = R_i * Ω * R_i^T + Σ  (linearization at eta=0)
        """
        ofv = 0.0
        eta_zero = np.zeros(params.n_eta())
        for subj_id in self.subject_ids():
            indiv = self.individual_model(subj_id)
            try:
                obj = indiv.log_likelihood(params.theta, eta_zero, params.sigma, trans=self.trans)
                ofv += obj
            except Exception:
                ofv += 1e10  # Penalize failures
        return ofv

    def ofv_foce(
        self,
        params: ParameterSet,
        eta_hat: dict[int, np.ndarray],
    ) -> float:
        """
        FOCE OFV given post-hoc ETAs.

        OFV_FOCE = Σ_i [ log|C_i(η̂_i)| + (y_i-f_i)^T C_i^{-1} (y_i-f_i) + η̂_i^T Ω^{-1} η̂_i ]
        """
        ofv = 0.0
        for subj_id in self.subject_ids():
            eta_i = eta_hat.get(subj_id, np.zeros(params.n_eta()))
            indiv = self.individual_model(subj_id)
            try:
                obj = indiv.obj_eta(
                    eta_i, params.theta, params.omega, params.sigma, trans=self.trans
                )
                ofv += obj
            except Exception:
                ofv += 1e10
        return ofv

    def post_hoc_etas(
        self,
        params: ParameterSet | None = None,
    ) -> dict[int, np.ndarray]:
        """
        Return initial (zero) ETAs for all subjects.

        Actual post-hoc optimization is done in the FOCE inner loop.
        """
        p = params or self.params
        return {sid: np.zeros(p.n_eta()) for sid in self.subject_ids()}
