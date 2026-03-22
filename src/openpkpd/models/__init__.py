"""Extended model types for pharmacometric analysis.

Provides parametric time-to-event, count, categorical, Markov, and hidden
Markov models that operate independently of the core PK model infrastructure.
"""

from __future__ import annotations

from openpkpd.models.categorical import (
    CategoricalData,
    CategoricalResult,
    ContinuousTimeMarkovModel,
    DiscreteTimeMarkovModel,
    ProportionalOddsModel,
)
from openpkpd.models.count import (
    CountData,
    CountModel,
    CountResult,
    NegativeBinomialModel,
    PoissonModel,
    ZeroInflatedPoissonModel,
)
from openpkpd.models.ddi import (
    DDIResult,
    DDIStudyAnalysis,
    competitive_inhibition_r,
    induction_r,
    time_dependent_inhibition_r,
)
from openpkpd.models.markov import (
    ContinuousTimeHMM,
    HMMData,
    HMMResult,
)
from openpkpd.models.pkpd import (
    EffectCompartmentModel,
    EmaxModel,
    HillModel,
    IndirectResponseModel,
    InhibEmaxModel,
    LinearPDModel,
    PDData,
    PDModel,
    PDResult,
    PlaceboResponseModel,
    SequentialPKPDWorkflow,
    TumorGrowthInhibitionModel,
    TurnoverModel,
)
from openpkpd.models.tmdd import (
    FullTMDD,
    MichaelisMentenTMDD,
    QSSATMDDModel,
)
from openpkpd.models.tte import (
    ConstantHazardModel,
    GompertzHazardModel,
    HazardFunction,
    LogLogisticHazardModel,
    RepeatedTTEModel,
    TTEData,
    TTEModel,
    TTEResult,
    WeibullHazardModel,
)

__all__ = [
    # TTE
    "TTEData",
    "TTEModel",
    "TTEResult",
    "HazardFunction",
    "ConstantHazardModel",
    "WeibullHazardModel",
    "GompertzHazardModel",
    "LogLogisticHazardModel",
    "RepeatedTTEModel",
    # Count
    "CountData",
    "CountModel",
    "CountResult",
    "PoissonModel",
    "NegativeBinomialModel",
    "ZeroInflatedPoissonModel",
    # Categorical / Markov
    "CategoricalData",
    "CategoricalResult",
    "ProportionalOddsModel",
    "DiscreteTimeMarkovModel",
    "ContinuousTimeMarkovModel",
    # HMM
    "HMMData",
    "HMMResult",
    "ContinuousTimeHMM",
    # PD / PK-PD
    "PDData",
    "PDResult",
    "PDModel",
    "LinearPDModel",
    "EmaxModel",
    "HillModel",
    "InhibEmaxModel",
    "IndirectResponseModel",
    "EffectCompartmentModel",
    "TurnoverModel",
    "PlaceboResponseModel",
    "TumorGrowthInhibitionModel",
    "SequentialPKPDWorkflow",
    # TMDD
    "FullTMDD",
    "QSSATMDDModel",
    "MichaelisMentenTMDD",
    # DDI
    "DDIResult",
    "competitive_inhibition_r",
    "time_dependent_inhibition_r",
    "induction_r",
    "DDIStudyAnalysis",
]
