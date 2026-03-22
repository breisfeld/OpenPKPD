"""Serializable workspace and workflow models for the desktop GUI."""

from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.dataset_asset import DatasetAsset
from openpkpd_gui.domain.model_spec import CovarianceConfig, EstimationConfig, ModelSpec
from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Project, Scenario, Workspace

__all__ = [
    "ArtifactRecord",
    "CovarianceConfig",
    "DatasetAsset",
    "EstimationConfig",
    "ModelSpec",
    "Project",
    "RunRecord",
    "RunStatus",
    "Scenario",
    "Workspace",
]
