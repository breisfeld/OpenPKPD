"""Workflow definitions used by the shell."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WorkflowDefinition:
    """Static metadata for one top-level workflow."""

    workflow_id: str
    label: str
    description: str
    section: str = "Project"


DEFAULT_WORKFLOWS: tuple[WorkflowDefinition, ...] = (
    WorkflowDefinition(
        "dashboard",
        "Dashboard",
        "Scenario readiness, workspace actions, and recommended next steps.",
        "Project",
    ),
    WorkflowDefinition("data", "Data", "Load and validate analysis datasets.", "Inputs"),
    WorkflowDefinition(
        "model", "Model", "Author models in builder or control-stream mode.", "Inputs"
    ),
    WorkflowDefinition(
        "fit", "Fit", "Configure estimation and launch background runs.", "Analyses"
    ),
    WorkflowDefinition("nca", "NCA", "Run standalone non-compartmental analyses.", "Analyses"),
    WorkflowDefinition(
        "covariate",
        "Covariate",
        "Run stepwise covariate search (SCM) against the active base model.",
        "Analyses",
    ),
    WorkflowDefinition(
        "advanced", "Advanced", "Access VPC, bootstrap, and future tools.", "Analyses"
    ),
    WorkflowDefinition("results", "Results", "Inspect run summaries and saved outputs.", "Review"),
    WorkflowDefinition(
        "diagnostics",
        "Diagnostics",
        "Review plots, tables, and residual summaries.",
        "Review",
    ),
)
