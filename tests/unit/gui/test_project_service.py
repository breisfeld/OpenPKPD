"""Unit tests for ProjectService — workspace lifecycle mutations."""

from __future__ import annotations

import pytest

from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.dataset_asset import DatasetAsset
from openpkpd_gui.domain.model_spec import ModelSpec
from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Project, Scenario, Workspace
from openpkpd_gui.services.project_service import ProjectService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _svc() -> ProjectService:
    return ProjectService()


def _ws(name: str = "W") -> Workspace:
    return Workspace(name=name)


def _run(workflow: str = "fit", status: RunStatus = RunStatus.SUCCEEDED) -> RunRecord:
    run = RunRecord(workflow=workflow)
    if status == RunStatus.RUNNING:
        run.mark_running()
    elif status == RunStatus.SUCCEEDED:
        run.mark_running()
        run.mark_succeeded()
    elif status == RunStatus.FAILED:
        run.mark_running()
        run.mark_failed("err")
    return run


def _artifact() -> ArtifactRecord:
    return ArtifactRecord(kind="report", label="Report", path="/tmp/r.html")


def _dataset(path: str = "/data/demo.csv") -> DatasetAsset:
    return DatasetAsset(source_path=path, display_name="demo.csv")


def _model_spec() -> ModelSpec:
    return ModelSpec(
        problem_title="Demo",
        pk_code="CL = THETA(1)",
        error_code="Y = F + EPS(1)",
        theta_rows=[{"init": 1.0}],
        omega_values=[[0.1]],
        sigma_values=[[0.05]],
    )


# ---------------------------------------------------------------------------
# new_workspace
# ---------------------------------------------------------------------------


def test_new_workspace_default_name() -> None:
    ws = _svc().new_workspace()
    assert ws.name == "Untitled Workspace"
    assert len(ws.projects) == 1


def test_new_workspace_custom_name_and_root() -> None:
    ws = _svc().new_workspace(name="MyWS", root_path="/tmp")
    assert ws.name == "MyWS"
    assert ws.root_path == "/tmp"


# ---------------------------------------------------------------------------
# create_project
# ---------------------------------------------------------------------------


def test_create_project_appends_and_activates() -> None:
    ws = _ws()
    first_id = ws.active_project_id
    project = _svc().create_project(ws, name="New Project")

    assert project in ws.projects
    assert ws.active_project_id == project.project_id
    assert project.project_id != first_id


def test_create_project_auto_name() -> None:
    ws = _ws()
    project = _svc().create_project(ws)
    assert project.name.startswith("Project")


# ---------------------------------------------------------------------------
# rename_project
# ---------------------------------------------------------------------------


def test_rename_project_active() -> None:
    ws = _ws()
    _svc().rename_project(ws, name="Renamed")
    assert ws.active_project.name == "Renamed"


def test_rename_project_by_id() -> None:
    ws = _ws()
    pid = ws.active_project_id
    _svc().rename_project(ws, project_id=pid, name="ByID")
    assert ws.active_project.name == "ByID"


def test_rename_project_strips_whitespace() -> None:
    ws = _ws()
    _svc().rename_project(ws, name="  Padded  ")
    assert ws.active_project.name == "Padded"


def test_rename_project_blank_keeps_existing() -> None:
    ws = _ws()
    original = ws.active_project.name
    _svc().rename_project(ws, name="   ")
    assert ws.active_project.name == original


def test_rename_project_unknown_id_raises() -> None:
    ws = _ws()
    with pytest.raises(KeyError):
        _svc().rename_project(ws, project_id="nonexistent", name="X")


# ---------------------------------------------------------------------------
# update_project_details / set_project_notes
# ---------------------------------------------------------------------------


def test_update_project_details_persists_all_fields() -> None:
    ws = _ws()
    _svc().update_project_details(
        ws, name="New", description="Desc", references="Ref", notes="Notes"
    )
    p = ws.active_project
    assert p.name == "New"
    assert p.metadata["description"] == "Desc"
    assert p.metadata["references"] == "Ref"
    assert p.metadata["notes"] == "Notes"


def test_update_project_details_empty_clears_keys() -> None:
    ws = _ws()
    _svc().update_project_details(ws, name="X", description="D", references="", notes="")
    p = ws.active_project
    assert "references" not in p.metadata
    assert "notes" not in p.metadata


def test_set_project_notes_adds_and_clears() -> None:
    ws = _ws()
    svc = _svc()
    svc.set_project_notes(ws, notes="hello")
    assert ws.active_project.metadata["notes"] == "hello"
    svc.set_project_notes(ws, notes="")
    assert "notes" not in ws.active_project.metadata


# ---------------------------------------------------------------------------
# delete_project
# ---------------------------------------------------------------------------


def test_delete_project_removes_it() -> None:
    ws = _ws()
    svc = _svc()
    p2 = svc.create_project(ws, name="Second")
    svc.delete_project(ws)  # deletes active (p2)
    assert p2 not in ws.projects
    assert len(ws.projects) == 1


def test_delete_project_updates_active_to_neighbour() -> None:
    ws = _ws()
    svc = _svc()
    p1 = ws.active_project
    p2 = svc.create_project(ws, name="Second")
    assert ws.active_project_id == p2.project_id
    svc.delete_project(ws)  # deletes p2 (active)
    assert ws.active_project_id == p1.project_id


def test_delete_project_last_raises() -> None:
    ws = _ws()
    with pytest.raises(ValueError, match="at least one project"):
        _svc().delete_project(ws)


def test_delete_project_by_id() -> None:
    ws = _ws()
    svc = _svc()
    p2 = svc.create_project(ws, name="ToDelete")
    svc.delete_project(ws, project_id=p2.project_id)
    assert p2 not in ws.projects


# ---------------------------------------------------------------------------
# duplicate_project
# ---------------------------------------------------------------------------


def test_duplicate_project_is_independent() -> None:
    ws = _ws()
    original = ws.active_project
    original.scenarios[0].name = "BaseScenario"
    dup = _svc().duplicate_project(ws)

    assert dup in ws.projects
    assert dup.project_id != original.project_id
    assert dup.scenarios[0].scenario_id != original.scenarios[0].scenario_id


def test_duplicate_project_default_name_suffix() -> None:
    ws = _ws()
    ws.active_project.name = "Alpha"
    dup = _svc().duplicate_project(ws)
    assert "Alpha" in dup.name and "Copy" in dup.name


def test_duplicate_project_with_copy_outputs() -> None:
    ws = _ws()
    ws.active_scenario.add_run(_run())
    ws.active_scenario.add_artifact(_artifact())
    dup = _svc().duplicate_project(ws, copy_outputs=True)
    assert len(dup.scenarios[0].runs) == 1
    assert len(dup.scenarios[0].artifacts) == 1


def test_duplicate_project_without_copy_outputs_clears() -> None:
    ws = _ws()
    ws.active_scenario.add_run(_run())
    dup = _svc().duplicate_project(ws, copy_outputs=False)
    assert dup.scenarios[0].runs == []


def test_duplicate_project_remaps_scenario_parent_ids() -> None:
    ws = _ws()
    svc = _svc()
    child = svc.create_scenario(ws)
    original_parent_id = child.parent_scenario_id
    dup = svc.duplicate_project(ws)

    # The child scenario in the clone should point to the cloned parent,
    # not the original parent.
    dup_child = next(s for s in dup.scenarios if s.name == child.name)
    assert dup_child.parent_scenario_id != original_parent_id
    dup_parent_ids = {s.scenario_id for s in dup.scenarios}
    assert dup_child.parent_scenario_id in dup_parent_ids


# ---------------------------------------------------------------------------
# import_project
# ---------------------------------------------------------------------------


def test_import_project_adds_clone() -> None:
    source_ws = _ws("source")
    target_ws = _ws("target")
    imported = _svc().import_project(target_ws, source_ws)

    assert imported in target_ws.projects
    assert imported.project_id != source_ws.active_project.project_id


def test_import_project_copies_outputs() -> None:
    source_ws = _ws()
    source_ws.active_scenario.add_run(_run())
    target_ws = _ws()
    imported = _svc().import_project(target_ws, source_ws)
    assert len(imported.scenarios[0].runs) == 1


# ---------------------------------------------------------------------------
# create_scenario
# ---------------------------------------------------------------------------


def test_create_scenario_clones_active() -> None:
    ws = _ws()
    ws.active_dataset = _dataset()
    scenario = _svc().create_scenario(ws)

    assert scenario in ws.active_project.scenarios
    assert scenario.active_dataset is not None
    assert scenario.scenario_id != ws.active_project.scenarios[0].scenario_id


def test_create_scenario_auto_name() -> None:
    ws = _ws()
    s = _svc().create_scenario(ws)
    assert s.name.startswith("Scenario")


def test_create_scenario_sets_parent_id() -> None:
    ws = _ws()
    baseline_id = ws.active_scenario.scenario_id
    child = _svc().create_scenario(ws)
    assert child.parent_scenario_id == baseline_id


# ---------------------------------------------------------------------------
# rename_scenario
# ---------------------------------------------------------------------------


def test_rename_scenario_active() -> None:
    ws = _ws()
    _svc().rename_scenario(ws, name="Renamed")
    assert ws.active_scenario.name == "Renamed"


def test_rename_scenario_blank_keeps_existing() -> None:
    ws = _ws()
    original = ws.active_scenario.name
    _svc().rename_scenario(ws, name="")
    assert ws.active_scenario.name == original


# ---------------------------------------------------------------------------
# update_scenario_details / set_scenario_notes
# ---------------------------------------------------------------------------


def test_update_scenario_details_persists() -> None:
    ws = _ws()
    _svc().update_scenario_details(
        ws, name="S", description="D", references="R", notes="N"
    )
    s = ws.active_scenario
    assert s.name == "S"
    assert s.metadata["description"] == "D"


def test_set_scenario_notes_roundtrip() -> None:
    ws = _ws()
    svc = _svc()
    svc.set_scenario_notes(ws, notes="note text")
    assert ws.active_scenario.metadata["notes"] == "note text"
    svc.set_scenario_notes(ws, notes="")
    assert "notes" not in ws.active_scenario.metadata


# ---------------------------------------------------------------------------
# duplicate_scenario
# ---------------------------------------------------------------------------


def test_duplicate_scenario_is_independent() -> None:
    ws = _ws()
    ws.active_dataset = _dataset()
    original_id = ws.active_scenario.scenario_id
    dup = _svc().duplicate_scenario(ws)

    assert dup.scenario_id != original_id
    assert dup.parent_scenario_id == original_id
    assert dup.active_dataset is not None


def test_duplicate_scenario_default_name_suffix() -> None:
    ws = _ws()
    ws.active_scenario.name = "Beta"
    dup = _svc().duplicate_scenario(ws)
    assert "Beta" in dup.name and "Copy" in dup.name


# ---------------------------------------------------------------------------
# import_scenario
# ---------------------------------------------------------------------------


def test_import_scenario_cross_workspace() -> None:
    source_ws = _ws("source")
    source_ws.active_scenario.add_run(_run())
    target_ws = _ws("target")
    imported = _svc().import_scenario(target_ws, source_ws)

    assert imported in target_ws.active_project.scenarios
    assert len(imported.runs) == 1


def test_import_scenario_unknown_scenario_raises() -> None:
    source_ws = _ws()
    target_ws = _ws()
    with pytest.raises(KeyError):
        _svc().import_scenario(
            target_ws, source_ws, source_scenario_id="nonexistent"
        )


# ---------------------------------------------------------------------------
# delete_scenario
# ---------------------------------------------------------------------------


def test_delete_scenario_removes_it() -> None:
    ws = _ws()
    svc = _svc()
    s2 = svc.create_scenario(ws)
    svc.delete_scenario(ws)  # deletes active (s2)
    assert s2 not in ws.active_project.scenarios


def test_delete_scenario_updates_active_to_neighbour() -> None:
    ws = _ws()
    svc = _svc()
    baseline = ws.active_project.scenarios[0]
    svc.create_scenario(ws)  # now active
    svc.delete_scenario(ws)
    assert ws.active_project.active_scenario_id == baseline.scenario_id


def test_delete_scenario_last_raises() -> None:
    ws = _ws()
    with pytest.raises(ValueError, match="at least one scenario"):
        _svc().delete_scenario(ws)


def test_delete_scenario_by_id() -> None:
    ws = _ws()
    svc = _svc()
    s2 = svc.create_scenario(ws)
    s2_id = s2.scenario_id
    # Switch back to baseline so s2 is not active
    ws.active_project.set_active_scenario(ws.active_project.scenarios[0].scenario_id)
    svc.delete_scenario(ws, scenario_id=s2_id)
    assert ws.active_project.find_scenario(s2_id) is None


# ---------------------------------------------------------------------------
# select_project / select_scenario
# ---------------------------------------------------------------------------


def test_select_project_changes_active() -> None:
    ws = _ws()
    svc = _svc()
    p2 = svc.create_project(ws, name="P2")
    svc.create_project(ws, name="P3")
    svc.select_project(ws, p2.project_id)
    assert ws.active_project_id == p2.project_id


def test_select_scenario_changes_active_and_project() -> None:
    ws = _ws()
    svc = _svc()
    p2 = svc.create_project(ws, name="P2")
    s = p2.scenarios[0]
    # Start with a different active project
    svc.select_project(ws, ws.projects[0].project_id)
    svc.select_scenario(ws, s.scenario_id)
    assert ws.active_project_id == p2.project_id
    assert ws.active_project.active_scenario_id == s.scenario_id


# ---------------------------------------------------------------------------
# attach_dataset
# ---------------------------------------------------------------------------


def test_attach_dataset_sets_dataset() -> None:
    ws = _ws()
    ds = _dataset()
    _svc().attach_dataset(ws, ds)
    assert ws.active_dataset is not None
    assert ws.active_dataset.source_path == ds.source_path


def test_attach_dataset_different_clears_outputs() -> None:
    ws = _ws()
    ws.active_scenario.add_run(_run())
    ws.active_scenario.add_artifact(_artifact())
    _svc().attach_dataset(ws, _dataset("/new/path.csv"))
    assert ws.active_scenario.runs == []
    assert ws.active_scenario.artifacts == []


def test_attach_dataset_same_preserves_outputs() -> None:
    ws = _ws()
    ds = _dataset()
    _svc().attach_dataset(ws, ds)
    ws.active_scenario.add_run(_run())
    _svc().attach_dataset(ws, _dataset())  # same path
    assert len(ws.active_scenario.runs) == 1


def test_attach_dataset_sets_root_path_if_unset() -> None:
    ws = _ws()
    assert ws.root_path is None
    _svc().attach_dataset(ws, _dataset("/data/demo.csv"))
    assert ws.root_path is not None


# ---------------------------------------------------------------------------
# set_model_spec
# ---------------------------------------------------------------------------


def test_set_model_spec_sets_spec() -> None:
    ws = _ws()
    spec = _model_spec()
    _svc().set_model_spec(ws, spec)
    assert ws.active_model_spec is not None
    assert ws.active_model_spec.problem_title == "Demo"


def test_set_model_spec_different_clears_outputs() -> None:
    ws = _ws()
    _svc().set_model_spec(ws, _model_spec())
    ws.active_scenario.add_run(_run())
    spec2 = _model_spec()
    spec2.problem_title = "Other"
    _svc().set_model_spec(ws, spec2)
    assert ws.active_scenario.runs == []


# ---------------------------------------------------------------------------
# add_run / add_artifact
# ---------------------------------------------------------------------------


def test_add_run_appends() -> None:
    ws = _ws()
    run = _run()
    _svc().add_run(ws, run)
    assert run in ws.active_scenario.runs


def test_add_artifact_appends() -> None:
    ws = _ws()
    art = _artifact()
    _svc().add_artifact(ws, art)
    assert art in ws.active_scenario.artifacts


# ---------------------------------------------------------------------------
# remember_recent_file / clear_recent_files
# ---------------------------------------------------------------------------


def test_remember_recent_file_prepends() -> None:
    ws = _ws()
    svc = _svc()
    svc.remember_recent_file(ws, "/a/b.json")
    svc.remember_recent_file(ws, "/c/d.json")
    assert ws.recent_files[0].endswith("d.json")
    assert ws.recent_files[1].endswith("b.json")


def test_remember_recent_file_deduplicates() -> None:
    ws = _ws()
    svc = _svc()
    svc.remember_recent_file(ws, "/a/b.json")
    svc.remember_recent_file(ws, "/c/d.json")
    svc.remember_recent_file(ws, "/a/b.json")
    assert len(ws.recent_files) == 2
    assert ws.recent_files[0].endswith("b.json")


def test_remember_recent_file_respects_limit() -> None:
    ws = _ws()
    svc = _svc()
    for i in range(10):
        svc.remember_recent_file(ws, f"/path/{i}.json")
    assert len(ws.recent_files) == ProjectService.MAX_RECENT_FILES


def test_clear_recent_files() -> None:
    ws = _ws()
    svc = _svc()
    svc.remember_recent_file(ws, "/a.json")
    svc.clear_recent_files(ws)
    assert ws.recent_files == []
