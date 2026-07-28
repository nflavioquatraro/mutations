"""Executable acceptance tests for the public workflow notebooks."""

from __future__ import annotations

import ast
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_CASES = (
    (
        PROJECT_ROOT / "notebooks" / "00_inspect_base_model.ipynb",
        "NB-001",
        "NB-001 validation passed",
        "inspection_summary",
    ),
    (
        PROJECT_ROOT / "notebooks" / "01_convert_dense_model.ipynb",
        "NB-002",
        "NB-002 validation passed",
        "validation_summary",
    ),
    (
        PROJECT_ROOT / "notebooks" / "02_train_expert_package.ipynb",
        "NB-003",
        "NB-003 validation passed",
        "validation_summary",
    ),
    (
        PROJECT_ROOT / "notebooks" / "03_fit_and_calibrate_router.ipynb",
        "NB-004",
        "NB-004 validation passed",
        "calibration_summary",
    ),
    (
        PROJECT_ROOT / "notebooks" / "04_compose_custom_model.ipynb",
        "NB-005",
        "NB-005 validation passed",
        "composition_summary",
    ),
    (
        PROJECT_ROOT / "notebooks" / "05_evaluate_registry.ipynb",
        "NB-006",
        "NB-006 validation passed",
        "evaluation_summary",
    ),
    (
        PROJECT_ROOT / "notebooks" / "06_export_and_reload.ipynb",
        "NB-007",
        "NB-007 validation passed",
        "lifecycle_summary",
    ),
)


def _load_notebook(path: Path) -> dict[str, Any]:
    """Load a notebook as JSON without requiring the optional Jupyter stack."""

    with path.open(encoding="utf-8") as notebook_file:
        notebook = json.load(notebook_file)
    if not isinstance(notebook, dict):
        raise TypeError(f"Notebook root must be an object: {path}")
    return notebook


def _cell_source(cell: dict[str, Any]) -> str:
    """Normalize the string-or-line-list source representation used by nbformat."""

    source = cell.get("source", "")
    if isinstance(source, str):
        return source
    if isinstance(source, list) and all(isinstance(line, str) for line in source):
        return "".join(source)
    raise TypeError("Notebook cell source must be a string or list of strings.")


@pytest.mark.parametrize(
    ("notebook_path", "feature_id", "success_message", "summary_name"),
    NOTEBOOK_CASES,
)
def test_foundation_notebook_is_thin_and_executes_offline(
    notebook_path: Path,
    feature_id: str,
    success_message: str,
    summary_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute cells sequentially and enforce the thin-notebook contract."""

    monkeypatch.setenv("ANCHORED_MOE_SMOKE_TEST", "1")
    notebook = _load_notebook(notebook_path)

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    cells = notebook["cells"]
    assert cells
    assert feature_id in "\n".join(_cell_source(cell) for cell in cells)

    namespace: dict[str, Any] = {"__name__": "__notebook__"}
    standard_output = io.StringIO()
    code_cell_count = 0
    with redirect_stdout(standard_output):
        for index, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue

            code_cell_count += 1
            source = _cell_source(cell)
            syntax_tree = ast.parse(source, filename=f"{notebook_path.name}:cell-{index}")
            assert not any(
                isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                for node in ast.walk(syntax_tree)
            ), "Reusable definitions belong in the anchored_moe package."
            assert "/Users/" not in source
            assert "sys.path" not in source
            assert cell["execution_count"] is None
            assert cell["outputs"] == []

            exec(
                compile(
                    syntax_tree,
                    filename=f"{notebook_path.name}:cell-{index}",
                    mode="exec",
                ),
                namespace,
            )

    assert code_cell_count >= 3
    assert success_message in standard_output.getvalue()
    summary = namespace[summary_name]
    assert summary["feature_id"] == feature_id


def test_notebook_sequence_matches_the_documented_workflow() -> None:
    """Keep configuration, operation, validation, and summary visibly ordered."""

    for notebook_path, *_ in NOTEBOOK_CASES:
        notebook = _load_notebook(notebook_path)
        markdown = [
            _cell_source(cell)
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        ]

        resolve_index = next(
            index for index, text in enumerate(markdown) if "Resolve configuration" in text
        )
        validation_index = next(
            index
            for index, text in enumerate(markdown)
            if "Validat" in text or "Inspect the backend contract" in text
        )
        assert resolve_index < validation_index
