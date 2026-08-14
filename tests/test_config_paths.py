"""
Tests for resolving config-referenced paths against the config file.

The pipeline is never fully constructed here: _resolve_config_relative only
needs _config_dir, so an empty shell object keeps the tests free of any
model, store or network dependency.
"""

from pathlib import Path

from ragwire import RAGWire


def _shell(config_dir):
    rag = object.__new__(RAGWire)
    rag._config_dir = Path(config_dir)
    return rag


def test_a_relative_path_anchors_to_the_config_directory(tmp_path):
    (tmp_path / "metadata.yaml").write_text("fields: []", encoding="utf-8")
    rag = _shell(tmp_path)

    resolved = rag._resolve_config_relative("metadata.yaml")

    assert resolved == str(tmp_path / "metadata.yaml")


def test_the_anchor_holds_from_any_working_directory(tmp_path, monkeypatch):
    # The point of the fix: an MCP client or service manager launches the
    # process from wherever it likes, and the reference must not move.
    (tmp_path / "metadata.yaml").write_text("fields: []", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    rag = _shell(tmp_path)

    assert rag._resolve_config_relative("metadata.yaml") == str(tmp_path / "metadata.yaml")


def test_an_absolute_path_is_passed_through(tmp_path):
    rag = _shell(tmp_path)
    absolute = str(tmp_path / "metadata.yaml")

    assert rag._resolve_config_relative(absolute) == absolute


def test_a_cwd_relative_path_still_works_as_a_fallback(tmp_path, monkeypatch):
    # The pre-fix behaviour: a path that only exists relative to the working
    # directory keeps resolving, so old setups do not break.
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    (cwd / "metadata.yaml").write_text("fields: []", encoding="utf-8")
    monkeypatch.chdir(cwd)
    rag = _shell(config_dir)

    assert rag._resolve_config_relative("metadata.yaml") == "metadata.yaml"


def test_a_missing_path_prefers_the_config_directory_for_its_error(tmp_path):
    # Neither location has the file. The anchored form is returned so the
    # eventual FileNotFoundError points beside the config, not at the CWD.
    rag = _shell(tmp_path)

    assert rag._resolve_config_relative("gone.yaml") == str(tmp_path / "gone.yaml")
