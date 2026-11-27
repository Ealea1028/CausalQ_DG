import sys

import pytest

from causalq.rein_head import load_rein_mask2former_head


def test_loads_head_once_without_initializing_rein_package(tmp_path, monkeypatch):
    package = tmp_path / "rein"
    source = package / "models/heads/rein_mask2former.py"
    source.parent.mkdir(parents=True)
    (package / "__init__.py").write_text("raise ImportError('optional backbone')\n")
    source.write_text("class ReinMask2FormerHead: pass\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(sys.modules, "causalq._upstream_rein_mask2former", None)

    head = load_rein_mask2former_head(tmp_path)
    assert head.__name__ == "ReinMask2FormerHead"
    assert load_rein_mask2former_head(tmp_path) is head


def test_reports_original_import_error_and_allows_retry(tmp_path, monkeypatch):
    source = tmp_path / "rein/models/heads/rein_mask2former.py"
    source.parent.mkdir(parents=True)
    source.write_text("raise ImportError('missing head dependency')\n")
    monkeypatch.setitem(sys.modules, "causalq._upstream_rein_mask2former", None)

    with pytest.raises(ImportError, match="missing head dependency"):
        load_rein_mask2former_head(tmp_path)

    source.write_text("class ReinMask2FormerHead: pass\n")
    assert load_rein_mask2former_head(tmp_path).__name__ == "ReinMask2FormerHead"
