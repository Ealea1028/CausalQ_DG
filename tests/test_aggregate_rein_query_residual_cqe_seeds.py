import json
from pathlib import Path

import pytest

from tools.aggregate_rein_query_residual_cqe_seeds import (
    EXPECTED,
    aggregate,
    digest,
    mean_t_interval_95,
    one_sided_sign_test_p_value,
)


def test_training_seed_statistics_match_prespecified_three_seed_values():
    values = [0.0027601538687463956, 0.0001636839096199605,
              3.5010955617531536e-05]
    interval = mean_t_interval_95(values)
    assert interval == pytest.approx([-0.0028332376102417605,
                                      0.0048058034328976856])
    assert one_sided_sign_test_p_value(3, 3) == 0.125


def test_aggregate_checks_hashes_and_reports_nonresolved_result(tmp_path, monkeypatch):
    paths = []
    values = {
        20260931: (0.6501274350304209, 0.6528875888991673),
        20261001: (0.6502510455774392, 0.6504147294870591),
        20261002: (0.6510691679286491, 0.6511041788842666),
    }
    for seed, (no_cqe, cqe) in values.items():
        payload = {
            "formal_paired_audit_ok": True,
            "training_git_sha": EXPECTED[seed]["training_git_sha"],
            "seed": seed,
            "updates_per_arm": 40000,
            "microbatches_per_arm": 160000,
            "source_indices_matched": True,
            "per_microbatch_inputs_matched": True,
            "no_cqe_miou": no_cqe,
            "cqe_miou": cqe,
            "cqe_minus_control_miou": cqe - no_cqe,
            "baseline_miou_reference": 0.6560509975136082,
            "paired_image_bootstrap": {
                "replicates": 2000,
                "confidence_level": 0.95,
                "delta_miou_cqe_minus_control_ci95": [-0.1, 0.1],
            },
        }
        path = tmp_path / f"{seed}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    monkeypatch.setattr(
        "tools.aggregate_rein_query_residual_cqe_seeds.digest",
        lambda path: EXPECTED[int(path.stem)]["audit_sha256"],
    )
    result = aggregate(paths, evaluation_git_sha="a" * 40)
    assert result["paired_cqe_minus_control_miou"]["positive_seed_count"] == 3
    assert result["training_seed_ci_excludes_zero"] is False
    assert result["supports_statistically_resolved_training_seed_improvement"] is False
    assert result["all_cqe_runs_below_rein_baseline"] is True
    assert result["no_additional_long_training_authorized"] is True


def test_digest_reads_binary_file(tmp_path):
    path = tmp_path / "artifact.json"
    path.write_bytes(b"evidence")
    assert digest(path) == "ee8250fb76e094b34b471f13a73dbbe51d1ae142e9df59d7c0d31ec20f0a0a8e"
