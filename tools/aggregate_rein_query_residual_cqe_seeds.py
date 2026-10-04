"""Aggregate the three immutable Phase 16 matched CQE audit artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path


EXPECTED = {
    20260931: {
        "audit_sha256": "c7c21c0e59413058a641e50a5bc695b13c95062d3dea6872f1b7aa8bdabab916",
        "training_git_sha": "86402159d46d364c227baaec78565cd93e7e5921",
    },
    20261001: {
        "audit_sha256": "16045175db6fd7972a91f225916800952041fa4b16a1ee49a8f56b25fa303016",
        "training_git_sha": "682d2b78e724a5dc208b8dbeb42021f7159eef09",
    },
    20261002: {
        "audit_sha256": "d874538a815f4be5bcf666d9f9c570ae63100ccb493a6aa6c2ca08df2312816a",
        "training_git_sha": "e00d5f4ff7f093b669cffcd256337a8928614534",
    },
}
BASELINE_MIOU = 0.6560509975136082
T_CRITICAL_95_DF2 = 4.302652729911275


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
    }


def mean_t_interval_95(values: list[float]) -> list[float]:
    mean = statistics.fmean(values)
    half_width = T_CRITICAL_95_DF2 * statistics.stdev(values) / math.sqrt(len(values))
    return [mean - half_width, mean + half_width]


def one_sided_sign_test_p_value(positive_count: int, sample_count: int) -> float:
    return sum(math.comb(sample_count, k) for k in range(positive_count, sample_count + 1)) / (2 ** sample_count)


def aggregate(paths: list[Path], *, evaluation_git_sha: str) -> dict:
    require(len(paths) == 3, "Exactly three audit artifacts are required")
    rows = []
    seen = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = payload.get("seed")
        require(seed in EXPECTED and seed not in seen, "Unexpected or duplicate seed")
        expected = EXPECTED[seed]
        require(digest(path) == expected["audit_sha256"], f"seed {seed}: audit hash mismatch")
        require(payload.get("formal_paired_audit_ok") is True, f"seed {seed}: audit did not pass")
        require(payload.get("training_git_sha") == expected["training_git_sha"],
                f"seed {seed}: training SHA mismatch")
        require(payload.get("updates_per_arm") == 40000
                and payload.get("microbatches_per_arm") == 160000,
                f"seed {seed}: training exposure mismatch")
        require(payload.get("source_indices_matched") is True
                and payload.get("per_microbatch_inputs_matched") is True,
                f"seed {seed}: paired protocol mismatch")
        require(abs(payload.get("baseline_miou_reference") - BASELINE_MIOU) <= 1e-15,
                f"seed {seed}: baseline mismatch")
        no_cqe = float(payload["no_cqe_miou"])
        cqe = float(payload["cqe_miou"])
        delta = float(payload["cqe_minus_control_miou"])
        require(math.isfinite(no_cqe) and math.isfinite(cqe) and math.isfinite(delta),
                f"seed {seed}: non-finite metric")
        require(abs((cqe - no_cqe) - delta) <= 1e-15,
                f"seed {seed}: delta reconstruction mismatch")
        bootstrap = payload.get("paired_image_bootstrap", {})
        require(bootstrap.get("replicates") == 2000
                and bootstrap.get("confidence_level") == 0.95,
                f"seed {seed}: image bootstrap protocol mismatch")
        rows.append({
            "seed": seed,
            "training_git_sha": payload["training_git_sha"],
            "audit_sha256": expected["audit_sha256"],
            "no_cqe_miou": no_cqe,
            "cqe_miou": cqe,
            "cqe_minus_control_miou": delta,
            "cqe_minus_rein_baseline_miou": cqe - BASELINE_MIOU,
            "paired_image_bootstrap_ci95": bootstrap["delta_miou_cqe_minus_control_ci95"],
        })
        seen.add(seed)
    rows.sort(key=lambda row: row["seed"])
    require(seen == set(EXPECTED), "Three-seed coverage mismatch")

    no_cqe_values = [row["no_cqe_miou"] for row in rows]
    cqe_values = [row["cqe_miou"] for row in rows]
    deltas = [row["cqe_minus_control_miou"] for row in rows]
    positive = sum(value > 0 for value in deltas)
    negative = sum(value < 0 for value in deltas)
    zero = len(deltas) - positive - negative
    training_interval = mean_t_interval_95(deltas)
    mean_delta = statistics.fmean(deltas)
    all_cqe_below_baseline = all(value < BASELINE_MIOU for value in cqe_values)
    interval_excludes_zero = training_interval[0] > 0 or training_interval[1] < 0

    return {
        "ok": True,
        "phase": 16,
        "purpose": "read_only_three_training_seed_cqe_performance_aggregate",
        "evaluation_git_sha": evaluation_git_sha,
        "seed_count": 3,
        "seeds": rows,
        "no_cqe_miou": mean_std(no_cqe_values),
        "cqe_miou": mean_std(cqe_values),
        "paired_cqe_minus_control_miou": {
            **mean_std(deltas),
            "student_t_ci95_df2": training_interval,
            "positive_seed_count": positive,
            "zero_seed_count": zero,
            "negative_seed_count": negative,
            "one_sided_exact_sign_test_p_value": one_sided_sign_test_p_value(positive, len(deltas)),
        },
        "rein_baseline_miou": BASELINE_MIOU,
        "mean_cqe_minus_rein_baseline_miou": statistics.fmean(cqe_values) - BASELINE_MIOU,
        "all_cqe_runs_below_rein_baseline": all_cqe_below_baseline,
        "training_seed_ci_excludes_zero": interval_excludes_zero,
        "directionally_consistent_positive_point_estimates": positive == len(deltas),
        "supports_statistically_resolved_training_seed_improvement": mean_delta > 0 and interval_excludes_zero,
        "conclusion": (
            "CQE has a positive point estimate in all three matched seeds, but the "
            "three-seed confidence interval includes zero and every CQE branch remains "
            "below the unmodified REIN baseline. This is weak directional evidence, not "
            "proof of a robust or practically meaningful performance improvement."
        ),
        "no_additional_long_training_authorized": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audits", nargs=3, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    evaluation_git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    try:
        result = aggregate(args.audits, evaluation_git_sha=evaluation_git_sha)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        result = {"ok": False, "evaluation_git_sha": evaluation_git_sha,
                  "error": f"{type(error).__name__}: {error}"}
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
