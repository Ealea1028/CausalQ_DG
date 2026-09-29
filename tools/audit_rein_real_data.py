"""Read-only audit of the saved Phase 16 real-data gate; no GPU imports."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from tools.check_rein_backbone import REIN_SHA, WEIGHT_SHA
from tools.check_rein_runtime import EXPECTED_VERSIONS


SOURCE_SHA = "17883de0c87abe3ed9bd051b260ba20bc259999e"
REPORT_SHA = "9ed6e11b7e66e481aac9ea7a7d8b2a8f8d090b592acfa91c97d71d7b8d2a8ab5"
LOG_SHA = "186be4d21cfb4b3ae1f84d5c08b198ab37404df762bb7b4079c261218b31f2c1"


def audit_report(report):
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    for key, value in dict(ok=True, real_data_smoke_ok=True, stage="complete",
                          phase=16, git_sha=SOURCE_SHA, rein_sha=REIN_SHA,
                          checkpoint_sha256=WEIGHT_SHA, optimizer_steps=20,
                          adapter_parameter_changed=True,
                          frozen_patch_weight_unchanged=True,
                          frozen_backbone_gradients_absent=True,
                          target_inference_count=5, loaded_tensor_count=343,
                          dtype="float32", seed=20260929).items():
        require(report.get(key) == value, f"Unexpected {key}: {report.get(key)!r}")
    require(report.get("versions") == EXPECTED_VERSIONS, "Runtime version mismatch")
    require(report.get("purpose") == "real_data_20step_optimization_smoke_not_accuracy_evaluation",
            "Unexpected report purpose")
    inventory = report["data_inventory"]
    require(inventory["source_pairs"] == 24966 and inventory["target_pairs"] == 500,
            "Dataset pairing counts mismatch")
    protocol = report["smoke_protocol"]
    for key, value in dict(steps=20, target_inference_samples=5, data_seed=0,
                          optimizer="AdamW", lr=1e-4, weight_decay=0.05,
                          betas=[0.9, 0.999], clip_norm=1.0, batch_size=1,
                          workers=0, dtype="float32", crop_size=[512, 512],
                          accuracy_evaluation=False, checkpoint_saved=False,
                          formal_baseline=False).items():
        require(protocol.get(key) == value, f"Unexpected smoke protocol {key}")
    records = report["real_training_records"]
    require([r["iteration"] for r in records] == list(range(1, 21)), "Noncontiguous steps")
    expected_losses = {f"{stage}.loss_{kind}" for stage in
                       ["decode"] + [f"decode.d{i}" for i in range(9)]
                       for kind in ("cls", "mask", "dice")}
    errors = []
    for record in records:
        require(record["valid_pixel_count"] > 0, "Empty source crop")
        require(set(record["losses"]) == expected_losses, "Incomplete auxiliary losses")
        values = [record["loss"], record["gradient_norm"], *record["losses"].values()]
        require(all(math.isfinite(v) for v in values), "Nonfinite training value")
        require(record["loss"] > 0 and record["gradient_norm"] >= 0, "Invalid loss/norm")
        errors.append(abs(record["loss"] - sum(record["losses"].values())))
    require(max(errors) < 1e-3, "Loss reconstruction mismatch")
    targets = report["target_inference_records"]
    require(len(targets) == 5, "Incomplete target inference")
    require(all(r["semantic_score_shape"] == [1, 19, 512, 512] for r in targets),
            "Wrong target prediction shape")
    error = report["max_normalization_roundtrip_error"]
    require(math.isfinite(error) and 0 <= error <= 1e-5, "Normalization roundtrip failed")
    for key in ("peak_allocated_gib", "peak_reserved_gib"):
        require(math.isfinite(report[key]) and report[key] > 0, "Invalid memory measurement")
    require(report["peak_reserved_gib"] >= report["peak_allocated_gib"], "Invalid memory ordering")
    return dict(real_data_saved_report_audit_ok=True, training_git_sha=SOURCE_SHA,
                optimizer_steps=20, target_inference_count=5,
                maximum_objective_reconstruction_error=max(errors),
                max_normalization_roundtrip_error=error,
                peak_allocated_gib=report["peak_allocated_gib"],
                peak_reserved_gib=report["peak_reserved_gib"],
                accuracy_claim=False, formal_training_authorized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--stderr-log", type=Path, required=True)
    args = parser.parse_args()
    try:
        for path, expected in ((args.report, REPORT_SHA), (args.stderr_log, LOG_SHA)):
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Evidence hash mismatch: {path}")
        result = audit_report(json.loads(args.report.read_text(encoding="utf-8")))
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps(dict(real_data_saved_report_audit_ok=False,
                              error=f"{type(exc).__name__}: {exc}"), indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
