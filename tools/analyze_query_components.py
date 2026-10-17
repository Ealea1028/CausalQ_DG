"""Phase20 source-only component diagnostics: accepted gate or full census."""

import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import zipfile

import torch

from causalq.analysis.query_components import (fixed_source_ids, intervention_seed, query_parameter_rows,
                                               responsibility_rows, effect_rows)
from causalq.datasets.strict_source import load_membership
from tools import check_anchor_query_effect as gate
from tools.audit_anchor_query_effect import audit_anchor, REPORT_SHA, SUMS_SHA, COMPONENTS_SHA
from tools.strict_source_resume import capture_rng, restore_rng

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = 20


def audit_remote_anchor(directory):
    expected = {"anchor_report.json": REPORT_SHA, "SHA256SUMS": SUMS_SHA, "components.jsonl": COMPONENTS_SHA}
    for name, digest in expected.items():
        if gate.file_sha(directory / name) != digest:
            raise ValueError(f"Original Phase19 artifact differs: {name}")
    return expected


def csv_write(path, rows):
    if not rows:
        raise ValueError("Diagnostic CSV has no rows")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_present(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return {"mean": sum(values) / len(values) if values else None, "defined_count": len(values)}


def summarize_effects(rows):
    return {component: {key: mean_present([r for r in rows if r["component"] == component], key)
        for key in ("original_mean_abs", "photometric_mean_abs", "cross_view_population_variance", "rms_normalized_pair_mse")}
        for component in ("full", "anchor", "residual")}


def analyze_pair(model, image, label, style, seed, *, renderer=None):
    """Two forwards only; independently replay appearance RNG and restore caller RNG."""
    if image.shape[0] != 1 or model.training:
        raise ValueError("Diagnostics require one image and model.eval()")
    devices = [image.device.index or 0] if image.is_cuda else []
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        torch.manual_seed(seed)
        photo = style.photometric_view(image)
        torch.manual_seed(seed)
        if not torch.equal(photo, style.photometric_view(image)):
            raise ValueError("Fixed-seed appearance replay differs")
        outputs, effects, checks = [], [], []
        for value in (image, photo):
            with gate.precision(str(image.device), image.is_cuda):
                output = model.forward_components(value, include_anchor=True)
            outputs.append(output)
            effects.append({"full": model.get_query_effect(output=output).float(),
                            "anchor": model.get_anchor_query_effect(output=output),
                            "residual": model.get_residual_contribution(output=output)})
            checks.append(gate.component_checks(model, output))
        responsibilities = responsibility_rows(outputs[0].query_score_maps, outputs[1].query_score_maps, label)
        statistics = effect_rows(effects[0], effects[1], label)
        if renderer is not None:
            # Rendering libraries cannot perturb subsequent intervention RNG.
            state = capture_rng()
            try:
                renderer(image, photo, label, outputs, effects)
            finally:
                restore_rng(state)
        return responsibilities, statistics, {"intervention_seed": seed,
            "original_input_sha256": gate.tree_digest((image, label)),
            "photometric_input_sha256": gate.tree_digest((photo, label)),
            "appearance_replay_exact": True, "decomposition_max_error": max(c["decomposition_max_error"] for c in checks),
            "counterfactual_routing_exact": all(c["counterfactual_routing_exact"] for c in checks),
            "effect_summary": summarize_effects(statistics)}


def audit_remote_diagnostics(directory):
    from tools.audit_query_component_diagnostics import REPORT_SHA, SUMS_SHA, CSV_HASHES
    expected = {'component_effect_report.json': REPORT_SHA, 'SHA256SUMS': SUMS_SHA, **CSV_HASHES}
    for name, digest in expected.items():
        if gate.file_sha(directory / name) != digest:
            raise ValueError(f'Original diagnostic artifact differs: {name}')
    return expected


def check_gate20_replay(sample, reference):
    for key in ('source_id', 'intervention_seed', 'original_input_sha256', 'photometric_input_sha256',
                'appearance_replay_exact', 'counterfactual_routing_exact'):
        if sample[key] != reference[key]:
            raise ValueError(f'Full-census gate20 replay differs: {key}')
    for component in ('full', 'anchor', 'residual'):
        for key, expected in reference['effect_summary'][component].items():
            actual = sample['effect_summary'][component][key]
            if actual['defined_count'] != expected['defined_count'] or not math.isclose(actual['mean'], expected['mean'], rel_tol=1e-6, abs_tol=1e-12):
                raise ValueError('Full-census gate20 effect summary replay differs')


def census_summary(samples):
    anchor = [s['effect_summary']['anchor']['cross_view_population_variance']['mean'] for s in samples]
    residual = [s['effect_summary']['residual']['cross_view_population_variance']['mean'] for s in samples]
    pairs = [(a, r) for a, r in zip(anchor, residual) if a is not None and r is not None]
    return {'image_count': len(samples), 'defined_image_count': len(pairs),
            'images_raw_anchor_variance_lower': sum(a < r for a, r in pairs),
            'images_raw_anchor_variance_equal': sum(a == r for a, r in pairs),
            'raw_anchor_minus_residual_variance_mean': sum(a - r for a, r in pairs) / len(pairs) if pairs else None,
            'inference_limit': 'Descriptive source census for one 20-step engineering checkpoint; no independence, specialization or DG utility claim'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--anchor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument('--full-source', action='store_true', help='Exactly all2497 source holdout IDs; no budget override')
    parser.add_argument('--diagnostic-dir', type=Path, help='Original accepted9439771 gate directory, required for full-source')
    args = parser.parse_args(argv)
    if args.full_source and args.diagnostic_dir is None:
        parser.error('--full-source requires --diagnostic-dir')
    centered_dir = args.checkpoint.resolve().parent
    output = args.output_dir.resolve()
    protected = [ROOT, args.data_root, args.pretrained_root, centered_dir, args.anchor_dir]
    if args.diagnostic_dir is not None:
        protected.append(args.diagnostic_dir)
    if any(output.is_relative_to(path.resolve()) for path in protected):
        parser.error("Diagnostic evidence must be outside code/data/weights/original runs")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "phase": 20, "scope": "source_component_diagnostics20x2",
        "created_utc": datetime.now(timezone.utc).isoformat(), "optimizer_updates_this_run": 0,
        "checkpoint_training_updates": 20, "formal_training_authorized": False, "target_data_accessed": False,
        "new_loss_enabled": False, "source_miou": None, "target_miou": None,
        "population_scope": "20 filename-hash-selected source holdout images; bounded engineering diagnostics only",
        **gate.source_manifest_metadata()}
    if args.full_source:
        report.update(scope='source_component_census2497x2', population_scope='All2497 frozen source holdout IDs; same20-step engineering checkpoint')
    try:
        report.update(git_sha=gate._git("rev-parse", "HEAD"), git_status=gate._git("status", "--porcelain"))
        if report["git_status"]:
            raise ValueError("Diagnostics require clean exact checkout")
        if "tools.train" in sys.modules or "causalq.datasets.cityscapes" in sys.modules:
            raise ValueError("Legacy/target module imported")
        if args.full_source:
            from tools.audit_query_component_diagnostics import audit_components
            report['diagnostic_audit'] = audit_components(ROOT / 'experiments/STRICT_DG_PHASE20')
            report['remote_diagnostic_artifacts'] = audit_remote_diagnostics(args.diagnostic_dir)
            accepted = json.loads((args.diagnostic_dir / 'component_effect_report.json').read_bytes())
            references = {s['source_id']: s for s in accepted['samples']}
            os.environ.setdefault('MPLCONFIGDIR', str(output / 'plot_cache'))
            import matplotlib
            report['plot_environment'] = {'matplotlib': matplotlib.__version__, 'backend': 'FigureCanvasAgg'}
        report["anchor_audit"] = audit_anchor(ROOT / "experiments/STRICT_DG_PHASE19")
        report["remote_anchor_artifacts"] = audit_remote_anchor(args.anchor_dir)
        report["centered_audit"] = gate.audit_centered(ROOT / "experiments/STRICT_DG_PHASE18")
        report["original_artifacts"] = gate.audit_original(centered_dir)
        if args.checkpoint.name != "state.pth" or gate.file_sha(args.checkpoint) != gate.CHECKPOINT_SHA:
            raise ValueError("Use original immutable centered checkpoint")
        expected_manifest = ROOT / "manifests/gta5_source_val.txt"
        if args.source_manifest.resolve() != expected_manifest.resolve() or gate.file_sha(args.source_manifest) != report["manifest_sha256"]["gta5_source_val.txt"]:
            raise ValueError("Use the frozen source holdout manifest")
        _, ids = load_membership()
        selected = list(ids) if args.full_source else fixed_source_ids(ids, SAMPLES)
        if args.full_source and len(selected) != 2497:
            raise ValueError('Full census requires exactly2497 source holdout IDs')
        sample_count = len(selected)
        report["selection"] = {"method": "sha256(causalq-components-v1:ID) ascending; 20 lowest", "ids": selected,
            "style_seed_rule": "first 8 hex digits of sha256(causalq-components-photo-v1:ID)",
            "geometry": "fixed SourceEvalTransform; no random crop", "views": ["original", "photometric"]}
        if args.full_source:
            report['selection'].update(method='all frozen source holdout IDs in manifest order',
                visualization_ids=fixed_source_ids(ids, SAMPLES), visualization_classes=[0, 2, 8, 13, 11])
        original = json.loads((centered_dir / "centered_report.json").read_bytes())
        config = original["config"]
        gate.validate_centered_config(config)
        report["style_config"] = config["style"]["photometric"]
        report["inputs"] = {"checkpoint": str(args.checkpoint.resolve()), "source_manifest": str(args.source_manifest.resolve()),
                            "anchor_directory": str(args.anchor_dir.resolve())}
        if not torch.cuda.is_available():
            raise RuntimeError("DINOv3/source diagnostics require AutoDL CUDA")
        if str(torch.__version__) != "2.7.0+cu126" or version("transformers") != "4.56.1" or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("Use pinned CUDA runtime and CUBLAS_WORKSPACE_CONFIG=:4096:8")
        report["environment"] = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "torch": str(torch.__version__), "cuda": torch.version.cuda, "transformers": version("transformers"),
            "gpu": torch.cuda.get_device_name(0)}
        gate.seed_everything(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
        report["runtime"] = {"initialization_seed": 0, "autocast": "bfloat16", "workers": 0,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_benchmark": torch.backends.cudnn.benchmark}
        if args.full_source:
            report['plot_environment'].update(mkl_threading_layer=os.environ.get('MKL_THREADING_LAYER'),
                mkl_num_threads=os.environ.get('MKL_NUM_THREADS'), numpy=version('numpy'), pillow=version('Pillow'))
        weights = args.pretrained_root / config["model"]["weights_dir"]
        if gate.file_sha(weights / "model.safetensors") != config["model"]["weights_sha256"]:
            raise ValueError("Backbone weights SHA mismatch")
        backbone = gate.DINOv3Backbone.from_pretrained(config["model"]["backbone"], weights=weights, freeze=True,
            intermediate_indices=config["model"]["intermediate_indices"], dtype=torch.bfloat16, local_files_only=True).to("cuda:0")
        if gate.tree_digest(backbone.state_dict()) != gate.BACKBONE_STATE_SHA:
            raise ValueError("Backbone state differs")
        model = gate.QuerySegmentor(backbone, decoder_channels=256, num_classes=19, dropout=.1, queries_per_class=2,
            interaction="static", cross_attention_layers=0, num_heads=8, temperature=.07, alpha_init=0., centered_residuals=True).to("cuda:0").eval()
        gate.restore_anchor_source(model, args.checkpoint, original)
        _, holdout = gate.build_source_datasets(args.data_root / "gta5", gate.TrainTransform((512, 512)), gate.SourceEvalTransform())
        style = gate.StyleInterventionBank(**{k: tuple(v) if isinstance(v, list) else v for k, v in report["style_config"].items()}).to("cuda:0")
        before = gate.tree_digest((model.state_dict(), style.state_dict(), capture_rng()))
        parameters = query_parameter_rows(model.get_anchors(), model.get_centered_residuals())
        responsibilities, effects, samples, calls = [], [], [], [0]
        figures = []
        def count_call(*unused):
            calls[0] += 1
        handle = backbone.register_forward_hook(count_call)
        try:
            for source_id in selected:
                sample = holdout[ids.index(source_id)]
                if sample["id"] != source_id:
                    raise ValueError("Source index/order differs")
                image, label = (sample[k].unsqueeze(0).to("cuda:0") for k in ("image", "label"))
                prior = calls[0]
                torch.cuda.synchronize()
                started = time.perf_counter()
                renderer = None
                if args.full_source and source_id in report['selection']['visualization_ids']:
                    from causalq.analysis.component_figures import render_component_figures
                    def renderer(image, photo, label, outputs, maps):
                        figures.append(render_component_figures(output / 'figures' / source_id,
                            source_id, image, photo, label, outputs, maps))
                rr, er, summary = analyze_pair(model, image, label, style, intervention_seed(source_id), renderer=renderer)
                torch.cuda.synchronize()
                summary["analysis_seconds_including_metrics_and_style_replay"] = time.perf_counter() - started
                if calls[0] - prior != 2:
                    raise ValueError("Diagnostics require exactly one backbone pass per view")
                responsibilities.extend({"source_id": source_id, **row} for row in rr)
                effects.extend({"source_id": source_id, **row} for row in er)
                samples.append({"source_id": source_id, **summary})
                if args.full_source and source_id in references:
                    check_gate20_replay(samples[-1], references[source_id])
                print(f"diagnostic_samples={len(samples)}/{sample_count} source_id={source_id}", flush=True)
        finally:
            handle.remove()
        if before != gate.tree_digest((model.state_dict(), style.state_dict(), capture_rng())):
            raise ValueError("Read-only diagnostics changed model/style/RNG")
        if gate.audit_original(centered_dir) != report["original_artifacts"] or audit_remote_anchor(args.anchor_dir) != report["remote_anchor_artifacts"]:
            raise ValueError("Original evidence changed")
        if calls[0] != sample_count * 2 or len(samples) != sample_count or len(responsibilities) != sample_count * 19 * 2 * 2 or len(effects) != sample_count * 19 * 3:
            raise ValueError("Diagnostic coverage differs")
        for name, rows in (("query_similarity.csv", parameters), ("responsibility.csv", responsibilities), ("effect_statistics.csv", effects)):
            csv_write(output / name, rows)
        report.update(ok=True, sample_count=sample_count, view_count=sample_count * 2, backbone_forward_calls=calls[0],
            samples=samples, parameter_rows=parameters, responsibility_row_count=len(responsibilities),
            effect_row_count=len(effects), effect_summary=summarize_effects(effects),
            responsibility_summary={view: {key: mean_present([r for r in responsibilities if r["view"] == view], key)
                for key in ("mean_pixel_entropy_nats", "usage_entropy_nats", "cross_view_population_variance", "slot_logsumexp_deletion_drop")}
                for view in ("original", "photometric")},
            model_rng_unchanged=True, original_files_unchanged=True, checkpoint_sha256=gate.CHECKPOINT_SHA,
            head_sha256=gate.tree_digest(gate.trainable_state_dict(model)), alpha=float(model.query_head.alpha.detach()),
            total_parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3, peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3)
        if args.full_source:
            if audit_remote_diagnostics(args.diagnostic_dir) != report['remote_diagnostic_artifacts']:
                raise ValueError('Original diagnostic evidence changed')
            if {f['source_id'] for f in figures} != set(references) or len(figures) != 20:
                raise ValueError('Systematic20-source visualization coverage differs')
            from tools.audit_query_component_diagnostics import CSV_HASHES
            if gate.file_sha(output / 'query_similarity.csv') != CSV_HASHES['query_similarity.csv']:
                raise ValueError('Original parameter CSV replay differs')
            (output / 'sample_index.jsonl').write_text(''.join(json.dumps(s, allow_nan=False) + '\n' for s in samples), encoding='utf-8')
            report.update(census_comparison=census_summary(samples), gate20_replay_count=20,
                gate20_replay_accepted=True, parameter_replay_exact=True, figures=figures,
                figure_count=len(list((output / 'figures').glob('*/*.png'))),
                max_decomposition_error=max(s['decomposition_max_error'] for s in samples),
                appearance_replay_exact=all(s['appearance_replay_exact'] for s in samples),
                counterfactual_routing_exact=all(s['counterfactual_routing_exact'] for s in samples),
                per_class_effect_summary={str(c): summarize_effects([r for r in effects if r['class_id'] == c]) for c in range(19)},
                per_class_responsibility_summary={str(c): {view: {str(slot): {key: mean_present([r for r in responsibilities if r['class_id'] == c and r['view'] == view and r['query_r'] == slot], key)
                    for key in ('mean_responsibility', 'mean_pixel_entropy_nats', 'usage_entropy_nats', 'above_uniform_activation_rate',
                                'top1_share_ties_split', 'cross_view_population_variance', 'slot_logsumexp_deletion_drop')}
                    for slot in range(2)}
                    for view in ('original', 'photometric')} for c in range(19)})
            report['samples'] = [s for s in samples if s['source_id'] in references]
            report['sample_index_sha256'] = gate.file_sha(output / 'sample_index.jsonl')
            report['figure_sha256'] = {p.relative_to(output).as_posix(): gate.file_sha(p) for p in sorted((output / 'figures').glob('*/*.png'))}
            if report['figure_count'] != 60:
                raise ValueError('Expected60 scientific PNG panels')
    except Exception as exc:
        report.update(ok=False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    report["output_sha256"] = {p.name: gate.file_sha(p) for p in sorted(output.glob("*.csv"))}
    if args.full_source and report.get('ok'):
        # Raw source previews and full CSV/index stay outside Git in this archive.
        try:
            with zipfile.ZipFile(output / 'analysis_export.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
                for name in (*report['output_sha256'], 'sample_index.jsonl', *report['figure_sha256']):
                    archive.write(output / name, name)
            report['analysis_export_sha256'] = gate.file_sha(output / 'analysis_export.zip')
        except Exception as exc:
            report.update(ok=False, error=f'Archive export failed: {type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    (output / "component_effect_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    names = ["component_effect_report.json", *report["output_sha256"]]
    if args.full_source and report.get('ok'):
        names.extend(['sample_index.jsonl', 'analysis_export.zip', *report['figure_sha256']])
    (output / "SHA256SUMS").write_text("".join(f"{gate.file_sha(output / name)}  {name}\n" for name in names), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
