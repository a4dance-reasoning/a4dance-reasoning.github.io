#!/usr/bin/env python3
"""
Compare two affordance JSONL files keyed by (image, affordance) -> label.

Supports mode-specific evaluation aligned with affordance_gen:
  standard   — closed-set metrics on BASE_VOCAB only (headline apples-to-apples).
  discover   — same closed-set block, plus discovery / novel-name diagnostics;
               optional --generation-stats from affordance_gen --discover-stats-json.

Lines are parsed as JSON; only fields image, affordance, label are used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from vocab import BASE_VOCAB, BASE_VOCAB_SET

app = typer.Typer(add_completion=False, help="Compare two affordance label JSONL files.")

MISSING_PRED_LABEL = -1


def load_jsonl(path: Path) -> dict[tuple[str, str], int]:
    rows: dict[tuple[str, str], int] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise typer.BadParameter(f"{path}:{line_no}: invalid JSON: {e}") from e
            key = (r["image"], r["affordance"])
            rows[key] = int(r["label"])
    return rows


def prediction_image_basenames(pred: dict[tuple[str, str], int]) -> frozenset[str]:
    return frozenset(k[0] for k in pred)


def restrict_gt_to_pred_images(
    gt: dict[tuple[str, str], int],
    images: frozenset[str],
) -> dict[tuple[str, str], int]:
    """Keep only GT rows whose `image` basename appears in `images`."""
    return {k: v for k, v in gt.items() if k[0] in images}


def load_json_optional(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class ComparisonResult:
    """Metrics: A = prediction, B = ground truth (full files, all keys)."""

    prediction_path: str
    ground_truth_path: str
    n_prediction: int
    n_ground_truth: int
    keys_in_both: int
    keys_only_in_prediction: int
    keys_only_in_ground_truth: int
    agree_on_overlap: int
    label_mismatches_on_overlap: int
    overlap_accuracy: float
    gt_correct: int
    gt_missing_in_prediction: int
    gt_wrong_label: int
    accuracy_vs_ground_truth: float
    prediction_extra_not_in_gt: int


def compute_comparison(
    pred: dict[tuple[str, str], int],
    gt: dict[tuple[str, str], int],
    pred_path: Path,
    gt_path: Path,
) -> ComparisonResult:
    keys_p = set(pred.keys())
    keys_g = set(gt.keys())
    common = keys_p & keys_g
    only_p = keys_p - keys_g
    only_g = keys_g - keys_p

    mismatches = sum(1 for k in common if pred[k] != gt[k])
    agree = len(common) - mismatches

    gt_correct = sum(1 for k in keys_g if k in keys_p and pred[k] == gt[k])
    gt_missing = len(only_g)
    gt_wrong = mismatches

    n_g = len(keys_g)
    overlap_acc = agree / len(common) if common else 0.0
    acc_vs_gt = gt_correct / n_g if n_g else 0.0

    return ComparisonResult(
        prediction_path=str(pred_path),
        ground_truth_path=str(gt_path),
        n_prediction=len(pred),
        n_ground_truth=len(gt),
        keys_in_both=len(common),
        keys_only_in_prediction=len(only_p),
        keys_only_in_ground_truth=len(only_g),
        agree_on_overlap=agree,
        label_mismatches_on_overlap=mismatches,
        overlap_accuracy=round(overlap_acc, 6),
        gt_correct=gt_correct,
        gt_missing_in_prediction=gt_missing,
        gt_wrong_label=gt_wrong,
        accuracy_vs_ground_truth=round(acc_vs_gt, 6),
        prediction_extra_not_in_gt=len(only_p),
    )


def _label_display(label: int) -> str:
    if label == MISSING_PRED_LABEL:
        return "<no pred row>"
    return str(label)


def closed_set_pairs(
    pred: dict[tuple[str, str], int],
    gt: dict[tuple[str, str], int],
) -> tuple[list[int], list[int], list[tuple[str, str]]]:
    """
    One sample per (image, affordance) where affordance ∈ BASE_VOCAB and the
    key exists in ground truth. y_pred uses MISSING if prediction lacks the key.
    """
    keys = sorted(k for k in gt if k[1] in BASE_VOCAB_SET)
    y_true: list[int] = []
    y_pred: list[int] = []
    for k in keys:
        y_true.append(int(gt[k]))
        y_pred.append(int(pred[k]) if k in pred else MISSING_PRED_LABEL)
    return y_true, y_pred, keys


def confusion_matrix_from_pairs(
    y_true: list[int], y_pred: list[int]
) -> tuple[list[int], list[list[int]]]:
    labels = sorted(set(y_true) | set(y_pred))
    idx = {c: i for i, c in enumerate(labels)}
    n = len(labels)
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred, strict=True):
        cm[idx[t]][idx[p]] += 1
    return labels, cm


def per_class_precision_recall_f1(
    y_true: list[int],
    y_pred: list[int],
    classes: list[int],
) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == c and p != c)
        support = sum(1 for t in y_true if t == c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        out[str(c)] = {
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1": round(f1, 6),
            "support": support,
        }
    return out


def binary_counts_label_one(y_true: list[int], y_pred: list[int]) -> dict[str, int]:
    """TP/TN/FP/FN with positive class = 1; missing pred (-1) treated as predicted 0."""
    tp = tn = fp = fn = 0
    for t, p in zip(y_true, y_pred, strict=True):
        pe = 0 if p == MISSING_PRED_LABEL else int(p)
        if t == 1 and pe == 1:
            tp += 1
        elif t == 0 and pe == 0:
            tn += 1
        elif t == 0 and pe == 1:
            fp += 1
        else:
            fn += 1
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def _closed_set_metrics_from_vectors(
    y_true: list[int],
    y_pred: list[int],
    eval_keys: list[tuple[str, str]],
    *,
    eval_keys_note: str,
    base_prediction_keys_not_in_gt: list[tuple[str, str]],
) -> dict[str, Any]:
    missing_on_closed = sum(1 for p in y_pred if p == MISSING_PRED_LABEL)

    if not y_true:
        return {
            "n_eval_keys": 0,
            "base_vocab": list(BASE_VOCAB),
            "eval_keys_sample_note": eval_keys_note,
            "accuracy": 0.0,
            "missing_prediction_rows": 0,
            "base_prediction_keys_not_in_gt": base_prediction_keys_not_in_gt,
            "confusion_matrix": {"labels": [], "matrix": []},
            "per_class": {},
            "macro_f1": 0.0,
            "binary_label_1": {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
            "binary_precision_label_1": 0.0,
            "binary_recall_label_1": 0.0,
            "binary_f1_label_1": 0.0,
        }

    correct = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
    cm_labels, cm = confusion_matrix_from_pairs(y_true, y_pred)
    gt_classes = sorted(set(y_true))
    per_class = per_class_precision_recall_f1(y_true, y_pred, gt_classes)
    k = len(gt_classes)
    macro_f1 = (
        sum(per_class[str(c)]["f1"] for c in gt_classes) / k if k else 0.0
    )
    bc = binary_counts_label_one(y_true, y_pred)
    p1 = bc["TP"] / (bc["TP"] + bc["FP"]) if (bc["TP"] + bc["FP"]) > 0 else 0.0
    r1 = bc["TP"] / (bc["TP"] + bc["FN"]) if (bc["TP"] + bc["FN"]) > 0 else 0.0
    f1p = (2 * p1 * r1 / (p1 + r1)) if (p1 + r1) > 0 else 0.0

    return {
        "n_eval_keys": len(eval_keys),
        "base_vocab": list(BASE_VOCAB),
        "eval_keys_sample_note": eval_keys_note,
        "accuracy": round(correct / len(y_true), 6),
        "missing_prediction_rows": missing_on_closed,
        "base_prediction_keys_not_in_gt": base_prediction_keys_not_in_gt,
        "confusion_matrix": {
            "labels": cm_labels,
            "matrix": cm,
            "note": "rows=true (GT), cols=pred; "
            f"{MISSING_PRED_LABEL} = no row in prediction for that (image, base affordance)",
        },
        "per_class": per_class,
        "macro_f1": round(macro_f1, 6),
        "binary_label_1": bc,
        "binary_precision_label_1": round(p1, 6),
        "binary_recall_label_1": round(r1, 6),
        "binary_f1_label_1": round(f1p, 6),
    }


def load_generated_base_affordances_by_image(path: Path) -> dict[str, set[str]]:
    """image basename -> base-vocab names listed in affordance-sets JSONL for that image."""
    out: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            img = str(r["image"])
            names = {str(x) for x in (r.get("affordances") or [])}
            out[img] = {a for a in names if a in BASE_VOCAB_SET}
    return out


def build_generated_base_closed_report(
    pred: dict[tuple[str, str], int],
    gt: dict[tuple[str, str], int],
    affordance_sets: Path,
) -> dict[str, Any]:
    """
    Like headline closed-set, but only (image, affordance) cells where:
    affordance ∈ BASE_VOCAB, the name appears in affordance-sets for that image,
    and a GT row exists. No padded base-vocab zeros.
    """
    gen_by_img = load_generated_base_affordances_by_image(affordance_sets)
    keys: list[tuple[str, str]] = []
    for img in sorted(gen_by_img.keys()):
        for aff in sorted(gen_by_img[img]):
            k = (img, aff)
            if k in gt:
                keys.append(k)
    y_true = [int(gt[k]) for k in keys]
    y_pred = [int(pred[k]) if k in pred else MISSING_PRED_LABEL for k in keys]
    note = (
        "keys = (image, affordance) with affordance in BASE_VOCAB and listed in "
        "affordance-sets for that image; GT row must exist (no base-vocab padding)"
    )
    return _closed_set_metrics_from_vectors(
        y_true,
        y_pred,
        keys,
        eval_keys_note=note,
        base_prediction_keys_not_in_gt=[],
    )


def build_closed_set_report(
    pred: dict[tuple[str, str], int],
    gt: dict[tuple[str, str], int],
) -> dict[str, Any]:
    y_true, y_pred, eval_keys = closed_set_pairs(pred, gt)
    base_pred_only = sorted(
        k for k in pred if k[1] in BASE_VOCAB_SET and k not in gt
    )
    return _closed_set_metrics_from_vectors(
        y_true,
        y_pred,
        eval_keys,
        eval_keys_note="closed_set_keys = all GT rows with affordance in BASE_VOCAB",
        base_prediction_keys_not_in_gt=base_pred_only,
    )


def build_discover_report(
    pred: dict[tuple[str, str], int],
    gt: dict[tuple[str, str], int],
    generation_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    pred_novel_keys = [k for k in pred if k[1] not in BASE_VOCAB_SET]
    gt_novel_keys = [k for k in gt if k[1] not in BASE_VOCAB_SET]
    pred_novel_set = set(pred_novel_keys)
    gt_novel_set = set(gt_novel_keys)

    novel_in_both = sorted(pred_novel_set & gt_novel_set)
    novel_pred_only = sorted(pred_novel_set - gt_novel_set)
    novel_gt_only = sorted(gt_novel_set - pred_novel_set)

    distinct_novel_names_pred = sorted({k[1] for k in pred_novel_keys})
    distinct_novel_names_gt = sorted({k[1] for k in gt_novel_keys})

    out: dict[str, Any] = {
        "non_base_prediction_rows": len(pred_novel_keys),
        "non_base_ground_truth_rows": len(gt_novel_keys),
        "non_base_keys_in_both": len(novel_in_both),
        "non_base_keys_only_in_prediction": len(novel_pred_only),
        "non_base_keys_only_in_ground_truth": len(novel_gt_only),
        "distinct_non_base_affordances_in_prediction": distinct_novel_names_pred,
        "distinct_non_base_affordances_in_ground_truth": distinct_novel_names_gt,
        "human_review_note": (
            "No automatic precision for novel names vs 'should exist' unless GT is "
            "exhaustive or you add human labels. After review, record optional "
            "novel_name_precision and novel_label_accuracy externally."
        ),
        "generation_stats_path": None,
        "generation_metrics": None,
    }

    if generation_stats is not None:
        out["generation_metrics"] = {
            "schema": generation_stats.get("schema"),
            "totals": generation_stats.get("totals"),
            "merge_note": generation_stats.get("note"),
        }

    return out


def print_closed_set(
    title: str,
    rep: dict[str, Any],
    *,
    eval_keys_label: str | None = None,
) -> None:
    typer.echo()
    typer.echo(f"========== {title} ==========")
    typer.echo(f"Base vocabulary ({len(BASE_VOCAB)}): {', '.join(BASE_VOCAB)}")
    kl = eval_keys_label or (
        "Closed-set eval keys (GT rows with affordance in base)"
    )
    typer.echo(f"{kl}: {rep['n_eval_keys']}")
    if rep["n_eval_keys"] == 0:
        typer.echo("  (no eval keys for this block)")
        typer.echo("==========================================")
        return
    typer.echo(f"Closed-set accuracy: {rep['accuracy']:.4f}")
    typer.echo(
        f"Missing prediction rows (base affordance in GT but absent in pred): "
        f"{rep['missing_prediction_rows']}"
    )
    cm = rep["confusion_matrix"]
    labels: list[int] = cm["labels"]
    matrix: list[list[int]] = cm["matrix"]
    typer.echo()
    typer.echo("Confusion matrix (rows=GT, cols=pred)")
    col_headers = [_label_display(x) for x in labels]
    cell_w = max(
        max(len(str(v)) for row in matrix for v in row) if matrix else 1,
        max(len(h) for h in col_headers) if col_headers else 4,
        4,
    )
    left_w = max(len(_label_display(x)) for x in labels) + 2 if labels else 14
    typer.echo(
        "true \\ pred".ljust(left_w)
        + "".join(f"{h:>{cell_w + 2}}" for h in col_headers)
    )
    for i, row in enumerate(matrix):
        typer.echo(
            _label_display(labels[i]).ljust(left_w)
            + "".join(f"{v:>{cell_w + 2}}" for v in row)
        )
    typer.echo(f"  {cm.get('note', '')}")
    typer.echo()
    typer.echo("Per-class (one-vs-rest precision / recall / F1)")
    for c_str, m in sorted(rep["per_class"].items(), key=lambda x: int(x[0])):
        typer.echo(
            f"  class {c_str}:  P={m['precision']:.4f}  R={m['recall']:.4f}  "
            f"F1={m['f1']:.4f}  support={m['support']}"
        )
    typer.echo(f"  macro-F1 (GT classes): {rep['macro_f1']:.4f}")
    typer.echo()
    b = rep["binary_label_1"]
    typer.echo(
        "Binary (positive = label 1; missing pred row counts as predicted 0): "
        f"TP={b['TP']} TN={b['TN']} FP={b['FP']} FN={b['FN']}"
    )
    typer.echo(
        f"  precision (class 1): {rep['binary_precision_label_1']:.4f}  "
        f"recall: {rep['binary_recall_label_1']:.4f}  "
        f"F1: {rep['binary_f1_label_1']:.4f}"
    )
    extra = rep.get("base_prediction_keys_not_in_gt") or []
    if extra:
        typer.echo(
            f"  Base-affordance prediction keys not in GT ({len(extra)}); "
            "not counted in closed-set (GT defines eval universe)."
        )
    typer.echo("==========================================")


def print_discover_extra(rep: dict[str, Any]) -> None:
    typer.echo()
    typer.echo("========== Discovery-mode summary (non-base rows) ==========")
    typer.echo(
        f"Non-base prediction rows: {rep['non_base_prediction_rows']}  |  "
        f"non-base GT rows: {rep['non_base_ground_truth_rows']}"
    )
    typer.echo(
        f"Non-base keys in both: {rep['non_base_keys_in_both']}  |  "
        f"only in pred: {rep['non_base_keys_only_in_prediction']}  |  "
        f"only in GT: {rep['non_base_keys_only_in_ground_truth']}"
    )
    typer.echo()
    typer.echo(rep["human_review_note"])
    gm = rep.get("generation_metrics")
    if gm and gm.get("totals"):
        typer.echo()
        typer.echo("Generation-time merge stats (--discover-stats-json):")
        for k, v in gm["totals"].items():
            typer.echo(f"  {k}: {v}")
    typer.echo("==============================================================")


def print_full_summary(res: ComparisonResult) -> None:
    typer.echo()
    typer.echo("---------- Full-file summary (all affordance names) ----------")
    typer.echo(f"Rows in prediction:   {res.n_prediction}")
    typer.echo(f"Rows in ground truth: {res.n_ground_truth}")
    typer.echo(f"Keys in both:         {res.keys_in_both}")
    typer.echo(f"Keys only in pred:    {res.keys_only_in_prediction}")
    typer.echo(f"Keys only in GT:      {res.keys_only_in_ground_truth}")
    typer.echo(
        f"Accuracy vs full GT:  {res.accuracy_vs_ground_truth:.4f} "
        f"({res.gt_correct} / {res.n_ground_truth})"
    )
    typer.echo("--------------------------------------------------------------")


@app.command()
def main(
    prediction: Path = typer.Option(
        ...,
        "--prediction",
        "-p",
        metavar="JSONL",
        help="Generated output JSONL (e.g. affordance_gen --output).",
    ),
    ground_truth: Path = typer.Option(
        ...,
        "--ground-truth",
        "-g",
        metavar="JSONL",
        help="Reference / ground-truth JSONL to compare against.",
    ),
    eval_mode: str = typer.Option(
        "standard",
        "--eval-mode",
        help="standard | discover — closed-set always; discover adds non-base diagnostics.",
    ),
    generation_stats: Path | None = typer.Option(
        None,
        "--generation-stats",
        metavar="JSON",
        help="Discover mode: JSON from affordance_gen --discover-stats-json (merge totals).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print every mismatch and key-only-in-one-file row (full key space).",
    ),
    limit: int = typer.Option(
        50,
        "--limit",
        "-n",
        help="When not --verbose: max lines per section (0 = print all).",
    ),
    json_out: Path | None = typer.Option(
        None,
        "--json-out",
        help="Write metrics JSON: standard → closed_set only; discover → closed_set + discover.",
    ),
    eval_prediction_images_only: bool = typer.Option(
        False,
        "--eval-prediction-images-only/--eval-full-ground-truth",
        help=(
            "Restrict ground-truth rows to image basenames that appear in the prediction file "
            "(recommended after affordance_gen --max-images or other subsets)."
        ),
    ),
    eval_generated_base_from_sets: Path | None = typer.Option(
        None,
        "--eval-generated-base-from-sets",
        metavar="JSONL",
        help=(
            "Affordance-sets JSONL from generate: add a second closed-set block evaluated only on "
            "(image, base affordance) pairs listed there (intersect GT rows); excludes padded zeros."
        ),
    ),
) -> None:
    """Compare prediction vs ground truth; use --eval-mode for standard vs discover metrics."""
    if not prediction.is_file():
        raise typer.BadParameter(f"Not a file: {prediction}")
    if not ground_truth.is_file():
        raise typer.BadParameter(f"Not a file: {ground_truth}")
    if eval_generated_base_from_sets is not None and not eval_generated_base_from_sets.is_file():
        raise typer.BadParameter(
            f"Not a file: {eval_generated_base_from_sets}"
        )
    em = eval_mode.strip().lower()
    if em not in ("standard", "discover"):
        raise typer.BadParameter("--eval-mode must be 'standard' or 'discover'")

    pred = load_jsonl(prediction)
    gt_full = load_jsonl(ground_truth)
    gt = gt_full
    if eval_prediction_images_only:
        imgs = prediction_image_basenames(pred)
        gt = restrict_gt_to_pred_images(gt_full, imgs)
        typer.echo(
            f"Eval: ground truth restricted to {len(imgs)} image(s) in prediction "
            f"({len(gt)} GT rows; full file has {len(gt_full)} rows)."
        )
    stats_blob = load_json_optional(generation_stats)

    closed = build_closed_set_report(pred, gt)
    disc_payload: dict[str, Any] | None = None
    if em == "discover":
        disc_payload = build_discover_report(pred, gt, stats_blob)

    print_closed_set("Closed-set metrics (BASE_VOCAB — headline)", closed)

    generated_base_rep: dict[str, Any] | None = None
    if eval_generated_base_from_sets is not None:
        generated_base_rep = build_generated_base_closed_report(
            pred, gt, eval_generated_base_from_sets
        )
        print_closed_set(
            "Closed-set metrics (BASE_VOCAB — generated affordances only, no padding)",
            generated_base_rep,
            eval_keys_label=(
                "Eval keys (base name in affordance-sets for image, and GT row exists)"
            ),
        )

    if em == "discover" and disc_payload is not None:
        print_discover_extra(disc_payload)

    res_full = compute_comparison(pred, gt, prediction, ground_truth)
    print_full_summary(res_full)

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        # One JSON file: standard → closed-set metrics only; discover → same + discover block.
        payload: dict[str, Any] = {
            "eval_mode": em,
            "prediction_path": str(prediction),
            "ground_truth_path": str(ground_truth),
            "eval_prediction_images_only": eval_prediction_images_only,
            "closed_set": closed,
        }
        if eval_prediction_images_only:
            imgs = prediction_image_basenames(pred)
            payload["prediction_image_count"] = len(imgs)
            payload["ground_truth_rows_full_file"] = len(gt_full)
            payload["ground_truth_rows_used"] = len(gt)
        if em == "discover":
            assert disc_payload is not None
            if generation_stats is not None:
                disc_payload = {
                    **disc_payload,
                    "generation_stats_path": str(generation_stats),
                }
            payload["discover"] = disc_payload
        if generated_base_rep is not None:
            payload["closed_set_generated_base_from_sets"] = generated_base_rep
            payload["eval_generated_base_from_sets_path"] = str(
                eval_generated_base_from_sets
            )
        json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        typer.echo()
        typer.echo(f"Wrote metrics JSON to {json_out}")

    keys_a = set(pred.keys())
    keys_b = set(gt.keys())
    common = keys_a & keys_b
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    mismatches: list[tuple[str, str, int, int]] = []
    for k in sorted(common):
        if pred[k] != gt[k]:
            mismatches.append((k[0], k[1], pred[k], gt[k]))

    key_mismatch_total = len(only_a) + len(only_b)
    label_mismatch_count = len(mismatches)

    typer.echo()
    typer.echo("--- Full key space: mismatch counts ---")
    typer.echo(
        f"Key mismatches: {key_mismatch_total} "
        f"({len(only_a)} only in prediction, {len(only_b)} only in GT)."
    )
    typer.echo(
        f"Label mismatches (same key, different label): {label_mismatch_count}"
    )

    def print_limited(title: str, items: list, formatter) -> None:
        typer.echo()
        typer.echo(f"--- {title} ({len(items)} total) ---")
        cap = None if verbose else (None if limit == 0 else limit)
        shown = items if cap is None else items[:cap]
        for item in shown:
            typer.echo(formatter(item))
        if cap is not None and len(items) > cap:
            typer.echo(f"... ({len(items) - cap} more; use --verbose for all)")

    if only_a:
        print_limited(
            "Only in prediction (all affordances)",
            sorted(only_a),
            lambda k: f"  {k[0]}\t{k[1]}\tlabel={pred[k]}",
        )
    if only_b:
        print_limited(
            "Only in ground truth (all affordances)",
            sorted(only_b),
            lambda k: f"  {k[0]}\t{k[1]}\tlabel={gt[k]}",
        )
    if mismatches:
        print_limited(
            "Label mismatches: image, affordance, pred, gt",
            mismatches,
            lambda t: f"  {t[0]}\t{t[1]}\tpred={t[2]}\tgt={t[3]}",
        )

    typer.echo()
    typer.echo("========== Done ==========")


if __name__ == "__main__":
    app()
