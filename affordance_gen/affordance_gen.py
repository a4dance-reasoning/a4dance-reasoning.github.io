#!/usr/bin/env python3
"""
V2 pipeline: affordance *generation* (task-conditioned) and optional *labeling*
(image + affordance only, no task).

Subcommands:
  generate  — write an *affordance sets* JSONL (one line per image): predicted name lists.
  label     — read that file; call VLM per (image, affordance) without task; write label JSONL.

This module is self-contained (no import of affordance_gen.py). Shared helpers that were
previously imported from V1 live in the "V2 backend" section below.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

import cv2
import numpy as np
import typer
from tqdm import tqdm

from vocab import BASE_VOCAB, BASE_VOCAB_SET

# Generation / labeling prompts: affordances are for a robot, not a human operator.
_VLM_ROBOT_VIEWPOINT = (
    "Perspective — Judge affordances for a mobile manipulation robot (e.g. a Boston Dynamics "
    "Spot-class quadruped without an arm and gripper, or a robot arm with a gripper), not for a "
    "human. Each property describes whether the object supports that interaction for the robot "
    "given typical onboard sensing and end-effectors. The required base names are already "
    "defined in that spirit (e.g. graspable means the robot’s gripper could meaningfully grasp it)."
)

_VLM_NAMING_CONSTRAINTS = (
    "Naming template — every key (required or new) must match the same style as the eight base "
    f"robot affordances: {', '.join(BASE_VOCAB)}. Short snake_case property words in that same "
    "family (mostly -able / similar participles). Prefer ONE token; at most TWO only if both "
    "stay adjective-like (e.g. low_profile). No noun-phrase stacks.\n"
    "Do NOT output keys containing or starting with: has_, is_, with_, or ending in _present; "
    "do not use consistent_, accessible_, clearance, underseat_, pin_pull, or similar "
    "descriptor piles — if unsure, omit the extra key.\n"
    "If a base affordance is already 1, do not add a longer compound that only repeats it "
    "(e.g. not handle_graspable when graspable is 1)."
)

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover
    torch = None
    AutoModel = None
    AutoTokenizer = None

app = typer.Typer(add_completion=False, help="V2: split affordance generation vs labeling.")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# V2 backend (inlined from affordance_gen.py so V2 does not depend on V1)
# ---------------------------------------------------------------------------

_DEFAULT_IMAGE_GLOB = "*.png|*.jpg|*.jpeg|*.webp"

LEXICAL_SYNONYMS: dict[str, str] = {
    "pickable": "graspable",
    "holdable": "graspable",
    "carryable": "liftable",
    "portable": "liftable",
    "shiftable": "movable",
    "repositionable": "movable",
    "dragable": "movable",
    "draggable": "movable",
    "push-able": "traversable",
    "pushable": "traversable",
    "wheelable": "rollable",
    "can_contain": "containable",
    "container-like": "containable",
    "can_open": "openable",
}

AFFORDANCE_SYNONYMS: dict[str, list[str]] = {
    "graspable": ["grip", "pick up by hand", "grasp", "holdable"],
    "movable": ["can move", "displaceable", "relocatable"],
    "stackable": ["stack", "pile", "nestable"],
    "supportable": ["support weight", "load bearing", "can hold items on top"],
    "liftable": ["lift", "pick up", "portable", "carry"],
    "rollable": ["roll", "wheels", "casters"],
    "openable": ["open", "lid", "door", "drawer"],
    "containable": ["holds contents", "hollow", "vessel", "container"],
    "traversable": [
        "walk through",
        "navigate",
        "passable",
        "pass through",
        "push",
        "shove",
        "slide along floor",
    ],
}

Uncertainty = Literal["low", "medium", "high"]


class LexicalCanonicalizer:
    """Map alternate affordance strings to controlled vocab (exact + synonym table only)."""

    def __init__(self, vocab: Sequence[str], synonyms: Mapping[str, str]) -> None:
        self._vocab = set(vocab)
        self._syn = {k.lower(): v.lower() for k, v in synonyms.items()}

    def canonicalize(self, name: str) -> str:
        raw = name.strip().lower()
        if raw in self._vocab:
            return raw
        return self._syn.get(raw, raw)


def expand_image_paths(image_dir: Path, glob_pattern: str) -> list[Path]:
    """Resolve one or more globs (split on |); dedupe by filename; sort by path."""
    patterns = [x.strip() for x in glob_pattern.split("|") if x.strip()]
    seen: set[str] = set()
    out: list[Path] = []
    for pat in patterns:
        for p in sorted(image_dir.glob(pat)):
            if not p.is_file():
                continue
            if p.name in seen:
                continue
            seen.add(p.name)
            out.append(p)
    out.sort(key=lambda p: p.name)
    return out


def select_image_paths(
    paths: list[Path],
    *,
    max_images: int | None,
    shuffle_seed: int | None,
    one_per_category: bool,
) -> list[Path]:
    out = list(paths)
    if one_per_category:
        by_cat: dict[str, Path] = {}
        for p in sorted(out, key=lambda x: x.name):
            stem = p.stem.lower()
            cat = stem.split("_")[0] if "_" in stem else stem
            if cat not in by_cat:
                by_cat[cat] = p
        out = sorted(by_cat.values(), key=lambda x: x.name)
    if shuffle_seed is not None:
        rng = random.Random(int(shuffle_seed))
        out = list(out)
        rng.shuffle(out)
    if max_images is not None:
        if max_images < 1:
            raise ValueError("max_images must be >= 1 when set")
        out = out[:max_images]
    return out


def infer_object_name_from_filename(filename: str) -> str:
    stem = Path(filename).stem.lower()
    parts = stem.split("_")
    if len(parts) >= 3:
        candidate = parts[2]
        if "-" in candidate:
            candidate = candidate.split("-")[0]
        return candidate
    return stem


_SMALL_GRASPABLE = {
    "mouse",
    "bowl",
    "cup",
    "mug",
    "bottle",
    "book",
    "phone",
    "remote",
    "apple",
}
_CONTAINERS = {"bowl", "cup", "mug", "box", "basket", "bin", "plate"}
_OPENABLE = {"door", "drawer", "cabinet", "box", "bottle", "jar"}
_ROLLABLE = {"ball", "wheel", "cart", "toycar", "can", "bottle"}
_SUPPORTABLE = {"table", "desk", "shelf", "cabinet", "counter", "cart"}
_STACKABLE = {"bowl", "plate", "box", "book", "cup"}
_TRAVERSABLE_FILENAME_HINTS = {"stairs"}
_NON_LIFTABLE = {"cabinet", "fridge", "sofa", "bed", "desk"}


def object_name_priors(object_name: str) -> dict[str, float]:
    on = object_name.lower()
    priors: dict[str, float] = {}

    if on in _SMALL_GRASPABLE:
        priors["graspable"] = 0.85
    if on in _CONTAINERS:
        priors["containable"] = 0.82
    elif on in {"mouse", "book"}:
        priors["containable"] = 0.15
    if on in _OPENABLE:
        priors["openable"] = 0.78
    elif on in {"mouse", "bowl"}:
        priors["openable"] = 0.12
    if on in _ROLLABLE:
        priors["rollable"] = 0.75
    elif on in {"mouse", "bowl", "book"}:
        priors["rollable"] = 0.15
    if on in _SUPPORTABLE:
        priors["supportable"] = 0.80
    elif on in {"mouse", "bowl", "cup"}:
        priors["supportable"] = 0.18
    if on in _STACKABLE:
        priors["stackable"] = 0.74
    elif on == "mouse":
        priors["stackable"] = 0.15
    if on in _TRAVERSABLE_FILENAME_HINTS:
        priors["traversable"] = 0.88
    if on in _NON_LIFTABLE:
        priors["liftable"] = 0.12
    elif on in _SMALL_GRASPABLE or on in {"bowl", "cup", "box"}:
        priors["liftable"] = 0.82

    if on in _NON_LIFTABLE:
        priors["movable"] = 0.35
    elif on in _SMALL_GRASPABLE or on in {"bowl", "cup", "box", "chair", "cart"}:
        priors["movable"] = 0.80

    return priors


def fuse_geometry_and_name_prior(
    geo_score: float,
    priors: Mapping[str, float],
    aff: str,
    weight: float,
) -> float:
    if weight <= 0.0 or aff not in priors:
        return geo_score
    p = priors[aff]
    return (1.0 - weight) * geo_score + weight * p


def load_memory_rows_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_image_tasks_jsonl(path: Path) -> dict[str, str]:
    tasks: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise typer.BadParameter(f"{path}:{line_no}: invalid JSON: {e}") from e
            img = str(r["image"])
            tasks[img] = str(r.get("task") or "").strip()
    return tasks


def affordance_names_from_memory_rows(rows: Sequence[dict]) -> set[str]:
    return {str(r["affordance"]) for r in rows if r.get("affordance") is not None}


@dataclass
class GeometrySignals:
    aspect_ratio: float
    fill_ratio: float
    solidity: float
    edge_density: float
    circle_score: float
    rectangularity: float
    h_mean: float
    w_mean: float


def _largest_contour_mask(gray: np.ndarray) -> tuple[np.ndarray | None, float]:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = 255 - binary
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 1:
        return None, 0.0
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(area / hull_area) if hull_area > 0 else 0.0
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [c], -1, 255, -1)
    return mask, solidity


def compute_geometry_signals(bgr: np.ndarray) -> GeometrySignals:
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask, solidity = _largest_contour_mask(gray)
    if mask is None:
        return GeometrySignals(
            aspect_ratio=w / max(h, 1),
            fill_ratio=0.0,
            solidity=0.0,
            edge_density=0.0,
            circle_score=0.0,
            rectangularity=0.0,
            h_mean=float(h),
            w_mean=float(w),
        )
    fill_ratio = float(np.sum(mask > 0) / (h * w))
    ys, xs = np.where(mask > 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    aspect_ratio = bw / bh

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(8, min(bw, bh) // 8),
        param1=50,
        param2=28,
        minRadius=3,
        maxRadius=max(6, min(bw, bh) // 4),
    )
    circle_score = 0.0
    if circles is not None:
        circle_score = min(1.0, len(circles[0]) / 6.0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangularity = 0.3
    if contours:
        c = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        if peri > 1e-6:
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            rectangularity = 1.0 if len(approx) == 4 else 0.3

    return GeometrySignals(
        aspect_ratio=aspect_ratio,
        fill_ratio=fill_ratio,
        solidity=solidity,
        edge_density=edge_density,
        circle_score=circle_score,
        rectangularity=rectangularity,
        h_mean=float(bh),
        w_mean=float(bw),
    )


def cheap_affordance_scores(
    signals: GeometrySignals,
) -> dict[str, tuple[float, Uncertainty]]:
    ar = signals.aspect_ratio
    fr = signals.fill_ratio
    sol = signals.solidity
    cs = signals.circle_score
    ed = signals.edge_density
    rect = signals.rectangularity

    out: dict[str, tuple[float, Uncertainty]] = {}

    g = 1.0 - min(1.0, abs(fr - 0.35) * 3.0)
    out["graspable"] = (float(np.clip(g, 0.0, 1.0)), "medium" if 0.15 < fr < 0.55 else "low")

    huge_flat = ar > 2.8 and fr > 0.72
    out["movable"] = (0.2 if huge_flat else 0.85, "high" if huge_flat else "low")

    out["stackable"] = (0.5, "high")

    sup = 0.7 if fr > 0.5 and ed < 0.12 else 0.25
    out["supportable"] = (float(sup), "medium")

    fixed_like = ar > 2.5 and fr > 0.65
    lift = 0.15 if fixed_like else 0.75
    out["liftable"] = (float(lift), "high" if fixed_like else "low")

    roll = min(1.0, cs * 1.2 + (0.25 if sol < 0.92 else 0.0))
    out["rollable"] = (float(roll), "medium" if cs < 0.35 else "low")

    open_s = min(1.0, rect * 0.5 + ed * 1.5)
    out["openable"] = (float(np.clip(open_s, 0.0, 1.0)), "high")

    contain = 0.65 if 0.2 < fr < 0.6 and sol < 0.95 else 0.35
    out["containable"] = (float(contain), "medium")

    # trav = 0.55 if ar > 1.6 and ed > 0.08 else 0.4
    # out["traversable"] = (float(np.clip(trav, 0.0, 1.0)), "high")

    if set(out) != set(BASE_VOCAB):
        raise ValueError(
            "cheap_affordance_scores keys must match vocab.BASE_VOCAB "
            f"(missing={set(BASE_VOCAB) - set(out)}, extra={set(out) - set(BASE_VOCAB)})"
        )
    return out


class EmbeddingMapper:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        if torch is None or AutoModel is None:
            raise RuntimeError("transformers/torch required for EmbeddingMapper")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        toks = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128,
        ).to(self.device)
        out = self.model(**toks).last_hidden_state
        mask = toks["attention_mask"].unsqueeze(-1)
        summed = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        vecs = torch.nn.functional.normalize(summed, p=2, dim=1)
        return vecs.cpu().numpy().astype(np.float32)

    def map_phrase_to_vocab(self, phrase: str) -> str:
        canonical = list(AFFORDANCE_SYNONYMS.keys())
        phrases = [phrase] + [
            f"{k}: " + ", ".join(AFFORDANCE_SYNONYMS[k]) for k in canonical
        ]
        emb = self.encode(phrases)
        sims = emb[0] @ emb[1:].T
        best = int(np.argmax(sims))
        return canonical[best]

    def nearest_in_set(self, phrase: str, candidates: Sequence[str]) -> tuple[str, float]:
        if not candidates:
            return phrase, 0.0
        texts = [phrase] + list(candidates)
        emb = self.encode(texts)
        sims = emb[0] @ emb[1:].T
        j = int(np.argmax(sims))
        return candidates[j], float(sims[j])


def normalize_vlm_affordance_key(raw: str) -> str:
    s = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if not s:
        return "unknown_affordance"
    return s


def merge_vlm_affordance_name(
    raw: str,
    known: set[str],
    mapper: EmbeddingMapper | None,
    merge_threshold: float,
) -> str:
    s = normalize_vlm_affordance_key(raw)
    if s in known:
        return s
    vocab_for_lex = sorted(set(BASE_VOCAB) | known)
    lex = LexicalCanonicalizer(vocab_for_lex, LEXICAL_SYNONYMS)
    phrase = s.replace("_", " ")
    c = lex.canonicalize(phrase)
    if c in known:
        return c
    if mapper is not None and known:
        best, sim = mapper.nearest_in_set(phrase, sorted(known))
        if sim >= merge_threshold:
            logger.debug("merge '%s' -> '%s' (sim=%.3f)", raw, best, sim)
            return best
    return s


def _novel_is_redundant_with_bases(novel: str, base_present: set[str]) -> bool:
    """Whether a non-base name duplicates an active base affordance (synonym / compound)."""
    if not base_present:
        return False
    s = normalize_vlm_affordance_key(novel)
    if s in base_present:
        return True
    if LEXICAL_SYNONYMS.get(s) in base_present:
        return True
    parts = [p for p in s.split("_") if p]
    for p in parts:
        if p in base_present:
            return True
        canon = LEXICAL_SYNONYMS.get(p)
        if canon is not None and canon in base_present:
            return True
    for b in base_present:
        if s.startswith(f"{b}_") or s.endswith(f"_{b}") or f"_{b}_" in s:
            return True
    lex = LexicalCanonicalizer(sorted(base_present), LEXICAL_SYNONYMS)
    if lex.canonicalize(s.replace("_", " ")) in base_present:
        return True
    return False


def _novel_key_style_rejected(novel: str) -> bool:
    """Strip novel keys that violate naming rules when the model ignores the prompt."""
    s = normalize_vlm_affordance_key(novel)
    if s.startswith(("has_", "is_", "with_")):
        return True
    if s.endswith("_present") or "_present_" in f"_{s}_":
        return True
    for bad in ("clearance", "accessible", "consistent", "underseat", "pin_pull"):
        if bad in s:
            return True
    return False


def _filter_redundant_novel_affordances(
    affordances: Sequence[str],
) -> tuple[list[str], list[str]]:
    """
    Remove novel names that only parrot an active base affordance. Returns
    (affordances_kept, novel_affordances).
    """
    uniq = sorted(set(affordances))
    base_present = {a for a in uniq if a in BASE_VOCAB_SET}
    kept_novel = [
        a
        for a in uniq
        if a not in BASE_VOCAB_SET
        and not _novel_is_redundant_with_bases(a, base_present)
        and not _novel_key_style_rejected(a)
    ]
    out = sorted(base_present | set(kept_novel))
    return out, kept_novel


def _image_bytes_and_mime(image_path: Path) -> tuple[str, str]:
    import base64

    suf = image_path.suffix.lower()
    mime = "image/png"
    if suf in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif suf == ".webp":
        mime = "image/webp"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return mime, b64


def vlm_openai_predict(
    image_path: Path,
    task: str,
    affordances: Sequence[str],
    model: str,
    *,
    latency_sink: list[float] | None = None,
) -> dict[str, int]:
    from openai import OpenAI

    client = OpenAI()
    mime, b64 = _image_bytes_and_mime(image_path)

    names = ", ".join(affordances)
    ctx = task.strip() or (
        "No task specified; infer robot-relevant affordances from the cropped object image alone."
    )
    prompt = (
        f"Task context: {ctx}\n\n"
        f"{_VLM_ROBOT_VIEWPOINT}\n\n"
        "Return ONE JSON object. Keys are snake_case affordance names; values are 0 or 1.\n"
        f"{_VLM_NAMING_CONSTRAINTS}\n\n"
        f"REQUIRED — score every name in {{{names}}} with 0 or 1 (be conservative if unsure).\n"
        "OPTIONAL — add a small number of extra keys only when needed. Each extra must match the "
        f"same naming pattern as the REQUIRED names above and the eight base template "
        f"({', '.join(BASE_VOCAB)}). Use 1 only when the crop clearly supports that property."
    )
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            }
        ],
        response_format={"type": "json_object"},
    )
    if latency_sink is not None:
        latency_sink.append(time.perf_counter() - t0)
    text = resp.choices[0].message.content or "{}"
    data = json.loads(text)
    result: dict[str, int] = {}
    for a in affordances:
        v = data.get(a, data.get(a.replace("able", ""), 0))
        result[a] = int(bool(v))

    def _coerce01(v: object) -> int:
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, int):
            return 1 if v != 0 else 0
        if isinstance(v, float):
            return 1 if v != 0.0 else 0
        return int(bool(v))

    for k, v in data.items():
        raw = str(k).strip()
        if not raw:
            continue
        key_norm = normalize_vlm_affordance_key(raw)
        if not key_norm or key_norm == "unknown_affordance":
            continue
        if key_norm in result:
            result[key_norm] = max(result[key_norm], _coerce01(v))
            continue
        val = _coerce01(v)
        result[key_norm] = max(result.get(key_norm, 0), val)
    return result


def vlm_openai_discover_predict(
    image_path: Path,
    task: str,
    model: str,
    reference_affordances: Sequence[str],
    *,
    latency_sink: list[float] | None = None,
) -> dict[str, int]:
    from openai import OpenAI

    client = OpenAI()
    mime, b64 = _image_bytes_and_mime(image_path)
    mandatory = ", ".join(BASE_VOCAB)
    extra = [x for x in reference_affordances if x not in BASE_VOCAB_SET]
    ref_extra = ", ".join(extra[:400]) if extra else "(none)"
    ctx = task.strip() or (
        "Infer robot-relevant affordances from the cropped object image alone."
    )
    prompt = (
        f"Task context: {ctx}\n\n"
        f"{_VLM_ROBOT_VIEWPOINT}\n\n"
        "Label a single cropped object for the robot. Return ONE JSON object: keys are snake_case "
        "affordance names; each value must be 0 or 1.\n"
        f"The REQUIRED keys list is the only naming template — every other key (known extras or "
        f"new) must match that same word shape and brevity as those eight: [{mandatory}].\n"
        f"{_VLM_NAMING_CONSTRAINTS}\n\n"
        "REQUIRED — include ALL of these keys (none missing), each with 0 or 1: "
        f"[{mandatory}]\n\n"
        "You MAY set 1 on additional known names when they apply (same object): "
        f"[{ref_extra}]\n\n"
        "ENCOURAGED — add a few NEW keys only when a robot property is missing from the lists "
        "above. Each new key must be written in the SAME style as the eight REQUIRED keys "
        f"({mandatory}) — not noun phrases, not a different convention. Use 1 only when "
        "visually grounded. Prefer fewer, cleaner extras."
    )
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            }
        ],
        response_format={"type": "json_object"},
    )
    if latency_sink is not None:
        latency_sink.append(time.perf_counter() - t0)
    text = resp.choices[0].message.content or "{}"
    data = json.loads(text)
    out: dict[str, int] = {}
    for k, v in data.items():
        key = str(k).strip()
        if not key:
            continue
        if isinstance(v, bool):
            out[key] = int(v)
        elif isinstance(v, int):
            out[key] = 1 if v != 0 else 0
        elif isinstance(v, float):
            out[key] = 1 if v != 0.0 else 0
    return out


ProbeFn = Callable[[str, str], int | None]


@dataclass
class CurationConfig:
    image_dir: Path
    memory_path: Path
    output_path: Path
    task: str
    use_vlm: bool
    vlm_backend: str
    vlm_model: str
    use_embeddings: bool
    embedding_model: str
    enable_probing: bool
    probe: ProbeFn | None
    cheap_threshold: float
    update_memory: bool
    memory_snapshot_path: Path | None
    image_glob: str
    name_prior_weight: float
    planner_output_path: Path | None
    discover_affordances: bool
    merge_similarity_threshold: float
    discover_stats_path: Path | None = None
    video_path: Path | None = None
    mapper: EmbeddingMapper | None = None
    show_progress: bool = True
    image_tasks: dict[str, str] | None = None
    max_images: int | None = None
    image_shuffle_seed: int | None = None
    image_one_per_category: bool = False


def expand_and_select_image_paths(cfg: CurationConfig) -> list[Path]:
    expanded = expand_image_paths(cfg.image_dir, cfg.image_glob)
    n0 = len(expanded)
    selected = select_image_paths(
        expanded,
        max_images=cfg.max_images,
        shuffle_seed=cfg.image_shuffle_seed,
        one_per_category=cfg.image_one_per_category,
    )
    if (
        len(selected) != n0
        or cfg.max_images is not None
        or cfg.image_shuffle_seed is not None
        or cfg.image_one_per_category
    ):
        logger.info(
            "Image selection: using %d of %d (max_images=%r, image_shuffle_seed=%r, "
            "image_one_per_category=%s)",
            len(selected),
            n0,
            cfg.max_images,
            cfg.image_shuffle_seed,
            cfg.image_one_per_category,
        )
    return selected


# ---------------------------------------------------------------------------
# V2 CLI
# ---------------------------------------------------------------------------


def _log_vlm_latency_summary(phase: str, latencies: list[float]) -> None:
    """Report wall-clock OpenAI chat.completions latency (one sample per API call)."""
    if not latencies:
        logger.info(
            "VLM latency (%s): no API calls timed (stub, geometry fallback, or empty run).",
            phase,
        )
        return
    total = float(sum(latencies))
    n = len(latencies)
    mean = total / n
    logger.info(
        "VLM latency (%s): %d OpenAI call(s); mean %.3fs/call; total %.1fs; "
        "min %.3fs; max %.3fs",
        phase,
        n,
        mean,
        total,
        min(latencies),
        max(latencies),
    )


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    # Use stdout so terminals that color stderr red (e.g. Cursor) do not treat INFO as errors.
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    # OpenAI SDK uses httpx; it logs every POST at INFO ("HTTP Request: POST ... 200 OK").
    for name in ("httpx", "httpcore", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _resolve_task(
    image_basename: str,
    global_task: str,
    image_tasks: dict[str, str] | None,
) -> str:
    if image_tasks:
        t = image_tasks.get(image_basename, "").strip()
        if t:
            return t
    return global_task


def _geometry_standard_affordances(
    img_path: Path,
    *,
    name_prior_weight: float,
    cheap_threshold: float,
) -> list[str]:
    """Standard-mode generation without VLM: active set = base affordances with fused score >= threshold."""
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return []
    signals = compute_geometry_signals(bgr)
    cheap = cheap_affordance_scores(signals)
    token = infer_object_name_from_filename(img_path.name)
    priors = object_name_priors(token)
    active: list[str] = []
    for aff in BASE_VOCAB:
        g, _ = cheap[aff]
        fs = fuse_geometry_and_name_prior(
            g, priors, aff, name_prior_weight
        )
        if fs >= cheap_threshold:
            active.append(aff)
    return sorted(active)


def _discover_merged_labels(
    img_path: Path,
    task: str,
    global_known: set[str],
    *,
    vlm_model: str,
    mapper: EmbeddingMapper | None,
    merge_threshold: float,
    latency_sink: list[float] | None = None,
) -> dict[str, int]:
    """One discover VLM call + merge; returns canonical name -> 0/1."""
    raw = vlm_openai_discover_predict(
        img_path,
        task,
        vlm_model,
        sorted(global_known),
        latency_sink=latency_sink,
    )
    canonical_vals: dict[str, int] = {}
    for raw_k, val in raw.items():
        canon = merge_vlm_affordance_name(
            raw_k, global_known, mapper, merge_threshold
        )
        v = int(val)
        canonical_vals[canon] = max(canonical_vals.get(canon, 0), v)
    return canonical_vals


def vlm_openai_label_no_task(
    image_path: Path,
    affordance: str,
    model: str,
    *,
    latency_sink: list[float] | None = None,
) -> tuple[int, str]:
    """Structured reasoning + confidence gate; no task / navigation context in the prompt."""
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY unset; labeling stub -> 0 for %s", affordance)
        return 0, ""

    client = OpenAI()
    mime, b64 = _image_bytes_and_mime(image_path)
    prompt = (
        f"You are labeling one cropped object image for a single affordance property: \"{affordance}\".\n"
        f"{_VLM_ROBOT_VIEWPOINT}\n"
        "Do not use any task, instruction, or navigation context beyond the image and this property.\n"
        "Judge only the main object in the crop; ignore irrelevant background at the borders when possible.\n\n"
        "Reason step by step, but be terse. Fill these fields in order; later fields must be consistent "
        "with earlier ones:\n"
        "  1) main_object: short noun phrase for the dominant object you actually see in the crop "
        "(<= 8 words). If you cannot identify a single main object, say \"unclear\".\n"
        f"  2) definition: one-sentence operational meaning of \"{affordance}\" for this robot "
        "(what concrete physical interaction it implies).\n"
        "  3) evidence_for: 1 short sentence citing what visible features in the crop SUPPORT label 1. "
        "If none, write \"none\".\n"
        "  4) evidence_against: 1 short sentence citing what visible features SUPPORT label 0 "
        "(occlusion, wrong object class, scale, fixed/embedded, ambiguity). You MUST attempt this; "
        "if truly nothing, write \"none\".\n"
        "  5) confidence: one of \"high\", \"medium\", \"low\".\n"
        "     - \"high\": the property is clearly supported by the visible main object — a typical "
        "robot would succeed in the obvious way. Minor occlusion, partial crop, or background "
        "clutter are NOT reasons to drop below high if the relevant features are still visible.\n"
        "     - \"medium\": property is plausibly supported and you would lean toward 1, but there "
        "is a real caveat (e.g. shape is borderline, object is partially attached, scale is "
        "uncertain). Use this when you would say \"probably yes\".\n"
        "     - \"low\": main_object is \"unclear\", the property is clearly absent, or "
        "evidence_for and evidence_against are roughly balanced.\n"
        "  6) label: integer. Set 1 if confidence is \"high\" OR \"medium\"; set 0 if "
        "confidence is \"low\".\n\n"
        "Reply with exactly one JSON object, no markdown, with these keys: "
        "\"main_object\", \"definition\", \"evidence_for\", \"evidence_against\", \"confidence\", "
        "\"evidence\", \"label\". \"evidence\" should be a single short sentence (<=40 words) "
        "summarizing your decision; keep it for downstream readers.\n"
        "Example: {\"main_object\": \"red mug\", \"definition\": \"a parallel-jaw gripper can close "
        "around the object and hold it\", \"evidence_for\": \"thin handle and rim within gripper width\", "
        "\"evidence_against\": \"none\", \"confidence\": \"high\", "
        "\"evidence\": \"Mug handle gives a clean, small graspable feature.\", \"label\": 1}"
    )
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            }
        ],
        response_format={"type": "json_object"},
    )
    if latency_sink is not None:
        latency_sink.append(time.perf_counter() - t0)
    text = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Label JSON parse failed for %s %s", image_path.name, affordance)
        return 0, ""
    raw_ev = data.get("evidence", "")
    evidence = " ".join(str(raw_ev).strip().split())[:500]
    conf = str(data.get("confidence", "")).strip().lower()
    v = data.get("label", 0)
    if isinstance(v, bool):
        raw_lab = int(v)
    elif isinstance(v, (int, float)):
        raw_lab = 1 if v != 0 else 0
    else:
        raw_lab = int(bool(v))
    # Gate: trust model's positive label when confidence is "high" or "medium";
    # force 0 only when the model itself flags low confidence (unclear/balanced).
    if conf == "low":
        lab = 0
    elif conf in ("high", "medium"):
        lab = raw_lab
    else:
        # Unknown / missing confidence string: fall back to the raw label.
        lab = raw_lab
    return lab, evidence


@app.command("generate")
def cmd_generate(
    image_dir: Path = typer.Option(..., "--image-dir", help="Directory of object crops."),
    memory_path: Path = typer.Option(
        ...,
        "--memory",
        help="Read-only memory JSONL (use empty file for no priors).",
    ),
    affordance_sets_out: Path = typer.Option(
        ...,
        "--affordance-sets-out",
        help=(
            "Output JSONL: one object per line with image, affordances (active names), "
            "novel_affordances (affordances not in BASE_VOCAB), gen_mode."
        ),
    ),
    gen_mode: str = typer.Option(
        "standard",
        "--gen-mode",
        help="standard: base vocab only (VLM and/or geometry). discover: open-ended VLM + merge.",
    ),
    task: str = typer.Option("", "--task", help="Global task for generation (VLM prompts)."),
    tasks_jsonl: Path | None = typer.Option(
        None, "--tasks-jsonl", help="Per-image tasks JSONL (image, task)."
    ),
    use_vlm: bool = typer.Option(
        True,
        "--use-vlm/--no-use-vlm",
        help="If set, use OpenAI VLM for generation when gen_mode matches.",
    ),
    vlm_backend: str = typer.Option("openai", "--vlm-backend", help="openai | stub"),
    vlm_model: str = typer.Option("gpt-5-mini", "--vlm-model"),
    use_embeddings: bool = typer.Option(
        False, "--use-embeddings/--no-use-embeddings", help="Discover: merge novel names via embeddings."
    ),
    merge_similarity_threshold: float = typer.Option(
        0.82,
        "--merge-similarity-threshold",
        min=0.0,
        max=1.0,
    ),
    name_prior_weight: float = typer.Option(
        0.25,
        "--name-prior-weight",
        min=0.0,
        max=1.0,
        help="Standard geometry path: filename prior blend weight.",
    ),
    cheap_threshold: float = typer.Option(
        0.5,
        "--cheap-threshold",
        min=0.0,
        max=1.0,
        help="Standard geometry path: affordance active if fused score >= this.",
    ),
    image_glob: str = typer.Option(_DEFAULT_IMAGE_GLOB, "--glob"),
    max_images: int | None = typer.Option(None, "--max-images", min=1),
    image_shuffle_seed: int | None = typer.Option(None, "--image-shuffle-seed"),
    image_one_per_category: bool = typer.Option(
        False,
        "--image-one-per-category/--no-image-one-per-category",
    ),
    embedding_model: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2",
        "--embedding-model",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    no_progress: bool = typer.Option(False, "--no-progress"),
) -> None:
    """Task-conditioned affordance-name lists per image; writes affordance-sets JSONL."""
    _setup_logging(verbose)
    gm = gen_mode.strip().lower()
    if gm not in ("standard", "discover"):
        raise typer.BadParameter("--gen-mode must be 'standard' or 'discover'")
    if gm == "discover" and not use_vlm:
        logger.warning("discover gen_mode expects --use-vlm; geometry cannot invent novel names.")

    image_tasks: dict[str, str] | None = None
    if tasks_jsonl is not None:
        if not tasks_jsonl.is_file():
            raise typer.BadParameter(f"Not a file: {tasks_jsonl}")
        image_tasks = load_image_tasks_jsonl(tasks_jsonl)

    memory_rows = load_memory_rows_jsonl(memory_path)
    global_known_seed: set[str] = set(BASE_VOCAB) | affordance_names_from_memory_rows(
        memory_rows
    )

    cfg_like = CurationConfig(
        image_dir=image_dir,
        memory_path=memory_path,
        output_path=affordance_sets_out,
        task=task,
        use_vlm=use_vlm,
        vlm_backend=vlm_backend,
        vlm_model=vlm_model,
        use_embeddings=use_embeddings,
        embedding_model=embedding_model,
        enable_probing=False,
        probe=None,
        cheap_threshold=cheap_threshold,
        update_memory=False,
        memory_snapshot_path=None,
        image_glob=image_glob,
        name_prior_weight=name_prior_weight,
        planner_output_path=None,
        discover_affordances=gm == "discover",
        merge_similarity_threshold=merge_similarity_threshold,
        discover_stats_path=None,
        video_path=None,
        mapper=None,
        show_progress=not no_progress,
        image_tasks=image_tasks,
        max_images=max_images,
        image_shuffle_seed=image_shuffle_seed,
        image_one_per_category=image_one_per_category,
    )
    paths = expand_and_select_image_paths(cfg_like)
    if not paths:
        logger.warning("No images under %s", image_dir)

    mapper: EmbeddingMapper | None = None
    if gm == "discover" and use_embeddings and AutoModel is not None:
        try:
            mapper = EmbeddingMapper(embedding_model)
        except Exception as e:
            logger.warning("Embeddings disabled: %s", e)

    affordance_sets_out.parent.mkdir(parents=True, exist_ok=True)
    if affordance_sets_out.exists():
        affordance_sets_out.unlink()
    n_written = 0
    global_known = set(global_known_seed)
    generation_latencies: list[float] = []

    def _memory_positive_for_image(basename: str) -> set[str]:
        return {
            str(r["affordance"])
            for r in memory_rows
            if str(r["image"]) == basename and int(r.get("label", 0)) == 1
        }

    iter_paths = paths
    if not no_progress and paths:
        iter_paths = tqdm(paths, desc="V2 generate", unit="img", file=sys.stdout)

    for img_path in iter_paths:
        name = img_path.name
        task_here = _resolve_task(name, task, image_tasks)
        affordances: list[str] = []

        if gm == "standard":
            if (
                use_vlm
                and vlm_backend == "openai"
                and os.environ.get("OPENAI_API_KEY")
            ):
                preds = vlm_openai_predict(
                    img_path,
                    task_here,
                    list(BASE_VOCAB),
                    vlm_model,
                    latency_sink=generation_latencies,
                )
                affordances = sorted(a for a, v in preds.items() if int(v) == 1)
            elif use_vlm and vlm_backend == "openai":
                logger.warning("OPENAI_API_KEY unset; falling back to geometry for %s", name)
                affordances = _geometry_standard_affordances(
                    img_path,
                    name_prior_weight=name_prior_weight,
                    cheap_threshold=cheap_threshold,
                )
            else:
                affordances = _geometry_standard_affordances(
                    img_path,
                    name_prior_weight=name_prior_weight,
                    cheap_threshold=cheap_threshold,
                )

        else:  # discover
            if use_vlm and vlm_backend == "openai" and os.environ.get("OPENAI_API_KEY"):
                mem_image = {
                    str(r["affordance"])
                    for r in memory_rows
                    if str(r["image"]) == name
                }
                known = set(global_known) | mem_image
                merged = _discover_merged_labels(
                    img_path,
                    task_here,
                    known,
                    vlm_model=vlm_model,
                    mapper=mapper,
                    merge_threshold=merge_similarity_threshold,
                    latency_sink=generation_latencies,
                )
                affordances = sorted(k for k, v in merged.items() if v == 1)
                global_known.update(merged.keys())
            else:
                logger.warning("Discover mode without OpenAI; skipping %s", name)
                affordances = []

        affordances = sorted(
            set(affordances) | _memory_positive_for_image(name)
        )
        affordances, novel_affordances = _filter_redundant_novel_affordances(affordances)

        record = {
            "image": name,
            "affordances": affordances,
            "novel_affordances": novel_affordances,
            "gen_mode": gm,
        }
        with affordance_sets_out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        n_written += 1

    logger.info("Wrote %d affordance-set line(s) to %s", n_written, affordance_sets_out)
    _log_vlm_latency_summary("generation", generation_latencies)


@app.command("label")
def cmd_label(
    image_dir: Path = typer.Option(..., "--image-dir"),
    affordance_sets: Path = typer.Option(
        ...,
        "--affordance-sets",
        help="JSONL from `generate` (--affordance-sets-out): per-image affordance name lists.",
    ),
    output_path: Path = typer.Option(
        ...,
        "--output",
        help="Output JSONL: image, affordance, label per line; VLM rows may include evidence.",
    ),
    vlm_model: str = typer.Option("gpt-5-mini", "--vlm-model"),
    pad_base_vocab_zero: bool = typer.Option(
        True,
        "--pad-base-vocab-zero/--no-pad-base-vocab-zero",
        help=(
            "After VLM calls, add (image, affordance, 0) for each BASE_VOCAB name "
            "missing for that image so compare_jsonl closed-set has no implicit gaps."
        ),
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    no_progress: bool = typer.Option(False, "--no-progress"),
) -> None:
    """Optional stage: binary labels per (image, affordance) without task in the VLM prompt."""
    _setup_logging(verbose)
    pairs: list[tuple[str, str]] = []
    with affordance_sets.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            img = str(r["image"])
            for aff in r.get("affordances") or []:
                pairs.append((img, str(aff)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file():
        output_path.unlink()

    label_latencies: list[float] = []
    written: set[tuple[str, str]] = set()
    it = pairs
    if not no_progress and pairs:
        it = tqdm(pairs, desc="V2 label", unit="pair", file=sys.stdout)

    for image_name, affordance in it:
        img_path = image_dir / image_name
        if not img_path.is_file():
            logger.warning("Missing image, skipping %s", img_path)
            continue
        lab, ev_text = vlm_openai_label_no_task(
            img_path, affordance, vlm_model, latency_sink=label_latencies
        )
        row: dict[str, str | int] = {
            "image": image_name,
            "affordance": affordance,
            "label": lab,
            "evidence": ev_text,
        }
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        written.add((image_name, affordance))

    if pad_base_vocab_zero:
        images = sorted({p[0] for p in pairs})
        extra = 0
        with output_path.open("a", encoding="utf-8") as f:
            for image_name in images:
                for aff in BASE_VOCAB:
                    if (image_name, aff) in written:
                        continue
                    row = {
                        "image": image_name,
                        "affordance": aff,
                        "label": 0,
                        "evidence": "",
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    extra += 1
        logger.info("Padded %d base-vocab zero row(s) for compare_jsonl.", extra)

    logger.info("Wrote labels (from affordance sets + optional base padding) to %s", output_path)
    _log_vlm_latency_summary("labeling", label_latencies)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
