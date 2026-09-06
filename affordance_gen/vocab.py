"""Shared controlled vocabulary for affordance_gen and evaluation (compare_jsonl)."""

from __future__ import annotations

# Keep in sync with affordance heuristics in affordance_gen (geometry covers these eight).
BASE_VOCAB: tuple[str, ...] = (
    "containable",
    "graspable",
    "liftable",
    "movable",
    "openable",
    "rollable",
    "stackable",
    "supportable",
)

BASE_VOCAB_SET: frozenset[str] = frozenset(BASE_VOCAB)
