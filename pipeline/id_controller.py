
import os
import time
import random
from typing import List, Tuple, Dict, Optional

import pandas as pd


# ============================================================
# ID Controller (ROBUST: supports ID / id case-insensitive)
# ============================================================

def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# ---------- column resolver ----------
def _resolve_id_column(df: pd.DataFrame) -> str:
    """
    Resolve ID column in a case-insensitive way.
    Accepts: ID, id, Id, iD
    """
    for c in df.columns:
        if c.lower() == "id":
            return c
    raise ValueError(f"[ID_CONTROLLER] No ID/id column found. Columns: {list(df.columns)}")


# ---------- readers ----------
def _read_csv_ids(path: str) -> List[str]:
    """
    Reads a CSV that contains an ID column (case-insensitive).
    Returns unique IDs preserving first-seen order.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []

    df = pd.read_csv(path, encoding="utf-8")
    id_col = _resolve_id_column(df)

    ids = df[id_col].astype(str).tolist()
    ids = [i.strip() for i in ids if i and str(i).strip()]

    out, seen = [], set()
    for x in ids:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _write_ids_csv(ids: List[str], path: str) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    pd.DataFrame({"ID": [str(i) for i in ids]}).to_csv(
        path, index=False, encoding="utf-8"
    )


def _read_dataset_ids(dataset_path: str) -> List[str]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"[ID_CONTROLLER] Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path, encoding="utf-8")
    id_col = _resolve_id_column(df)

    ids = df[id_col].astype(str).tolist()
    ids = [i.strip() for i in ids if i and str(i).strip()]

    out, seen = [], set()
    for x in ids:
        if x not in seen:
            out.append(x)
            seen.add(x)

    if not out:
        raise ValueError(f"[ID_CONTROLLER] Dataset has zero usable IDs: {dataset_path}")

    return out


def _read_generated_ids(generated_csv_path: str) -> List[str]:
    """
    Reads Generated_Fake_News.csv (or equivalent).
    Returns unique IDs preserving first-seen order.
    """
    if not os.path.exists(generated_csv_path) or os.path.getsize(generated_csv_path) == 0:
        return []

    df = pd.read_csv(generated_csv_path, encoding="utf-8")
    id_col = _resolve_id_column(df)

    ids = df[id_col].astype(str).tolist()
    ids = [i.strip() for i in ids if i and str(i).strip()]

    out, seen = [], set()
    for x in ids:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


# ---------- sampling ----------
def _deterministic_pick(pool: List[str], k: int, seed: int) -> List[str]:
    """
    Deterministic sampling based on python's RNG (stable across pandas versions).
    Preserves the order returned by random.sample (deterministic given pool order + seed).
    """
    if k <= 0:
        return []
    if k > len(pool):
        raise ValueError(f"[ID_CONTROLLER] desired_n={k} > pool_size={len(pool)}")

    rng = random.Random(seed)
    return [str(x) for x in rng.sample(list(pool), k)]


# ============================================================
# Public API
# ============================================================
def ensure_sample_ids(
    sample_ids_path: str,
    dataset_path: str,
    desired_n: int,
    mode: str = "keep",
    random_state: int = 123,
    recreate_if_missing: bool = True,
    recreate_force: bool = False,
) -> Dict[str, object]:
    """
    Ensures that sample_ids.csv exists and matches desired behavior.

    mode:
      - keep:    if exists, keep as is
      - expand:  if exists and desired_n > current, append new IDs (no duplicates)
      - rebuild: rebuild sample deterministically to desired_n

    recreate_if_missing:
      - if sample missing/empty and mode==keep, you can decide whether to create or not.

    recreate_force:
      - forces rebuild regardless of mode.
    """
    mode = mode.lower().strip()
    if mode not in ("keep", "expand", "rebuild"):
        raise ValueError("mode must be one of: keep | expand | rebuild")

    if desired_n is None or int(desired_n) < 0:
        raise ValueError("[ID_CONTROLLER] desired_n must be a non-negative integer")

    desired_n = int(desired_n)

    sample_exists = os.path.exists(sample_ids_path) and os.path.getsize(sample_ids_path) > 0
    dataset_ids = _read_dataset_ids(dataset_path)

    effective_mode = "rebuild" if recreate_force else mode

    # ---- missing sample_ids ----
    if not sample_exists:
        if not recreate_if_missing and effective_mode == "keep":
            return {
                "mode": "keep_missing_no_create",
                "sample_n": 0,
                "added_n": 0,
                "added_ids": [],
                "notes": ["sample_ids missing, no creation"],
            }

        sample_ids = _deterministic_pick(dataset_ids, desired_n, random_state)
        _write_ids_csv(sample_ids, sample_ids_path)

        return {
            "mode": "rebuild_created",
            "sample_n": len(sample_ids),
            "added_n": len(sample_ids),
            "added_ids": sample_ids,
            "notes": [
                f"created deterministic sample (seed={random_state})",
                f"path={sample_ids_path}",
                f"timestamp={_now()}",
            ],
        }

    # ---- sample exists ----
    sample_ids = _read_csv_ids(sample_ids_path)

    if effective_mode == "keep":
        return {
            "mode": "keep",
            "sample_n": len(sample_ids),
            "added_n": 0,
            "added_ids": [],
            "notes": ["kept existing sample_ids"],
        }

    if effective_mode == "rebuild":
        sample_ids_new = _deterministic_pick(dataset_ids, desired_n, random_state)
        _write_ids_csv(sample_ids_new, sample_ids_path)
        return {
            "mode": "rebuild",
            "sample_n": len(sample_ids_new),
            "added_n": len(sample_ids_new),
            "added_ids": sample_ids_new,
            "notes": ["rebuilt deterministic sample"],
        }

    # ---- expand ----
    need = max(0, desired_n - len(sample_ids))
    existing = set(sample_ids)

    if need > 0:
        remaining = [i for i in dataset_ids if i not in existing]
        add = _deterministic_pick(remaining, need, random_state)
        sample_ids_expanded = sample_ids + add
        _write_ids_csv(sample_ids_expanded, sample_ids_path)
        return {
            "mode": "expand",
            "sample_n": len(sample_ids_expanded),
            "added_n": len(add),
            "added_ids": add,
            "notes": ["expanded deterministic sample (kept previous IDs, appended new ones)"],
        }
    else:
        return {
            "mode": "expand",
            "sample_n": len(sample_ids),
            "added_n": 0,
            "added_ids": [],
            "notes": ["expand: sample already >= desired_n, downstream cleanup will still run"],
        }


def compute_missing_ids(
    sample_ids_path: str,
    generated_csv_path: str,
    skipped_csv_path: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """
    Returns (missing_ids, done_ids) **preserving the order of sample_ids**.

    This is crucial for robust resume:
      - missing_ids will be processed in the same stable order every time
      - no sorting (sorting breaks 'resume from last unprocessed' intuition)
    """
    sample_ids = _read_csv_ids(sample_ids_path)
    gen_ids = _read_generated_ids(generated_csv_path)
    skipped_ids = set(_read_csv_ids(skipped_csv_path)) if skipped_csv_path else set()

    gen_set = set(gen_ids)

    done_in_sample_order = [i for i in sample_ids if i in gen_set]
    missing_in_sample_order = [i for i in sample_ids if (i not in gen_set) and (i not in skipped_ids)]

    return missing_in_sample_order, done_in_sample_order
