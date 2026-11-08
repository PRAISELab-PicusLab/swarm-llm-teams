import os
import ast
import json
import re
import time
from typing import Optional, Dict, Any, List, Set, Tuple

import pandas as pd
from tqdm import tqdm

FLUSH_EVERY_N_IDS = 1

from .agents import (
    SemanticAnalyzer,
    SalientSentenceEditor,
    NumberModifier,
    NarrativeModifier,
    TitleEditor,
    call_ollama_structured,
    get_agent_json_schema,
    get_last_agent_message,
    user_proxy,
)

from .check_json_format import (
    check_semantic_analyzer_format,
    check_salient_sentence_editor_format,
    check_number_modifier_format,
    check_narrative_modifier_format,
    check_title_editor_format,
)

from .warn_logger import warn as _warn, set_context as _warn_set_context

from sentence_transformers import SentenceTransformer, util

import nltk
from nltk.tokenize import sent_tokenize



JSON_ONLY_SUFFIX = (
    "\n\nIMPORTANT: Return ONLY a single valid JSON object. "
    "Do not add any commentary, markdown, or extra text."
)


class MultiAgentGenerationPipeline:
   

    def __init__(self, embedder_name: str = "all-MiniLM-L6-v2"):
        self.embedder_name = embedder_name
        self._embedder: SentenceTransformer | None = None
        self._nltk_ready = False

    def _ensure_nltk(self) -> None:
        if self._nltk_ready:
            return
        nltk.download("punkt", quiet=True)
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            # Some NLTK versions do not ship punkt_tab.
            pass
        self._nltk_ready = True

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embedder_name)
        return self._embedder


    def run_generation_pipeline(
        self,
        df_input: pd.DataFrame,
        base_out_dir: str,
        sample_size: int | None = 100,
        sample_ids_path: str | None = None,
        resume: bool = True,
        process_only_missing_round0: bool = False,
        df_prev: pd.DataFrame | None = None,
        prev_round_dir: str | None = None,
        prev_generation_dir: str | None = None,
        prev_orch_dir: str | None = None,
        detector_name: str | None = None,
    ) -> pd.DataFrame:
       
        return _run_generation_pipeline_impl(
            self,
            df_input=df_input,
            base_out_dir=base_out_dir,
            sample_size=sample_size,
            sample_ids_path=sample_ids_path,
            resume=resume,
            process_only_missing_round0=process_only_missing_round0,
            df_prev=df_prev,
            prev_round_dir=prev_round_dir,
            prev_generation_dir=prev_generation_dir,
            prev_orch_dir=prev_orch_dir,
            detector_name=detector_name,
        )

# Default singleton used by module-level helper functions.
_PIPELINE = MultiAgentGenerationPipeline()

def run_generation_pipeline(
    df_input: pd.DataFrame,
    base_out_dir: str,
    sample_size: int | None = 100,
    sample_ids_path: str | None = None,
    resume: bool = True,
    process_only_missing_round0: bool = False,
    df_prev: pd.DataFrame | None = None,
    prev_round_dir: str | None = None,
    prev_generation_dir: str | None = None,
    prev_orch_dir: str | None = None,
    detector_name: str | None = None,
) -> pd.DataFrame:
    """Backward-compatible module-level wrapper."""
    return _PIPELINE.run_generation_pipeline(
        df_input=df_input,
        base_out_dir=base_out_dir,
        sample_size=sample_size,
        sample_ids_path=sample_ids_path,
        resume=resume,
        process_only_missing_round0=process_only_missing_round0,
        df_prev=df_prev,
        prev_round_dir=prev_round_dir,
        prev_generation_dir=prev_generation_dir,
        prev_orch_dir=prev_orch_dir,
        detector_name=detector_name,
    )



def _ensure_dir(p: str) -> None:
    # NB: su filesystem paralleli (Lustre) makedirs(exist_ok=True)
    # puo' sollevare FileExistsError in modo SPURIO sotto carico concorrente: il
    # ricontrollo interno isdir() di CPython puo' fallire per lag dei metadati anche
    # se la dir esiste gia'. Rendiamo la creazione idempotente: se il path e' gia'
    # una directory, e' un successo; rilanciamo solo se e' un vero file non-dir.
    try:
        os.makedirs(p, exist_ok=True)
    except FileExistsError:
        if not os.path.isdir(p):
            raise

def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _infer_round_index_from_path(base_out_dir: str) -> int:
    s = str(base_out_dir or "")
    m = re.search(r"round[_\- ]?(\d+)", s, flags=re.IGNORECASE)
    if not m:
        return -999
    try:
        return int(m.group(1))
    except Exception:
        return -999

def _get_prev_round_idx(round_idx: int) -> int:
    if round_idx is None:
        return -1
    if round_idx == -999:
        return -1
    return int(round_idx) - 1

def _find_output_root(start_dir: str) -> Optional[str]:
    cur = os.path.abspath(start_dir)
    for _ in range(0, 10):
        if os.path.basename(cur).lower() == "output":
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None

def _read_ids_from_csv(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        return set()
    if "ID" not in df.columns:
        return set()
    return set(df["ID"].astype(str).tolist())

def _safe_append_dedupe_csv(
    path: str,
    df_new: pd.DataFrame,
    subset_cols: Optional[List[str]],
    keep: str = "last",
):
    if df_new is None or df_new.empty:
        return
    if os.path.exists(path):
        try:
            df_old = pd.read_csv(path, encoding="utf-8")
        except Exception:
            df_old = df_new.iloc[0:0].copy()
    else:
        df_old = df_new.iloc[0:0].copy()

    df_all = pd.concat([df_old, df_new], ignore_index=True)
    if subset_cols is None:
        df_all = df_all.drop_duplicates(keep=keep)
    else:
        ok = True
        for c in subset_cols:
            if c not in df_all.columns:
                ok = False
        if ok:
            df_all = df_all.drop_duplicates(subset=subset_cols, keep=keep)
    _ensure_dir(os.path.dirname(path) or ".")
    df_all.to_csv(path, index=False, encoding="utf-8")

def _safe_save_generated(final_output_path: str, new_rows: pd.DataFrame):
    cols = ["ID", "title", "text"]
    new_rows = new_rows[cols].copy()
    new_rows["ID"] = new_rows["ID"].astype(str)

    if os.path.exists(final_output_path):
        try:
            df_old = pd.read_csv(final_output_path, encoding="utf-8")
        except Exception:
            df_old = pd.DataFrame(columns=cols)
        if "ID" in df_old.columns:
            df_old["ID"] = df_old["ID"].astype(str)
        else:
            df_old = pd.DataFrame(columns=cols)
    else:
        df_old = pd.DataFrame(columns=cols)

    df_all = pd.concat([df_old[cols], new_rows], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["ID"], keep="first")
    _ensure_dir(os.path.dirname(final_output_path) or ".")
    df_all.to_csv(final_output_path, index=False, encoding="utf-8")

def _safe_save_generated_subset(final_output_path: str, new_rows: pd.DataFrame, keep_ids: Set[str]):
    """Round>0 strict: keep ONLY active IDs; do NOT accumulate across rounds."""
    cols = ["ID", "title", "text"]
    keep_ids = set(str(x) for x in (keep_ids or set()))

    if new_rows is None:
        new_rows = pd.DataFrame(columns=cols)
    else:
        new_rows = new_rows.copy()
    for c in cols:
        if c not in new_rows.columns:
            new_rows[c] = ""
    new_rows = new_rows[cols].copy()
    new_rows["ID"] = new_rows["ID"].astype(str)
    new_rows = new_rows[new_rows["ID"].isin(keep_ids)].copy()

    if os.path.exists(final_output_path):
        try:
            df_old = pd.read_csv(final_output_path, encoding="utf-8")
        except Exception:
            df_old = pd.DataFrame(columns=cols)
    else:
        df_old = pd.DataFrame(columns=cols)

    for c in cols:
        if c not in df_old.columns:
            df_old[c] = ""
    df_old = df_old[cols].copy()
    df_old["ID"] = df_old["ID"].astype(str)
    df_old = df_old[df_old["ID"].isin(keep_ids)].copy()

    df_all = pd.concat([df_old, new_rows], ignore_index=True)
    df_all["ID"] = df_all["ID"].astype(str)
    df_all = df_all.drop_duplicates(subset=["ID"], keep="last")

    _ensure_dir(os.path.dirname(final_output_path) or ".")
    df_all.to_csv(final_output_path, index=False, encoding="utf-8")

def _write_panel_parquet(panel_path: str, df_panel: pd.DataFrame) -> None:
    _ensure_dir(os.path.dirname(panel_path) or ".")
    try:
        df_panel.to_parquet(panel_path, index=False)
    except Exception as e:
        csv_fallback = os.path.splitext(panel_path)[0] + ".csv"
        df_panel.to_csv(csv_fallback, index=False, encoding="utf-8")
        print(f"[PIPE][WARN] Parquet failed -> wrote CSV fallback: {csv_fallback}. Error: {e}")

def _append_skip(df_skips: pd.DataFrame, article_id: str, stage: str, reason: str):
    df_skips.loc[len(df_skips)] = [str(article_id), str(stage), str(reason)[:4000]]

def _append_reuse(df_reuse: pd.DataFrame, article_id: str, what: str, reason: str):
    df_reuse.loc[len(df_reuse)] = [str(article_id), str(what), str(reason)[:4000]]


def _atomic_write_df_csv(df: pd.DataFrame, path: str) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)

def _atomic_write_panel(panel_path: str, df_panel: pd.DataFrame) -> None:
    """
    Atomic write for panel:
    - try parquet -> write tmp parquet then replace
    - fallback csv -> write tmp csv then replace
    """
    _ensure_dir(os.path.dirname(panel_path) or ".")
    tmp_parquet = panel_path + ".tmp"
    try:
        df_panel.to_parquet(tmp_parquet, index=False)
        os.replace(tmp_parquet, panel_path)
        return
    except Exception:
        try:
            if os.path.exists(tmp_parquet):
                os.remove(tmp_parquet)
        except Exception:
            pass

    csv_fallback = os.path.splitext(panel_path)[0] + ".csv"
    tmp_csv = csv_fallback + ".tmp"
    df_panel.to_csv(tmp_csv, index=False, encoding="utf-8")
    os.replace(tmp_csv, csv_fallback)

def _persist_logs_atomic(
    *,
    df_skips: pd.DataFrame,
    skips_path: str,
    df_reuse: pd.DataFrame,
    reuse_path: str,
    df_roundres: pd.DataFrame,
    round_results_path: str,
) -> None:
    try:
        _atomic_write_df_csv(df_skips, skips_path)
    except Exception:
        pass
    try:
        _atomic_write_df_csv(df_reuse, reuse_path)
    except Exception:
        pass
    try:
        _atomic_write_df_csv(df_roundres, round_results_path)
    except Exception:
        pass

def _persist_panel_atomic(
    *,
    panel_path: str,
    round_idx_for_panel: int,
    target_ids: List[str],
    gen_csv_path: str,
    skipped_ids_round: Set[str],
) -> None:
    try:
        # Bug-4 fix: pass gen_dir so the panel includes SA/SSE/NM/RR enriched columns
        _gen_dir = os.path.dirname(gen_csv_path) if gen_csv_path else None
        df_panel = _build_panel_round_strict(
            round_idx=round_idx_for_panel,
            target_ids=target_ids,
            gen_csv_path=gen_csv_path,
            skipped_ids_round=skipped_ids_round,
            gen_dir=_gen_dir,
        )
        _atomic_write_panel(panel_path, df_panel)
    except Exception:
        pass

def _skipped_ids_from_df(df_skips: pd.DataFrame) -> Set[str]:
    if df_skips is None or df_skips.empty or "ID" not in df_skips.columns:
        return set()
    return set(df_skips["ID"].astype(str).tolist())

def _append_skip_and_flush(
    *,
    df_skips: pd.DataFrame,
    article_id: str,
    stage: str,
    reason: str,
    skips_path: str,
    df_reuse: pd.DataFrame,
    reuse_path: str,
    df_roundres: pd.DataFrame,
    round_results_path: str,
    panel_path: str,
    round_idx_for_panel: int,
    panel_target_ids: List[str],
    gen_csv_path: str,
) -> None:
    _append_skip(df_skips, article_id, stage, reason)
    _persist_logs_atomic(
        df_skips=df_skips,
        skips_path=skips_path,
        df_reuse=df_reuse,
        reuse_path=reuse_path,
        df_roundres=df_roundres,
        round_results_path=round_results_path,
    )
    _persist_panel_atomic(
        panel_path=panel_path,
        round_idx_for_panel=round_idx_for_panel,
        target_ids=[str(x) for x in panel_target_ids],
        gen_csv_path=gen_csv_path,
        skipped_ids_round=_skipped_ids_from_df(df_skips),
    )

def _append_reuse_and_flush(
    *,
    df_reuse: pd.DataFrame,
    article_id: str,
    what: str,
    reason: str,
    df_skips: pd.DataFrame,
    skips_path: str,
    reuse_path: str,
    df_roundres: pd.DataFrame,
    round_results_path: str,
    panel_path: str,
    round_idx_for_panel: int,
    panel_target_ids: List[str],
    gen_csv_path: str,
) -> None:
    _append_reuse(df_reuse, article_id, what, reason)
    _persist_logs_atomic(
        df_skips=df_skips,
        skips_path=skips_path,
        df_reuse=df_reuse,
        reuse_path=reuse_path,
        df_roundres=df_roundres,
        round_results_path=round_results_path,
    )
    _persist_panel_atomic(
        panel_path=panel_path,
        round_idx_for_panel=round_idx_for_panel,
        target_ids=[str(x) for x in panel_target_ids],
        gen_csv_path=gen_csv_path,
        skipped_ids_round=_skipped_ids_from_df(df_skips),
    )

def _append_roundres_and_flush(
    *,
    df_roundres: pd.DataFrame,
    rr: Dict[str, Any],
    df_skips: pd.DataFrame,
    skips_path: str,
    df_reuse: pd.DataFrame,
    reuse_path: str,
    round_results_path: str,
    panel_path: str,
    round_idx_for_panel: int,
    panel_target_ids: List[str],
    gen_csv_path: str,
) -> None:
    df_roundres.loc[len(df_roundres)] = rr
    _persist_logs_atomic(
        df_skips=df_skips,
        skips_path=skips_path,
        df_reuse=df_reuse,
        reuse_path=reuse_path,
        df_roundres=df_roundres,
        round_results_path=round_results_path,
    )
    _persist_panel_atomic(
        panel_path=panel_path,
        round_idx_for_panel=round_idx_for_panel,
        target_ids=[str(x) for x in panel_target_ids],
        gen_csv_path=gen_csv_path,
        skipped_ids_round=_skipped_ids_from_df(df_skips),
    )

def _flush_all_after_id(
    *,
    df_skips: pd.DataFrame,
    skips_path: str,
    df_reuse: pd.DataFrame,
    reuse_path: str,
    df_roundres: pd.DataFrame,
    round_results_path: str,
    panel_path: str,
    round_idx_for_panel: int,
    panel_target_ids: List[str],
    gen_csv_path: str,
) -> None:
    _persist_logs_atomic(
        df_skips=df_skips,
        skips_path=skips_path,
        df_reuse=df_reuse,
        reuse_path=reuse_path,
        df_roundres=df_roundres,
        round_results_path=round_results_path,
    )
    _persist_panel_atomic(
        panel_path=panel_path,
        round_idx_for_panel=round_idx_for_panel,
        target_ids=[str(x) for x in panel_target_ids],
        gen_csv_path=gen_csv_path,
        skipped_ids_round=_skipped_ids_from_df(df_skips),
    )


def _realign_numbers_post_sse(
    original_sentence: str,
    modified_sentence: str,
    n2s_map: Dict[str, str],
) -> Tuple[List[str], Dict[str, str], List[str], List[str]]:
    """Returns (numbers_in_original, numbers_in_modified, lost_numbers, renamed_numbers).
    numbers_in_modified is a dict {original_form: current_form} for surviving numbers."""
    orig_norm = re.sub(r"\s+", " ", (original_sentence or "")).strip()
    anchored = [num for num, sent in n2s_map.items()
                if re.sub(r"\s+", " ", (sent or "")).strip() == orig_norm]
    if not anchored and orig_norm:
        anchored = [num for num, sent in n2s_map.items()
                    if re.sub(r"\s+", " ", (sent or "")).strip() in orig_norm]

    numbers_in_original: List[str] = list(anchored)
    numbers_in_modified: Dict[str, str] = {}
    lost_numbers: List[str] = []
    renamed_numbers: List[str] = []

    # Ordered numeric token lists for positional fallback (handles value changes by SSE)
    _orig_num_tokens = re.findall(r"[\d][\d,\.]*", original_sentence or "")
    _mod_num_tokens  = re.findall(r"[\d][\d,\.]*", modified_sentence or "")

    for num in numbers_in_original:
        # 1. Exact string match
        if num in modified_sentence:
            numbers_in_modified[num] = num
            continue
        digits_num = re.sub(r"[^\d]", "", num)
        if digits_num:
            tokens = re.findall(r"\S+", modified_sentence)
            # 2. Exact digit match (e.g. "1,234" ↔ "1234")
            matched = next(
                (tok for tok in tokens if re.sub(r"[^\d]", "", tok) == digits_num),
                None,
            )
            if matched is None:
                # 3. Digit substring (loose, original behaviour)
                matched = next(
                    (tok for tok in tokens if digits_num in re.sub(r"[^\d]", "", tok)),
                    None,
                )
            if matched:
                numbers_in_modified[num] = matched
                if matched != num:
                    renamed_numbers.append(matched)
                continue
        # 4. Positional fallback: SSE changed the number's value but kept it in the same
        #    ordinal position within the sentence (e.g. "123 million" → "456 million").
        try:
            orig_digits = [re.sub(r"[^\d]", "", t) for t in _orig_num_tokens]
            num_digits  = re.sub(r"[^\d]", "", num)
            if num_digits and num_digits in orig_digits:
                pos = orig_digits.index(num_digits)
                if pos < len(_mod_num_tokens):
                    mapped = _mod_num_tokens[pos]
                    numbers_in_modified[num] = mapped
                    if mapped != num:
                        renamed_numbers.append(mapped)
                    continue
        except Exception:
            pass
        lost_numbers.append(num)

    return numbers_in_original, numbers_in_modified, lost_numbers, renamed_numbers


def _repair_lost_numbers_sse(
    original_sentence: str,
    modified_sentence: str,
    lost_numbers: List[str],
) -> Tuple[str, List[str]]:
    """Re-insert key numbers that SSE dropped from the sentence.

    For each lost number, locates its substring boundary in original_sentence,
    identifies the immediately adjacent word to the left (left anchor) and right
    (right anchor), then splices the number adjacent to whichever anchor is found
    in modified_sentence.

    PARENTHETICAL FALLBACK DISABLED (2026-05-15): when no anchor matches, the
    number is NOT appended as " (N)" anymore — that produced corrupted artifacts
    like "3 5 percent0%" because the anchor matching is fuzzy. Instead the number
    is returned in `unrepaired_lost` for downstream auditing; the sentence is
    left unchanged.

    Returns (repaired_sentence, unrepaired_lost_numbers).
    """
    repaired = modified_sentence
    unrepaired: List[str] = []
    for num in lost_numbers:
        num_str = str(num).strip()
        if not num_str:
            continue
        # Skip if already present (exact string or same digit sequence)
        if num_str in repaired:
            continue
        digits = re.sub(r"[^\d]", "", num_str)
        if digits and digits in re.sub(r"[^\d]", "", repaired):
            continue

        # Find num_str as a substring in original_sentence to get neighbour context
        orig_lower = original_sentence.lower()
        num_lower  = num_str.lower()
        idx = orig_lower.find(num_lower)
        left_anchor = right_anchor = ""
        if idx >= 0:
            before = original_sentence[:idx].rstrip()
            after  = original_sentence[idx + len(num_str):].lstrip()
            left_tokens  = re.findall(r"\S+", before)
            right_tokens = re.findall(r"\S+", after)
            if left_tokens:
                left_anchor = left_tokens[-1].strip(",.;:!?()")
            if right_tokens:
                right_anchor = right_tokens[0].strip(",.;:!?()")

        inserted = False
        if left_anchor:
            m = re.search(re.escape(left_anchor), repaired, re.IGNORECASE)
            if m:
                pos = m.end()
                repaired = repaired[:pos] + " " + num_str + repaired[pos:]
                inserted = True

        if not inserted and right_anchor:
            m = re.search(re.escape(right_anchor), repaired, re.IGNORECASE)
            if m:
                pos = m.start()
                repaired = repaired[:pos] + num_str + " " + repaired[pos:]
                inserted = True

        if not inserted:
            unrepaired.append(num_str)
            print(f"[SSE][REPAIR] no anchor found for '{num_str}'; left as unrepaired (sentence unchanged)")

    return repaired, unrepaired


# ─────────────────────────────────────────────────────────
# Agent drift monitoring (Fase A — audit only, no retry / no rejection)
# ─────────────────────────────────────────────────────────
# Strict thresholds: NM is expected to touch ONLY the number.
NM_DRIFT_WORD_DELTA_THRESHOLD = 0.20
NM_DRIFT_TOKEN_OVERLAP_THRESHOLD = 0.70


def _count_numbers_in(text: str) -> int:
    """Count numeric tokens in a sentence."""
    if not text:
        return 0
    return len(re.findall(r"\d[\d,.’]*", text))


def _word_metrics(original: str, modified: str) -> Tuple[float, float]:
    """Return (word_delta_pct, token_overlap) between two sentences."""
    orig_tokens = re.findall(r"\w+", (original or "").lower())
    mod_tokens = re.findall(r"\w+", (modified or "").lower())
    n_orig = len(orig_tokens)
    if n_orig == 0:
        return 0.0, 1.0
    delta_pct = abs(len(mod_tokens) - n_orig) / float(n_orig)
    orig_set = set(orig_tokens)
    if not orig_set:
        overlap = 1.0
    else:
        overlap = len(orig_set & set(mod_tokens)) / float(len(orig_set))
    return float(delta_pct), float(overlap)


def _compute_nm_drift_violations(
    target_number: str,
    original_sentence: str,
    modified_sentence: str,
) -> List[Dict[str, Any]]:
    """Return list of NM drift violations. Empty if NM behaved correctly."""
    violations: List[Dict[str, Any]] = []
    orig = original_sentence or ""
    mod = modified_sentence or ""
    tgt = (target_number or "").strip()

    # 1. target_number still present → NM did not actually modify
    if tgt and tgt in mod:
        violations.append({
            "violation_type": "nm_target_not_removed",
            "details": {"target": tgt},
        })

    # 2. no numbers left in modified sentence (target was silently deleted)
    if _count_numbers_in(mod) == 0:
        violations.append({
            "violation_type": "nm_no_numbers_left",
            "details": {"target": tgt},
        })

    # 3 + 4. structural drift on word count / overlap
    delta_pct, overlap = _word_metrics(orig, mod)
    if delta_pct > NM_DRIFT_WORD_DELTA_THRESHOLD:
        violations.append({
            "violation_type": "nm_high_word_delta",
            "details": {"word_delta_pct": round(delta_pct, 4),
                        "threshold": NM_DRIFT_WORD_DELTA_THRESHOLD},
        })
    if overlap < NM_DRIFT_TOKEN_OVERLAP_THRESHOLD:
        violations.append({
            "violation_type": "nm_low_token_overlap",
            "details": {"token_overlap": round(overlap, 4),
                        "threshold": NM_DRIFT_TOKEN_OVERLAP_THRESHOLD},
        })
    return violations


def _compute_sse_drift_violations(
    original_sentence: str,
    modified_sentence: str,
    lost_numbers_before_repair: List[str],
    renamed_numbers: List[str],
) -> List[Dict[str, Any]]:
    """Return list of SSE territorial violations on numbers.

    `lost_numbers_before_repair` is the list of numbers SSE *dropped* before the
    parenthetical repair logic (now disabled) had a chance to anchor them. The
    drop itself is a territorial violation (SSE was supposed to preserve numbers).
    `renamed_numbers` lists numbers whose VALUE was changed by SSE — that is
    NM's territory.
    """
    violations: List[Dict[str, Any]] = []
    for num in (lost_numbers_before_repair or []):
        violations.append({
            "violation_type": "sse_dropped_number",
            "details": {"number": str(num)},
        })
    for num in (renamed_numbers or []):
        violations.append({
            "violation_type": "sse_renamed_number",
            "details": {"number": str(num)},
        })
    return violations


def _append_agent_drift_log(
    drift_log_path: str,
    article_id: str,
    round_idx: int,
    agent: str,
    violations: List[Dict[str, Any]],
    target: str,
    original_sentence: str,
    modified_sentence: str,
) -> None:
    """Append one row per violation to the per-round agent_drift_log.csv."""
    if not violations:
        return
    delta_pct, overlap = _word_metrics(original_sentence, modified_sentence)
    cols = ["ID", "round_idx", "agent", "violation_type", "target",
            "original_sentence", "modified_sentence",
            "word_delta_pct", "token_overlap", "details"]
    rows = []
    for v in violations:
        rows.append({
            "ID": str(article_id),
            "round_idx": int(round_idx) if round_idx is not None else -1,
            "agent": str(agent),
            "violation_type": str(v.get("violation_type", "")),
            "target": str(target or ""),
            "original_sentence": str(original_sentence or ""),
            "modified_sentence": str(modified_sentence or ""),
            "word_delta_pct": round(delta_pct, 4),
            "token_overlap": round(overlap, 4),
            "details": json.dumps(v.get("details", {}), ensure_ascii=False),
        })
    df_new = pd.DataFrame(rows, columns=cols)
    try:
        if os.path.exists(drift_log_path) and os.path.getsize(drift_log_path) > 0:
            df_new.to_csv(drift_log_path, mode="a", header=False, index=False, encoding="utf-8")
        else:
            df_new.to_csv(drift_log_path, index=False, encoding="utf-8")
    except Exception as e:
        print(f"[DRIFT-LOG][WARN] could not append to {drift_log_path}: {e}")


def _append_sse_intermediate_exact(
    path: str,
    article_id: str,
    original_sentence: str,
    modified_sentence: str,
    numbers_in_original: Optional[List[str]] = None,
    numbers_in_modified: Optional[Dict[str, str]] = None,
    lost_numbers: Optional[List[str]] = None,
    renamed_numbers: Optional[List[str]] = None,
):
    cols = ["ID", "original_sentence", "modified_sentence",
            "numbers_in_original", "numbers_in_modified", "lost_numbers", "renamed_numbers"]
    row = {
        "ID": str(article_id),
        "original_sentence": str(original_sentence or ""),
        "modified_sentence": str(modified_sentence or ""),
        "numbers_in_original": json.dumps(numbers_in_original or [], ensure_ascii=False),
        "numbers_in_modified": json.dumps(numbers_in_modified or {}, ensure_ascii=False),
        "lost_numbers": json.dumps(lost_numbers or [], ensure_ascii=False),
        "renamed_numbers": json.dumps(renamed_numbers or [], ensure_ascii=False),
    }
    df_new = pd.DataFrame([row], columns=cols)

    if os.path.exists(path):
        try:
            df_old = pd.read_csv(path, encoding="utf-8")
        except Exception:
            df_old = pd.DataFrame(columns=cols)
    else:
        df_old = pd.DataFrame(columns=cols)

    for c in cols:
        if c not in df_old.columns:
            df_old[c] = ""
    df_old = df_old[cols].copy()

    df_all = pd.concat([df_old, df_new], ignore_index=True)
    df_all["ID"] = df_all["ID"].astype(str)
    df_all = df_all.drop_duplicates(subset=["ID", "original_sentence", "modified_sentence"], keep="first")

    _ensure_dir(os.path.dirname(path) or ".")
    df_all.to_csv(path, index=False, encoding="utf-8")

def _append_nm_intermediate_exact(
    path: str,
    article_id: str,
    target_number: str,
    original_sentence: str,
    modified_sentence: str,
    success: bool = True,
    reason: Optional[str] = None,
    modified_number: Optional[str] = None,
):
    """Persist one NM edit row (2026-05-20 revision).

    Adds two columns aligned with the metric contract:
      - original_number: equals target_number (= ORIGINAL value from round-0
        SA — fixed across rounds). metrics_runner.py:737-742 reads
        (original_number, modified_number) to compute NM_change_magnitude
        as cumulative edit-distance from the round-0 value.
      - modified_number: the value extracted from modified_sentence (passed
        explicitly by the caller, or auto-extracted here if None).
        Empty when the row represents a skip (lost_by_sse, anchor_lost,
        target_missing) → contributes 0 to change_magnitude.
    """
    if modified_number is None:
        modified_number = _extract_number_from_sentence(
            modified_sentence or "", str(target_number or "")
        )
    cols = ["ID", "target_number", "original_number", "modified_number",
            "original_sentence", "modified_sentence", "success", "reason"]
    row = {
        "ID": str(article_id),
        "target_number": str(target_number or ""),
        "original_number": str(target_number or ""),
        "modified_number": str(modified_number or ""),
        "original_sentence": str(original_sentence or ""),
        "modified_sentence": str(modified_sentence or ""),
        "success": bool(success),
        "reason": str(reason or ""),
    }
    df_new = pd.DataFrame([row], columns=cols)

    if os.path.exists(path):
        try:
            df_old = pd.read_csv(path, encoding="utf-8")
        except Exception:
            df_old = pd.DataFrame(columns=cols)
    else:
        df_old = pd.DataFrame(columns=cols)

    for c in cols:
        if c not in df_old.columns:
            df_old[c] = "" if c not in ("success",) else True
    df_old = df_old[cols].copy()

    df_all = pd.concat([df_old, df_new], ignore_index=True)
    df_all["ID"] = df_all["ID"].astype(str)
    df_all = df_all.drop_duplicates(subset=["ID", "target_number", "original_sentence", "modified_sentence"], keep="first")

    _ensure_dir(os.path.dirname(path) or ".")
    df_all.to_csv(path, index=False, encoding="utf-8")


def _extract_number_from_sentence(sentence: str, kn_current: str) -> str:
    """Find the first number token in sentence that differs from kn_current."""
    digits_cur = re.sub(r"[^\d]", "", str(kn_current))
    for tok in re.findall(
        r"\b\d[\d,.]*(?:\s*(?:months?|years?|days?|%|billion|million|trillion|thousand|hundred))?\b",
        sentence, flags=re.IGNORECASE,
    ):
        tok = tok.strip()
        if re.sub(r"[^\d]", "", tok) != digits_cur:
            return tok
    return kn_current


def _save_agent_snapshot(
    snap_dir: Optional[str],
    agent: str,
    article_id: str,
    round_idx: int,
    is_pivot: bool,
    edits: List[dict],
) -> None:
    """Writes a lightweight agent snapshot JSON to snap_dir/after_<agent>_<id>.json."""
    if not snap_dir:
        return
    _ensure_dir(snap_dir)
    payload = {
        "ID": str(article_id),
        "round": int(round_idx) if round_idx != -999 else 0,
        "agent": agent,
        "is_pivot": bool(is_pivot),
        "edits": edits,
    }
    path = os.path.join(snap_dir, f"after_{agent.lower()}_{article_id}.json")
    try:
        with open(path, "w", encoding="utf-8") as _f:
            json.dump(payload, _f, ensure_ascii=False, indent=2)
    except Exception as _e:
        print(f"[SNAPSHOT][WARN] Failed to write {path}: {_e}")


def _load_agent_snapshot(
    snaps_base_dir: Optional[str],
    agent: str,
    article_id: str,
    search_before_round: int,
) -> Optional[dict]:
    """Search backward from search_before_round-1 to 0 for the latest snapshot."""
    if not snaps_base_dir:
        return None
    for r in range(search_before_round - 1, -1, -1):
        path = os.path.join(
            snaps_base_dir, f"round_{r:03d}", f"after_{agent.lower()}_{article_id}.json"
        )
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as _f:
                    return json.load(_f)
            except Exception:
                continue
    return None


# §10 — similarity model singleton
_SIM_MODEL = None


def _get_sim_model():
    global _SIM_MODEL
    if _SIM_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SIM_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as _e:
            _warn("REALIGN_UNAVAILABLE", "SentenceTransformer unavailable", error=str(_e))
            _SIM_MODEL = False
    return _SIM_MODEL if _SIM_MODEL is not False else None


def _find_best_match_in_text(query_sentence: str, text: str, model=None) -> tuple:
    """Return (best_sentence, score). Falls back to (query_sentence, 0.0) on failure."""
    if model is None:
        model = _get_sim_model()
    if model is None or not query_sentence or not text:
        return query_sentence, 0.0
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sentences:
        return query_sentence, 0.0
    try:
        import torch
        q_emb = model.encode([query_sentence], convert_to_tensor=True)
        s_embs = model.encode(sentences, convert_to_tensor=True)
        cos_scores = torch.nn.functional.cosine_similarity(q_emb, s_embs)
        best_idx = int(cos_scores.argmax())
        score = float(cos_scores[best_idx])
    except Exception as _e:
        _warn("REALIGN_FAILED", "similarity computation failed", error=str(_e))
        return query_sentence, 0.0
    if score < 0.5:
        _warn("REALIGN_LOW", "low similarity score",
              similarity=score, query=query_sentence[:60])
        return query_sentence, score
    return sentences[best_idx], score


def _save_article_snapshot(art_snap_dir: Optional[str], article_id, text: str) -> None:
    if not art_snap_dir or not text:
        return
    _ensure_dir(art_snap_dir)
    path = os.path.join(art_snap_dir, f"article_{article_id}.txt")
    try:
        with open(path, "w", encoding="utf-8") as _f:
            _f.write(text)
    except Exception as _e:
        print(f"[SNAPSHOT][WARN] Failed to write article snapshot {path}: {_e}")


def _load_article_snapshot(art_snaps_base: Optional[str], article_id, search_before_round: int) -> str:
    """Return article text from most recent snapshot before search_before_round, or ''."""
    if not art_snaps_base:
        return ""
    for r in range(search_before_round - 1, -1, -1):
        path = os.path.join(art_snaps_base, f"round_{r:03d}", f"article_{article_id}.txt")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as _f:
                    return _f.read()
            except Exception:
                continue
    return ""


def _abort_all_keys(
    where: str,
    df_skips: Optional[pd.DataFrame],
    skips_path: Optional[str],
    checkpoint_path: Optional[str] = None,
    extra: Optional[Dict[str, object]] = None,
    # crash-safe extras
    df_reuse: Optional[pd.DataFrame] = None,
    reuse_path: Optional[str] = None,
    df_roundres: Optional[pd.DataFrame] = None,
    round_results_path: Optional[str] = None,
    panel_path: Optional[str] = None,
    round_idx_for_panel: Optional[int] = None,
    panel_target_ids: Optional[List[str]] = None,
    gen_csv_path: Optional[str] = None,
):
    try:
        if df_skips is not None and skips_path:
            df_skips.loc[len(df_skips)] = ["ALL", "GLOBAL", f"ALL_KEYS_EXHAUSTED hard-stop at {where}"[:4000]]
            _atomic_write_df_csv(df_skips, skips_path)
    except Exception:
        pass

    try:
        if df_reuse is not None and reuse_path:
            _atomic_write_df_csv(df_reuse, reuse_path)
    except Exception:
        pass
    try:
        if df_roundres is not None and round_results_path:
            _atomic_write_df_csv(df_roundres, round_results_path)
    except Exception:
        pass
    try:
        if panel_path and (round_idx_for_panel is not None) and panel_target_ids is not None and gen_csv_path:
            skipped_ids = _skipped_ids_from_df(df_skips) if df_skips is not None else set()
            _persist_panel_atomic(
                panel_path=panel_path,
                round_idx_for_panel=int(round_idx_for_panel),
                target_ids=[str(x) for x in panel_target_ids],
                gen_csv_path=gen_csv_path,
                skipped_ids_round=skipped_ids,
            )
    except Exception:
        pass

    try:
        if checkpoint_path:
            payload = {"timestamp": _now_iso(), "where": where, "note": "TPD/ALL_KEYS_EXHAUSTED hard-stop"}
            if extra:
                payload.update(extra)
            _ensure_dir(os.path.dirname(checkpoint_path) or ".")
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    raise RuntimeError(f"[PIPE][HARD-STOP] ALL_KEYS_EXHAUSTED at {where}. Pipeline terminated.")

def _hard_stop_if_tpd(*args, **kwargs):
    return


def _chat_with_retries(agent_name: str, agent_obj, message: str, retries: int = 3, backoff_base: float = 1.5, verbose: bool = True):
    last_err = None
    schema = get_agent_json_schema(agent_obj)
    for attempt in range(retries):
        try:
            if verbose:
                print(f"[LLM] {agent_name} attempt {attempt+1}/{retries}")
            return call_ollama_structured(agent_obj, message=message, schema=schema)
        except Exception as e:
            last_err = e
            time.sleep(backoff_base ** attempt)
    # After all retries exhausted: log and return None so the caller can skip the ID
    print(f"[LLM][WARN] {agent_name} failed after {retries} attempts — ID will be skipped. Last error: {last_err}")
    return None



def _strip_wrappers(s: str) -> str:
    if not s:
        return ""
    x = str(s).strip()
    x = x.replace("<JSON>", "").replace("</JSON>", "").strip()
    if x.startswith("```"):
        x = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", x).strip()
        x = x.strip("`").strip()
    return x.strip()

def _extract_first_balanced_object_ignore_strings(s: str) -> str:
    if not s:
        return ""
    s = str(s)
    start = s.find("{")
    if start == -1:
        return ""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        else:
            if ch == '"':
                in_string = True
                continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1].strip()
    return ""

def _extract_json_block(raw: str) -> str:
    if not raw:
        return ""
    cleaned = _strip_wrappers(raw)
    candidate = _extract_first_balanced_object_ignore_strings(cleaned)
    return candidate.strip() if candidate else ""

def _try_parse_json_any(resp: Any) -> Optional[dict]:
    if resp is None:
        return None
    if isinstance(resp, dict):
        return resp
    s = _extract_json_block(str(resp))
    if not s:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None

def _basic_json_sanitize(s: str) -> str:
    if not s:
        return ""
    x = str(s)
    x = x.replace("“", '"').replace("”", '"').replace("„", '"')
    x = x.replace("’", "'").replace("‘", "'")
    x = re.sub(r",\s*([}\]])", r"\1", x)
    x = re.sub(r"\bNone\b", "null", x)
    x = re.sub(r"\bTrue\b", "true", x)
    x = re.sub(r"\bFalse\b", "false", x)
    x = x.lstrip("\ufeff")
    return x.strip()

def _try_literal_then_dump(raw: str) -> Optional[str]:
    try:
        obj = ast.literal_eval(raw)
        if isinstance(obj, dict):
            dumped = json.dumps(obj, ensure_ascii=False)
            json.loads(dumped)
            return dumped
    except Exception:
        return None
    return None

def _local_repair_json(raw_answer: str, attempts: int = 6) -> Optional[str]:
    raw = _extract_json_block(raw_answer)
    if not raw:
        return None
    candidates = []
    c1 = _basic_json_sanitize(raw)
    candidates.append(c1)
    try:
        from json_repair import repair_json
        candidates.append(repair_json(c1))
    except Exception:
        pass
    c3 = _try_literal_then_dump(raw)
    if c3:
        candidates.append(c3)
    c4 = _basic_json_sanitize(raw.replace("\r\n", "\n").replace("\r", "\n"))
    c4 = re.sub(r"\t", " ", c4)
    candidates.append(c4)
    try:
        from json_repair import repair_json
        candidates.append(repair_json(c4))
    except Exception:
        pass

    tried = 0
    for cand in candidates:
        if not cand:
            continue
        tried += 1
        if tried > attempts:
            break
        try:
            json.loads(cand)
            return cand
        except Exception:
            continue
    return None

def _parse_with_local_repair(raw_answer: Any) -> Optional[dict]:
    # Unpack {"content": "..."} wrapper returned by call_ollama_structured.
    # Without this, _try_parse_json_any returns the wrapper dict as-is and
    # all downstream _coerce_* functions fail (no expected keys present).
    if isinstance(raw_answer, dict) and set(raw_answer.keys()) == {"content"}:
        raw_answer = raw_answer.get("content", "")
    obj = _try_parse_json_any(raw_answer)
    if isinstance(obj, dict):
        return obj
    repaired = _local_repair_json(str(raw_answer), attempts=6)
    if repaired:
        try:
            return json.loads(repaired)
        except Exception:
            pass
    return None


def _sanitize_numbers(nums: list) -> list:
    """Filter to digit-bearing strings and strip literal surrounding quote chars.

    Some LLMs (notably when the SA prompt instructs "enclose each number in
    double quotes") emit values like '"4.1"' where the quote characters are
    INSIDE the JSON string value, not just the JSON syntactic wrapper. This
    propagates downstream and breaks `n2s_map.get('"4.1"')` lookups. We strip
    those literal quotes here so the rest of the pipeline sees clean numbers.
    """
    if not isinstance(nums, list):
        return []
    out = []
    for raw in nums:
        if not isinstance(raw, str):
            continue
        s = raw.strip().strip('"').strip("'").strip()
        if s and re.search(r"\d", s):
            out.append(s)
    return out

# Bug-5 fix: known placeholder values copied verbatim from the SA system-prompt examples.
# Any number matching these is rejected regardless of whether it appears in the text.
_SA_PROMPT_EXAMPLE_NUMBERS: set = {
    "2024", "3.14", "1.5 billion", "0746", "10,000",
}

def _sanitize_numbers_strict(nums: list, source_text: str) -> list:
    """
    Bug-5 fix: filter out numbers that are not present in source_text and
    remove any value that is a known placeholder from the SA prompt examples.

    IMPORTANT: the blocklist is applied ONLY if the number is NOT found in the
    source text.  A value like "2024" or "10,000" that genuinely appears in the
    article is always accepted; only a value from the blocklist that is absent
    from the text is rejected as a hallucinated prompt-example.
    Falls back to _sanitize_numbers behaviour if source_text is empty.
    """
    if not isinstance(nums, list):
        return []
    if not source_text:
        return _sanitize_numbers(nums)

    text_lower = source_text.lower()
    # Build the set of "numeric token phrases" from text: a run of digits possibly
    # followed by an optional unit word. Used for strict membership check.
    # Example: "magnitude 6.2 earthquake, 3 towns, 15 people" → {"6.2", "3", "15"}.
    text_numeric_tokens = set()
    for m in re.finditer(r"\d[\d,.’]*(?:\s*[A-Za-z%°][A-Za-z]*)?", source_text):
        tok = m.group(0).strip()
        if tok:
            text_numeric_tokens.add(tok)
            # Also normalize digit-only stem (e.g. "6.2" from "6.2 percent")
            stem = re.match(r"\d[\d,.’]*", tok)
            if stem:
                text_numeric_tokens.add(stem.group(0))
    text_numeric_tokens_lower = {t.lower() for t in text_numeric_tokens}

    validated = []
    for raw in nums:
        # Strip literal surrounding quote chars (`"4.1"` → `4.1`).
        s = str(raw).strip().strip('"').strip("'").strip()
        if not s:
            continue
        # Keep only if it contains digits
        if not re.search(r"\d", s):
            continue

        s_lower = s.lower()

        # Strict word-boundary check: `s` must occur in text NOT adjacent to a
        # digit or dot. This blocks false positives like:
        #   "1"        matching inside "15"  ("1" preceded by space but followed by "5")
        #   "3 million" matching inside "2.3 million" ("3" preceded by ".")
        boundary_pat = r"(?<![\d.])" + re.escape(s) + r"(?![\d.])"
        in_text = re.search(boundary_pat, source_text, re.IGNORECASE) is not None

        # Fallback: strict numeric-token match (handles "1500" vs "1,500" or
        # "6.2 percent" vs "6.2"). Requires the digit-stem of `s` to be the
        # digit-stem of some numeric token in the source — NO substring match.
        if not in_text:
            digits_only = re.sub(r"[^\d]", "", s)
            if len(digits_only) >= 1:
                token_digit_stems = {re.sub(r"[^\d]", "", t) for t in text_numeric_tokens}
                if digits_only in token_digit_stems:
                    in_text = True

        if in_text:
            # Bugfix: store the CLEAN value (without literal surrounding quotes),
            # not the raw with-quotes form. Otherwise downstream lookups like
            # `n2s_map.get('"4.1"')` fail because the article text uses `4.1`.
            validated.append(s)
            continue

        # Number is NOT in the text: now apply blocklist to catch prompt-example
        # hallucinations.
        if s_lower in _SA_PROMPT_EXAMPLE_NUMBERS:
            _warn("SA_NUMBER_PLACEHOLDER", "prompt-example placeholder removed", value=s)
            continue

        # Not in text and not a known placeholder — discard as hallucinated.
        _warn("SA_NUMBER_DISCARDED", "number not found in text", value=s)
    return validated


def _build_n2s_from_text(key_numbers: list, text: str) -> Dict[str, str]:
    """Synthesize number_to_sentence_map by locating each number in `text`.

    Used to fill in entries the LLM omitted (very common with format="json" and
    Ollama < 0.3.0 where the schema cannot be enforced server-side). Returns a
    dict `{key_number: matched_sentence}` containing one entry per key_number
    found in the text; numbers not found are silently skipped (downstream code
    already handles missing keys).

    Matching strategy per number:
      1. Verbatim substring in any sentence.
      2. Digit-only substring (handles "1500" vs "1,500", "4.1" vs "4.1%").
    """
    if not text or not key_numbers:
        return {}
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sentences:
        return {}
    out: Dict[str, str] = {}
    for raw in key_numbers:
        if not isinstance(raw, str):
            continue
        num_clean = raw.strip().strip('"').strip("'").strip()
        if not num_clean:
            continue
        # Verbatim first
        matched = next((s for s in sentences if num_clean in s), None)
        # Digit-only fallback
        if matched is None:
            digits_num = re.sub(r"[^\d]", "", num_clean)
            if len(digits_num) >= 1:
                matched = next(
                    (s for s in sentences
                     if digits_num in re.sub(r"[^\d]", "", s)),
                    None,
                )
        if matched:
            out[raw] = matched
    return out


def _extract_numbers_from_text(text: str) -> list:
    """
    Bug-3 fix: regex fallback to re-extract numbers from a (possibly SSE-modified)
    text when the original SA key_numbers are no longer present.
    """
    if not text:
        return []
    pattern = (
        r"\b\d{1,3}(?:[,.\u2019]\d+)*"
        r"(?:\s*(?:billion|million|thousand|hundred|percent|"
        r"days?|months?|years?|hours?|minutes?|seconds?|"
        r"km|kg|lb|oz|ft|cm|m|%))?\b"
    )
    found = re.findall(pattern, text, re.IGNORECASE)
    seen: set = set()
    result = []
    for f in found:
        f = f.strip()
        if f not in seen and re.search(r"\d", f):
            seen.add(f)
            result.append(f)
    return result[:8]  # cap at 8 to avoid flooding NM


def _reconcile_key_numbers(original_key_numbers: list, current_text: str) -> list:
    """
    Bug-3 fix: after SSE has modified the article, some key_numbers from SA may
    no longer exist in the text (SSE changed the sentence surface form).
    Keep only those still present; if none survive, fall back to regex extraction.
    """
    if not current_text:
        return list(original_key_numbers)
    ct_lower = current_text.lower()
    surviving = []
    for kn in original_key_numbers:
        s = str(kn).strip().strip('"')
        if s.lower() in ct_lower:
            surviving.append(kn)
            continue
        digits = re.sub(r"[^0-9]", "", s)
        if len(digits) >= 2 and digits in re.sub(r"[^0-9]", "", current_text):
            surviving.append(kn)
    if surviving:
        return surviving
    # No original numbers survived SSE rewrites: extract fresh from current text
    fresh = _extract_numbers_from_text(current_text)
    if fresh:
        _warn("NM_KEY_NUMBERS_REEXTRACTED",
              "all original key_numbers lost after SSE; re-extracted from current text",
              count=len(fresh))
    return fresh

def _coerce_sa(obj: dict) -> Optional[Tuple[List[str], List[str], Dict[str, str]]]:
    if not isinstance(obj, dict):
        return None
    ks = obj.get("key_sentences", None)
    if not isinstance(ks, list):
        return None
    nums = obj.get("key_numbers", obj.get("numbers", []))
    if not isinstance(nums, list):
        nums = []
    ks = [str(x) for x in ks]
    nums = _sanitize_numbers([str(x) for x in nums])

    raw_n2s = obj.get("number_to_sentence_map", {})
    if not isinstance(raw_n2s, dict):
        raw_n2s = {}

    nums_set = set(nums)
    validated_map: Dict[str, str] = {}
    for raw_num, sentence in raw_n2s.items():
        if not isinstance(sentence, str):
            continue
        # Bugfix: strip literal quotes from key (LLMs sometimes wrap values in
        # extra `"` chars when the prompt says "enclose in double quotes").
        num = str(raw_num).strip().strip('"').strip("'").strip() if isinstance(raw_num, str) else str(raw_num)
        if num not in nums_set:
            continue
        digits_num = re.sub(r"[^\d]", "", num)
        digits_sent = re.sub(r"[^\d]", "", sentence)
        if digits_num and digits_num in digits_sent:
            validated_map[num] = sentence
        else:
            _warn("SA_NUMBER_MAP_MISMATCH", "number not in mapped sentence",
                  number=num, sentence=sentence[:60])

    return ks, nums, validated_map

def _coerce_sse_nm(obj: dict) -> Optional[Tuple[str, str]]:
    if not isinstance(obj, dict):
        return None
    if {"original_sentence", "modified_sentence"}.issubset(obj.keys()):
        o = str(obj.get("original_sentence", "") or "")
        m = str(obj.get("modified_sentence", "") or "")
        if not o.strip():
            return None
        if not m.strip():
            print("[COERCE][WARN] empty modified_sentence — falling back to original (no-op)")
            return o, o
        return o, m
    return None

NAM_TECHNIQUES = {
    "Presenting Irrelevant Data", "Misrepresentation of Position",
    "Whataboutism", "Causal Oversimplification", "Obfuscation",
    "Appeal to Authority", "Black-and-white Fallacy",
    "Loaded Language", "Exaggeration or Minimisation",
}

def _coerce_nam(obj: dict) -> Optional[Tuple[str, str]]:
    """Return (modified_article, technique_used) or None.

    `technique_used` is a canonical name from NAM_TECHNIQUES, or '' if missing/unknown
    (logged, never blocking).
    """
    if not isinstance(obj, dict):
        return None
    article = None
    if isinstance(obj.get("modified_article"), str) and obj["modified_article"].strip():
        article = obj["modified_article"]
    elif isinstance(obj.get("modified_text"), str) and obj["modified_text"].strip():
        article = obj["modified_text"]
    if article is None:
        return None

    tech = str(obj.get("technique_used", "") or "").strip()
    if tech and tech not in NAM_TECHNIQUES:
        # Case-insensitive / substring matching against canonical names
        norm = tech.lower()
        for canonical in NAM_TECHNIQUES:
            if canonical.lower() in norm or norm in canonical.lower():
                tech = canonical
                break
        else:
            print(f"[NaM][COERCE][WARN] unknown technique '{tech}' — clearing to ''")
            tech = ""
    return article, tech

def _coerce_te(obj: dict) -> Optional[str]:
    if not isinstance(obj, dict):
        return None
    if "modified_title" in obj and isinstance(obj["modified_title"], str) and obj["modified_title"].strip():
        return obj["modified_title"].strip()
    if "assigned_title" in obj and isinstance(obj["assigned_title"], str) and obj["assigned_title"].strip():
        return obj["assigned_title"].strip()
    return None


def _match_sentence_by_embedding(current_article: str, key_sentence: str) -> Optional[str]:
    _PIPELINE._ensure_nltk()
    sents = sent_tokenize(current_article or "")
    if not sents:
        return None
    emb_sents = _PIPELINE.embedder.encode(sents, convert_to_tensor=True)
    emb_key = _PIPELINE.embedder.encode([key_sentence], convert_to_tensor=True)[0]
    idx = int(util.cos_sim(emb_key, emb_sents).argmax())
    return sents[idx]

def _find_sentence_with_number(article_text: str, key_number: str) -> Optional[str]:
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    # Bug-3 fix: normalise the key_number by stripping wrapping quotes and dashes
    # so "90-day" matches both "90-day" and "90 day" surface forms.
    kn_raw = _norm(key_number).strip('"').strip("'")
    kn_norm = _norm(kn_raw)
    kn_nodash = re.sub(r"[-\s]", "", kn_norm).lower()

    _PIPELINE._ensure_nltk()
    nm_sents = sent_tokenize(article_text or "")

    # Pass 1: exact normalised match
    for sent in nm_sents:
        s_norm = _norm(sent)
        if kn_norm and kn_norm in s_norm:
            return sent

    # Pass 2: dash/space-insensitive match (handles "90-day" vs "90 day")
    for sent in nm_sents:
        s_nodash = re.sub(r"[-\s]", "", sent).lower()
        if kn_nodash and kn_nodash in s_nodash:
            return sent

    # Pass 3: digit-sequence-only match (last resort)
    digits = re.sub(r"[^\d]", "", kn_norm)
    if digits:
        for sent in nm_sents:
            if digits in re.sub(r"[^\d]", "", sent):
                return sent

    return None


def _normalize_agent_name(x: str) -> str:
    x = str(x or "").strip().upper()
    mapping = {
        "SA": "SA", "SEMANTICANALYZER": "SA", "SEMANTIC_ANALYZER": "SA",
        "SSE": "SSE", "SALIENTSENTENCEEDITOR": "SSE", "SALIENT_SENTENCE_EDITOR": "SSE",
        "NM": "NM", "NUMBERMODIFIER": "NM", "NUMBER_MODIFIER": "NM",
        "NAM": "NaM", "NARRATIVEMODIFIER": "NaM", "NARRATIVE_MODIFIER": "NaM",
        "TE": "TE", "TITLEEDITOR": "TE", "TITLE_EDITOR": "TE",
    }
    return mapping.get(x, x)

def _infer_pivot_agent(row_ctrl: pd.Series) -> str:
    p = _normalize_agent_name(row_ctrl.get("pivot_agent", ""))
    if p in ("SA", "SSE", "NM", "NaM", "TE"):
        return p
    for col, a in [
        ("state_SA", "SA"),
        ("state_SSE", "SSE"),
        ("state_NM", "NM"),
        ("state_NaM", "NaM"),
        ("state_TE", "TE"),
    ]:
        if col in row_ctrl.index:
            v = str(row_ctrl.get(col, "KEEP") or "KEEP").strip().upper()
            if v != "KEEP":
                return a
    return "NaM"

def _compute_run_from_pivot(pivot: str) -> Dict[str, bool]:
    """
    Cascade logic: the pivot agent and ALL agents that come after it in the chain
    SA→SSE→NM→NaM→TE receive a real LLM call. Agents before the pivot are skipped.
    No mechanical reapply, no coherence-only calls.
    """
    pivot = _normalize_agent_name(pivot)
    _CHAIN = ["SA", "SSE", "NM", "NaM", "TE"]
    if pivot not in _CHAIN:
        pivot = "NaM"  # fallback
    idx = _CHAIN.index(pivot)
    return dict(
        run_SA=  (_CHAIN.index("SA")  >= idx),
        run_SSE= (_CHAIN.index("SSE") >= idx),
        run_NM=  (_CHAIN.index("NM")  >= idx),
        run_NaM= (_CHAIN.index("NaM") >= idx),
        run_TE=  (_CHAIN.index("TE")  >= idx),
    )

def _extract_objective(row_ctrl: pd.Series) -> str:
    obj = str(row_ctrl.get("objective", "") or "").strip()
    if obj:
        return obj
    g = str(row_ctrl.get("guide_en", "") or "").strip()
    if not g:
        return ""
    m = re.search(r"(OBJECTIVE|Objective)\s*:\s*(.+)", g)
    if m:
        tail = m.group(2).strip()
        tail = tail.splitlines()[0].strip()
        return tail
    return ""



def _extract_trajectory_feedback(row_ctrl: pd.Series) -> Dict[str, Any]:
    raw = row_ctrl.get("trajectory_feedback_json", "")
    if raw is not None:
        text = str(raw).strip()
        if text and text.lower() not in ("nan", "none"):
            try:
                obj = json.loads(text)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

    # CSV fallback if the compact JSON column is unavailable.
    feedback = {
        "feature": str(row_ctrl.get("chosen_feature", "") or ""),
        "current_value": row_ctrl.get("trajectory_current_value", None),
        "baseline_median": row_ctrl.get("trajectory_baseline_median", None),
        "previous_delta": row_ctrl.get("trajectory_previous_delta", None),
        "cumulative_delta": row_ctrl.get("trajectory_cumulative_delta", None),
        "no_effect_flag": row_ctrl.get("trajectory_no_effect_flag", False),
    }

    def _clean(v):
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass
        if isinstance(v, str):
            sv = v.strip()
            if not sv or sv.lower() in ("nan", "none", "null"):
                return None
            if sv.lower() in ("true", "false"):
                return sv.lower() == "true"
            try:
                return float(sv)
            except Exception:
                return sv
        return v

    return {k: _clean(v) for k, v in feedback.items()}

def _load_orch_control_df(prev_orch_dir: str, detector_name: str) -> Optional[pd.DataFrame]:
    """Load orchestrator decisions used to control round>0 generation.

    Different callers historically passed different roots:
      - the previous round directory (e.g., Output/rounds/round_000)
      - the orchestrator folder (e.g., .../orchestrator)
      - the detector-specific orchestrator folder (e.g., .../orchestrator/<detector>)

    To make the pipeline robust, try all common locations.
    """

    candidates = [
        # Caller passes prev_round_dir
        os.path.join(prev_orch_dir, "orchestrator", detector_name, "orch_decisions_with_reasons.csv"),
        # Caller passes .../orchestrator
        os.path.join(prev_orch_dir, detector_name, "orch_decisions_with_reasons.csv"),
        # Caller passes .../orchestrator/<detector>
        os.path.join(prev_orch_dir, "orch_decisions_with_reasons.csv"),
    ]

    decisions_path = None
    for p in candidates:
        if p and os.path.exists(p):
            decisions_path = p
            break

    if decisions_path is None:
        print("[PIPE][CTRL] Missing orchestrator decisions -> control disabled. Tried:\n- " + "\n- ".join(candidates))
        return None
    try:
        df_dec = pd.read_csv(decisions_path, encoding="utf-8")
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        print(f"[PIPE][CTRL] Empty or malformed orch decisions file: {decisions_path} -> control disabled.")
        return None
    if "ID" not in df_dec.columns:
        print("[PIPE][CTRL] Missing ID column in orchestrator decisions -> control disabled.")
        return None
    df_dec["ID"] = df_dec["ID"].astype(str)
    df_dec = df_dec.set_index("ID")
    text_cols = ("pivot_agent", "objective", "chosen_feature", "guide_en", "trajectory_feedback_json")
    for col in text_cols:
        if col not in df_dec.columns:
            df_dec[col] = ""
        df_dec[col] = df_dec[col].fillna("").astype(str)

    for col in ("trajectory_current_value", "trajectory_baseline_median",
                "trajectory_previous_delta", "trajectory_cumulative_delta",
                "trajectory_no_effect_flag"):
        if col not in df_dec.columns:
            df_dec[col] = pd.NA
    print(f"[PIPE][CTRL] Loaded orchestrator decisions: {decisions_path} (IDs={len(df_dec)})")
    return df_dec


def _anchors_master_path(base_out_dir: str, detector_name: Optional[str]) -> str:
    out_root = _find_output_root(base_out_dir) or os.path.abspath(os.path.join(base_out_dir, "..", "..", ".."))
    det = (detector_name or "global").strip() or "global"
    p = os.path.join(out_root, "anchors_master", det, "anchors_master.csv")
    _ensure_dir(os.path.dirname(p))
    return p

def _load_anchors_master(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=["ID", "key_sentences", "key_numbers", "updated_round", "timestamp"])
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.DataFrame(columns=["ID", "key_sentences", "key_numbers", "updated_round", "timestamp"])
    if "ID" not in df.columns:
        return pd.DataFrame(columns=["ID", "key_sentences", "key_numbers", "updated_round", "timestamp"])
    df["ID"] = df["ID"].astype(str)
    for c in ["key_sentences", "key_numbers"]:
        if c not in df.columns:
            df[c] = "[]"
        df[c] = df[c].fillna("[]").astype(str)
    if "updated_round" not in df.columns:
        df["updated_round"] = -1
    if "timestamp" not in df.columns:
        df["timestamp"] = ""
    return df.drop_duplicates(subset=["ID"], keep="last").reset_index(drop=True)

def _safe_list_from_cell(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(v) for v in arr]
        except Exception:
            pass
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [str(z) for z in v]
    except Exception:
        pass
    return [t.strip() for t in s.split(",") if t.strip()]

def _atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)

def _update_anchor_master(master_path: str, article_id: str, key_sentences: List[str], key_numbers: List[str], round_idx: int) -> None:
    dfm = _load_anchors_master(master_path)
    row = {
        "ID": str(article_id),
        "key_sentences": json.dumps(list(key_sentences), ensure_ascii=False),
        "key_numbers": json.dumps(list(key_numbers), ensure_ascii=False),
        "updated_round": int(round_idx),
        "timestamp": _now_iso(),
    }
    dfm = dfm[dfm["ID"].astype(str) != str(article_id)]
    dfm = pd.concat([dfm, pd.DataFrame([row])], ignore_index=True)
    _atomic_write_csv(dfm, master_path)

def _get_anchors_from_master(master_path: str, article_id: str) -> Optional[Tuple[List[str], List[str]]]:
    dfm = _load_anchors_master(master_path)
    m = dfm[dfm["ID"].astype(str) == str(article_id)]
    if m.empty:
        return None
    r = m.iloc[-1]
    ks = _safe_list_from_cell(r.get("key_sentences", "[]"))
    kn = _safe_list_from_cell(r.get("key_numbers", "[]"))
    ks = [str(x) for x in ks]
    kn = _sanitize_numbers([str(x) for x in kn])
    return ks, kn


def _load_prev_sse_modified_sentences(prev_generation_dir: str, article_id: str) -> List[str]:
    if not prev_generation_dir:
        return []
    p = os.path.join(prev_generation_dir, "SalientSentenceEditor.csv")
    if not os.path.exists(p):
        return []
    try:
        df = pd.read_csv(p, encoding="utf-8")
    except Exception:
        return []
    if not {"ID", "modified_sentence"}.issubset(set(df.columns)):
        return []
    df["ID"] = df["ID"].astype(str)
    df = df[df["ID"] == str(article_id)].copy()
    if df.empty:
        return []
    out = [str(x).strip() for x in df["modified_sentence"].fillna("").astype(str).tolist() if str(x).strip()]
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq

def _load_prev_nm_modified_sentences(prev_generation_dir: str, article_id: str) -> List[str]:
    if not prev_generation_dir:
        return []
    p = os.path.join(prev_generation_dir, "NumberModifier.csv")
    if not os.path.exists(p):
        return []
    try:
        df = pd.read_csv(p, encoding="utf-8")
    except Exception:
        return []
    if not {"ID", "modified_sentence"}.issubset(set(df.columns)):
        return []
    df["ID"] = df["ID"].astype(str)
    df = df[df["ID"] == str(article_id)].copy()
    if df.empty:
        return []
    out = [str(x).strip() for x in df["modified_sentence"].fillna("").astype(str).tolist() if str(x).strip()]
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


def _load_prev_sse_rows(prev_generation_dir: str, article_id: str) -> pd.DataFrame:
    _extra_cols = ["numbers_in_original", "numbers_in_modified", "lost_numbers", "renamed_numbers"]
    _base_cols = ["ID", "original_sentence", "modified_sentence"]
    _all_cols = _base_cols + _extra_cols
    p = os.path.join(prev_generation_dir or "", "SalientSentenceEditor.csv")
    if not prev_generation_dir or not os.path.exists(p):
        return pd.DataFrame(columns=_all_cols)
    try:
        df = pd.read_csv(p, encoding="utf-8")
    except Exception:
        return pd.DataFrame(columns=_all_cols)
    if not {"ID", "original_sentence", "modified_sentence"}.issubset(set(df.columns)):
        return pd.DataFrame(columns=_all_cols)
    _col_defaults = {
        "numbers_in_original": "[]",
        "numbers_in_modified": "{}",
        "lost_numbers": "[]",
        "renamed_numbers": "[]",
    }
    for c in _extra_cols:
        if c not in df.columns:
            df[c] = _col_defaults[c]
    df["ID"] = df["ID"].astype(str)
    df = df[df["ID"] == str(article_id)].copy()
    for c in ["original_sentence", "modified_sentence"]:
        df[c] = df[c].fillna("").astype(str)
    for c in _extra_cols:
        df[c] = df[c].fillna(_col_defaults[c]).astype(str)
    df = df[(df["original_sentence"].str.strip() != "") & (df["modified_sentence"].str.strip() != "")]
    return df[_all_cols].reset_index(drop=True)

def _load_prev_nm_rows(prev_generation_dir: str, article_id: str) -> pd.DataFrame:
    out_cols = ["ID", "target_number", "original_number", "modified_number",
                "original_sentence", "modified_sentence"]
    p = os.path.join(prev_generation_dir or "", "NumberModifier.csv")
    if not prev_generation_dir or not os.path.exists(p):
        return pd.DataFrame(columns=out_cols)
    try:
        df = pd.read_csv(p, encoding="utf-8")
    except Exception:
        return pd.DataFrame(columns=out_cols)
    need = {"ID", "target_number", "original_sentence", "modified_sentence"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame(columns=out_cols)
    df["ID"] = df["ID"].astype(str)
    df = df[df["ID"] == str(article_id)].copy()
    # Backfill the two 2026-05-20 columns for legacy CSVs that predate this revision.
    if "original_number" not in df.columns:
        df["original_number"] = df["target_number"].astype(str)
    if "modified_number" not in df.columns:
        df["modified_number"] = df.apply(
            lambda r: _extract_number_from_sentence(
                str(r.get("modified_sentence", "") or ""),
                str(r.get("target_number", "") or ""),
            ),
            axis=1,
        )
    for c in ["target_number", "original_number", "modified_number",
              "original_sentence", "modified_sentence"]:
        df[c] = df[c].fillna("").astype(str)
    df = df[(df["original_sentence"].str.strip() != "") & (df["modified_sentence"].str.strip() != "")]
    return df[out_cols].reset_index(drop=True)

def _load_prev_sa_output(prev_generation_dir: str, article_id: str) -> Tuple[List[str], List[str], Dict[str, str]]:
    p = os.path.join(prev_generation_dir or "", "SemanticAnalyzer.csv")
    if not prev_generation_dir or not os.path.exists(p):
        return [], [], {}
    try:
        df = pd.read_csv(p, encoding="utf-8")
    except Exception:
        return [], [], {}
    if "ID" not in df.columns:
        return [], [], {}
    df["ID"] = df["ID"].astype(str)
    df = df[df["ID"] == str(article_id)].copy()
    if df.empty:
        return [], [], {}
    if "round_idx" in df.columns:
        try:
            df["round_idx"] = pd.to_numeric(df["round_idx"], errors="coerce")
            df = df.sort_values(["round_idx"], kind="stable")
        except Exception:
            pass
    row = df.iloc[-1]
    ks = _safe_list_from_cell(row.get("key_sentences", "[]"))
    kn = _sanitize_numbers(_safe_list_from_cell(row.get("key_numbers", "[]")))
    if "number_to_sentence_map" in df.columns:
        try:
            n2s_map = json.loads(row.get("number_to_sentence_map") or "{}")
            if not isinstance(n2s_map, dict):
                n2s_map = {}
        except Exception:
            n2s_map = {}
    else:
        n2s_map = {}
    return [str(x) for x in ks], [str(x) for x in kn], n2s_map


def _load_prev_nam_article(prev_generation_dir: str, article_id: str) -> str:
    p = os.path.join(prev_generation_dir or "", "NarrativeModifier.csv")
    if not prev_generation_dir or not os.path.exists(p):
        return ""
    try:
        df = pd.read_csv(p, encoding="utf-8")
    except Exception:
        return ""
    if "ID" not in df.columns or "modified_article" not in df.columns:
        return ""
    df["ID"] = df["ID"].astype(str)
    df = df[df["ID"] == str(article_id)].copy()
    if df.empty:
        return ""
    if "round_idx" in df.columns:
        try:
            df["round_idx"] = pd.to_numeric(df["round_idx"], errors="coerce")
            df = df.sort_values(["round_idx"], kind="stable")
        except Exception:
            pass
    return str(df.iloc[-1].get("modified_article", "") or "")


def _replace_sentence_variant(text: str, candidates: List[str], replacement: str) -> str:
    base = text or ""
    if not _norm_text_loose(replacement):
        return base
    for cand in candidates or []:
        cand = str(cand or "")
        if _norm_text_loose(cand) and cand in base:
            return base.replace(cand, replacement, 1)
    return base



def _infer_round0_generation_dir(base_out_dir: str) -> Optional[str]:
    out_root = _find_output_root(base_out_dir)
    if not out_root or not os.path.exists(out_root):
        return None

    candidates = []
    for name in ("round_0", "round0", "Round_0", "Round0"):
        p = os.path.join(out_root, name)
        if os.path.isdir(p):
            candidates.append(p)

    if not candidates:
        try:
            for entry in os.listdir(out_root):
                p = os.path.join(out_root, entry)
                if os.path.isdir(p) and re.search(r"round[_\- ]?0", entry, flags=re.IGNORECASE):
                    candidates.append(p)
        except Exception:
            pass

    for c in candidates:
        try:
            for root, dirs, files in os.walk(c):
                if "SalientSentenceEditor.csv" in files and "NumberModifier.csv" in files:
                    return root
                rel = os.path.relpath(root, c)
                if rel.count(os.sep) >= 6:
                    dirs[:] = []
        except Exception:
            continue
    return None

def _load_round0_sse_original_sentences(round0_gen_dir: Optional[str], article_id: str) -> List[str]:
    if not round0_gen_dir:
        return []
    p = os.path.join(round0_gen_dir, "SalientSentenceEditor.csv")
    if not os.path.exists(p):
        return []
    try:
        df = pd.read_csv(p, encoding="utf-8")
    except Exception:
        return []
    if not {"ID", "original_sentence"}.issubset(set(df.columns)):
        return []
    df["ID"] = df["ID"].astype(str)
    df = df[df["ID"] == str(article_id)].copy()
    if df.empty:
        return []
    out = [str(x).strip() for x in df["original_sentence"].fillna("").astype(str).tolist() if str(x).strip()]
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq

def _load_round0_nm_rows(round0_gen_dir: Optional[str], article_id: str) -> pd.DataFrame:
    if not round0_gen_dir:
        return pd.DataFrame(columns=["ID", "target_number", "original_sentence"])
    p = os.path.join(round0_gen_dir, "NumberModifier.csv")
    if not os.path.exists(p):
        return pd.DataFrame(columns=["ID", "target_number", "original_sentence"])
    try:
        df = pd.read_csv(p, encoding="utf-8")
    except Exception:
        return pd.DataFrame(columns=["ID", "target_number", "original_sentence"])
    need = {"ID", "target_number", "original_sentence"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame(columns=["ID", "target_number", "original_sentence"])
    df["ID"] = df["ID"].astype(str)
    df = df[df["ID"] == str(article_id)].copy()
    for c in ["target_number", "original_sentence"]:
        df[c] = df[c].fillna("").astype(str)
    df = df[(df["target_number"].str.strip() != "") & (df["original_sentence"].str.strip() != "")]
    return df[["ID", "target_number", "original_sentence"]].reset_index(drop=True)

def _pick_best_reference_sentence(target_sentence: str, reference_sentences: List[str]) -> str:
    if not reference_sentences:
        return ""
    try:
        _PIPELINE._ensure_nltk()
        emb_refs = _PIPELINE.embedder.encode(reference_sentences, convert_to_tensor=True)
        emb_t = _PIPELINE.embedder.encode([target_sentence], convert_to_tensor=True)[0]
        idx = int(util.cos_sim(emb_t, emb_refs).argmax())
        return reference_sentences[idx]
    except Exception:
        return reference_sentences[0]


def _realign_anchors_to_current_text(
    prev_ks: List[str],
    current_text: str,
    threshold: float = 0.65,
) -> Tuple[List[str], List[str]]:
    """Re-align key_sentences from a previous round to the current text via
    embedding similarity.  Returns (aligned_sentences, lost_sentences).
    aligned_sentences are the best-matching sentences IN current_text.
    lost_sentences are anchors whose best match fell below threshold."""
    if not prev_ks or not current_text:
        return [], list(prev_ks)
    try:
        _PIPELINE._ensure_nltk()
        sents = sent_tokenize(current_text)
        if not sents:
            return [], list(prev_ks)
        emb_sents = _PIPELINE.embedder.encode(sents, convert_to_tensor=True)
        aligned, lost = [], []
        for ks in prev_ks:
            emb_ks = _PIPELINE.embedder.encode([ks], convert_to_tensor=True)[0]
            sims = util.cos_sim(emb_ks, emb_sents)[0]
            best_sim = float(sims.max())
            best_idx = int(sims.argmax())
            if best_sim >= threshold:
                aligned.append(sents[best_idx])
            else:
                lost.append(ks)
        return aligned, lost
    except Exception:
        return list(prev_ks), []




def _norm_text_loose(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _any_change_list(prev_list: List[str], curr_list: List[str]) -> bool:
    prev_n = [_norm_text_loose(x) for x in (prev_list or []) if _norm_text_loose(x)]
    curr_n = [_norm_text_loose(x) for x in (curr_list or []) if _norm_text_loose(x)]
    if len(prev_n) != len(curr_n):
        return True
    prev_set = set(prev_n)
    for x in curr_n:
        if x not in prev_set:
            return True
    return False


def _sa_payload_round0(text: str) -> str:
    payload = {
        "current_article": text,
        "output_schema": {
            "key_sentences": ["<sentence verbatim from article>"],
            "key_numbers": ["<numeric expression verbatim from article>"],
            "number_to_sentence_map": {
                "<number>": "<exact sentence verbatim from article that contains it>"
            },
        },
    }
    return json.dumps(payload, ensure_ascii=False) + JSON_ONLY_SUFFIX

def _sa_payload_round_gt0(objective: str, original_text: str, prev_ks: List[str], prev_kn: List[str], chosen_feature: str = "", current_article: str = "", trajectory_feedback: Optional[Dict[str, Any]] = None) -> str:
    """SA pivot payload (2026-05-20 revision).

    Compression ratio features now surface `original_sentence_count` /
    `original_number_count` (computed from the round-0 REAL article) — these
    are the actual denominators used by the metric, not r-1 counts.
    """
    feature = (chosen_feature or "").strip()
    payload: Dict[str, Any] = {
        "objective": str(objective or "").strip(),
        "current_article": current_article or original_text or "",
    }
    if feature == "SA_semantic_coverage":
        # Metric is mean(max_cosine(sent_in_original_article, key_sentences)) — the
        # coverage target is the round-0 article, so we surface it here verbatim.
        payload["original_article"] = original_text or ""
        if prev_ks:
            payload["previous_key_sentences"] = list(prev_ks)
    elif feature == "SA_text_compression_ratio":
        # Denominator of the metric: count of sentences in the REAL round-0 article.
        # previous_sentence_count = how many key_sentences the agent extracted at r-1
        # (NOT cumulative — just the previous round). Lets the agent decide whether
        # to extract MORE or FEWER this round in line with the objective direction.
        try:
            _orig_sents = sent_tokenize(original_text) if (original_text or "").strip() else []
        except Exception:
            _orig_sents = []
        payload["original_sentence_count"] = int(len(_orig_sents))
        payload["previous_sentence_count"] = int(len(prev_ks or []))
    elif feature == "SA_numeric_coverage":
        # Metric is num_recall(key_numbers, numbers_extracted_from_original_article).
        payload["original_article"] = original_text or ""
        if prev_kn:
            payload["previous_key_numbers"] = list(prev_kn)
    elif feature == "SA_numeric_compression_ratio":
        # Denominator of the metric: count of distinct numeric values in the REAL round-0 article.
        # previous_number_count: how many key_numbers extracted at r-1, same purpose
        # as previous_sentence_count for the sentence compression feature.
        _orig_nums = _extract_numbers_from_text(original_text or "")
        payload["original_number_count"] = int(len(_orig_nums))
        payload["previous_number_count"] = int(len(prev_kn or []))
    payload["output_schema"] = {
        "key_sentences": ["<sentence verbatim from article>"],
        "key_numbers": ["<numeric expression verbatim from article>"],
        "number_to_sentence_map": {
            "<number>": "<exact sentence verbatim from article that contains it>"
        },
    }
    return json.dumps(payload, ensure_ascii=False) + JSON_ONLY_SUFFIX

def _sse_payload_round0(target_sentence: str) -> str:
    return f"Edit the following sentence:\n{target_sentence}" + JSON_ONLY_SUFFIX

def _sse_payload_pivot(objective: str, current_sentence: str,
                       original_sentence: str = "",
                       round_num: int = 0,
                       alignment_score: float = 1.0,
                       trajectory_feedback: Optional[Dict[str, Any]] = None) -> str:
    """SSE pivot payload (2026-05-20 revision).

    `current_sentence` is the sentence in the current article (post-NaM r-1)
    that represents the agent's last modification — operando of the rewrite.
    `original_sentence` is the round-0 REAL sentence cosine-matched against
    `current_sentence`, i.e. the ground-truth anchor the metric uses (see
    `metrics_runner.py:856-916` for SSE_semantic_drift / lexical_stability in
    anchored mode).

    The output schema mirrors the agent's contract: it must echo back
    `original_sentence` (= current_sentence, which is what it was asked to
    rewrite) and produce `modified_sentence`.
    """
    payload = {
        "objective": str(objective or "").strip(),
        "current_sentence": current_sentence or "",
        "original_sentence": original_sentence or current_sentence or "",
        "output_schema": {"original_sentence": "...", "modified_sentence": "..."},
    }
    return json.dumps(payload, ensure_ascii=False) + JSON_ONLY_SUFFIX

def _nm_payload_round0(sentence_with_number: str, target_number: str = "") -> str:
    payload = {
        "original_sentence": sentence_with_number or "",
        "target_number": str(target_number or ""),
        "instruction": "Edit the sentence by modifying the target number coherently while preserving the rest as much as possible.",
        "output_schema": {"original_sentence": "...", "modified_sentence": "..."},
    }
    return json.dumps(payload, ensure_ascii=False) + JSON_ONLY_SUFFIX

def _nm_payload_pivot(objective: str, current_sentence: str, target_number: str,
                      current_number: str = "",
                      alignment_score: float = 1.0,
                      trajectory_feedback: Optional[Dict[str, Any]] = None,
                      chosen_feature: str = "") -> str:
    """NM pivot payload (2026-05-20 channel-split revision).

    `target_number` is the ORIGINAL numeric value extracted by SA at round 0
      — the immutable identity anchor. NM_change_magnitude measures cumulative
      edit-distance between target_number and modified_number across rounds.
    `current_number` is the numeric token actually present in current_sentence
      right now (= lineage[-1] = result of the most recent successful NM edit
      on this target, or = target_number if NM has never modified it before).
      It already encodes "what I produced last round" — no separate
      previous_modified_number field is needed.
    `current_sentence` is the sentence in the current article (post-NaM r-1)
      that contains `current_number` — operando.

    The agent edits current_number INSIDE current_sentence; the identity of the
    target across rounds is target_number.
    """
    payload: Dict[str, Any] = {
        "objective": str(objective or "").strip(),
        "current_sentence": current_sentence or "",
        "target_number": str(target_number or ""),
    }
    cn = str(current_number or "").strip()
    if cn:
        payload["current_number"] = cn
    payload["output_schema"] = {"original_sentence": "...", "modified_sentence": "..."}
    return json.dumps(payload, ensure_ascii=False) + JSON_ONLY_SUFFIX

def _nam_payload_round0(narr_input_text: str) -> str:
    payload = {
        "objective": (
            "Edit the article to make it subtly false but plausible, "
            "applying one of the manipulation techniques from the list in your instructions. "
            "Preserve the overall topic and journalistic tone. "
            "Do not summarize; rewrite with targeted manipulation."
        ),
        "current_article": narr_input_text or "",
        "output_schema": {"modified_article": "..."},
    }
    return json.dumps(payload, ensure_ascii=False) + JSON_ONLY_SUFFIX

def _load_ks0_for_id(output_root: Optional[str], article_id: str) -> List[str]:
    # KS_0 are the round-0 key_sentences used by metrics_runner to score
    # NaM_contradiction_prob (premise=current_article, hypothesis=" ".join(KS_0)).
    # Showing the agent the same anchor the metric uses keeps the feedback signal aligned.
    if not output_root:
        return []
    path = os.path.join(output_root, "meta_frozen", "baseline_references.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            refs = json.load(f)
    except Exception:
        return []
    rec = refs.get(str(article_id), {})
    ks0 = rec.get("KS_0", []) if isinstance(rec, dict) else []
    return [str(s) for s in ks0 if str(s).strip()]


def _load_num2sent0_for_id(output_root: Optional[str], article_id: str) -> Dict[str, str]:
    """NUM2SENT_0 maps each round-0 key_number to the round-0 sentence that
    contained it (persisted in baseline_references.json by metrics_runner).
    Used by the NM pivot anchor strategy to disambiguate when multiple
    sentences in the current article contain the same target_number.
    """
    if not output_root:
        return {}
    path = os.path.join(output_root, "meta_frozen", "baseline_references.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            refs = json.load(f)
    except Exception:
        return {}
    rec = refs.get(str(article_id), {})
    n2s = rec.get("NUM2SENT_0", {}) if isinstance(rec, dict) else {}
    if not isinstance(n2s, dict):
        return {}
    return {str(k): str(v) for k, v in n2s.items() if str(v).strip()}


def _load_title_orig_for_id(output_root: Optional[str], article_id: str) -> str:
    """Round-0 original (REAL) title for this ID — the SAME anchor the style
    metric uses for the signed deviation current_title - original_title.
    Persisted in baseline_references.json as `title_orig` by metrics_runner."""
    if not output_root:
        return ""
    path = os.path.join(output_root, "meta_frozen", "baseline_references.json")
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            refs = json.load(f)
    except Exception:
        return ""
    rec = refs.get(str(article_id), {})
    return str(rec.get("title_orig", "")) if isinstance(rec, dict) else ""


def _load_kn0_for_id(output_root: Optional[str], article_id: str) -> List[str]:
    """KN_0 are the round-0 key_numbers — the fixed denominator universe
    for the new NM_edit_success_rate metric."""
    if not output_root:
        return []
    path = os.path.join(output_root, "meta_frozen", "baseline_references.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            refs = json.load(f)
    except Exception:
        return []
    rec = refs.get(str(article_id), {})
    kn0 = rec.get("KN_0", []) if isinstance(rec, dict) else []
    return [str(s) for s in kn0 if str(s).strip()]


def _sentences_containing_number(text: str, number: str) -> List[str]:
    """Return all sentences in `text` that contain `number` as an exact token
    (with the same loose normalisation used by _find_sentence_with_number).
    Used by the NM pivot anchor to pick the sentence to edit deterministically
    instead of via cosine similarity alone."""
    if not text or not str(number).strip():
        return []
    try:
        _PIPELINE._ensure_nltk()
        sents = sent_tokenize(text)
    except Exception:
        sents = re.split(r"(?<=[.!?])\s+", text)
    num_norm = re.sub(r"\s+", " ", str(number)).strip().strip('"').strip("'")
    num_nodash = re.sub(r"[-\s]", "", num_norm).lower()
    out: List[str] = []
    for s in sents:
        s_clean = (s or "").strip()
        if not s_clean:
            continue
        # Exact substring (post-normalisation) OR dash-collapsed substring.
        s_norm = re.sub(r"\s+", " ", s_clean)
        if num_norm and num_norm in s_norm:
            out.append(s_clean)
            continue
        s_nodash = re.sub(r"[-\s]", "", s_norm).lower()
        if num_nodash and num_nodash in s_nodash:
            out.append(s_clean)
    return out


def _sentence_contains_any_number(sentence: str, numbers: List[str]) -> bool:
    """True if `sentence` contains any token in `numbers` (loose, same matching
    as _sentences_containing_number). Used by _compute_sse_nm_targets to
    decide channel assignment."""
    if not sentence or not numbers:
        return False
    s_norm = re.sub(r"\s+", " ", str(sentence)).strip()
    s_nodash = re.sub(r"[-\s]", "", s_norm).lower()
    for n in numbers:
        n_str = str(n or "").strip().strip('"').strip("'")
        if not n_str:
            continue
        n_norm = re.sub(r"\s+", " ", n_str)
        if n_norm and n_norm in s_norm:
            return True
        n_nodash = re.sub(r"[-\s]", "", n_norm).lower()
        if n_nodash and n_nodash in s_nodash:
            return True
    return False


def _compute_sse_nm_targets(
    key_sentences: List[str],
    key_numbers: List[str],
    number_to_sentence_map: Dict[str, str],
) -> Tuple[List[str], List[Dict[str, str]]]:
    """Split SA output into disjoint SSE and NM channels (2026-05-20 refactor).

    Rule: a sentence containing any key_number belongs to NM exclusively.
    SSE only sees pure-semantic sentences (no key_numbers inside).

    Returns:
        sse_targets: key_sentences with NO key_number inside.
        nm_targets: list of {"number": n, "sentence": s} from n2s_map.
                    If a key_number lacks a sentence mapping it is dropped
                    (SA failed to anchor it — orchestrator guard will skip).
    """
    nums = [str(n) for n in (key_numbers or []) if str(n).strip()]
    sse_targets: List[str] = []
    for s in (key_sentences or []):
        s_str = str(s or "").strip()
        if not s_str:
            continue
        if _sentence_contains_any_number(s_str, nums):
            continue
        sse_targets.append(s_str)

    nm_targets: List[Dict[str, str]] = []
    seen_pairs: Set[Tuple[str, str]] = set()
    for n in nums:
        sent = str((number_to_sentence_map or {}).get(n, "") or "").strip()
        if not sent:
            continue
        key = (n, sent)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        nm_targets.append({"number": n, "sentence": sent})
    return sse_targets, nm_targets


def _lineage_current_number(
    output_root: Optional[str],
    detector_name: str,
    article_id: str,
    target_number: str,
    up_to_round: int,
) -> str:
    """Walk NumberModifier.csv across rounds [0, up_to_round) and return the
    most recent successful modified_number for (article_id, target_number).
    Falls back to target_number if no successful edit found.

    The lineage tracks: KN_0[i] -> mod_r0 -> mod_r1 -> ... -> mod_r{up_to_round-1}
    so the NM pivot at `up_to_round` knows which numeric token is actually
    in the current article (post-NaM).
    """
    if not output_root or not article_id:
        return target_number
    last = str(target_number or "")
    for r in range(0, max(int(up_to_round), 0)):
        rdir = os.path.join(output_root, "rounds", f"round_{r:03d}", "generation")
        # Round 0 generation is shared (no detector subdir); subsequent rounds
        # live under <generation>/<detector>/.
        candidates = [rdir, os.path.join(rdir, detector_name or "")]
        for base in candidates:
            p = os.path.join(base, "NumberModifier.csv")
            if not os.path.exists(p):
                continue
            try:
                df = pd.read_csv(p, encoding="utf-8")
            except Exception:
                continue
            if df.empty:
                continue
            df["ID"] = df["ID"].astype(str)
            sub = df[(df["ID"] == str(article_id)) &
                     (df.get("target_number", "").astype(str) == str(target_number))]
            if sub.empty:
                continue
            if "success" in sub.columns:
                sub_ok = sub[sub["success"].astype(str).str.lower().isin(["true", "1"])]
                # If no successful row this round, skip — do NOT overwrite `last`
                # with a NaN / empty modified_number from a failed-skip row.
                if sub_ok.empty:
                    break
                sub = sub_ok
            if "modified_number" in sub.columns:
                mn_raw = sub.iloc[-1].get("modified_number")
                # pandas reads empty CSV cells as NaN → str(NaN) == "nan",
                # which would silently contaminate the lineage. Guard explicitly.
                if pd.isna(mn_raw):
                    mn = ""
                else:
                    mn = str(mn_raw).strip()
                    if mn.lower() == "nan":
                        mn = ""
                if mn:
                    last = mn
            break
    return last


def _load_sse_targets0_for_id(output_root: Optional[str], article_id: str) -> List[str]:
    """SSE_TARGETS_0 = round-0 SSE channel anchors (key_sentences with no
    key_numbers inside). Persisted in baseline_references.json next to KS_0."""
    if not output_root:
        return []
    path = os.path.join(output_root, "meta_frozen", "baseline_references.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            refs = json.load(f)
    except Exception:
        return []
    rec = refs.get(str(article_id), {})
    out = rec.get("SSE_TARGETS_0", []) if isinstance(rec, dict) else []
    return [str(s) for s in out if str(s).strip()]


def _apply_channel_replacements(
    base_text: str,
    replacements: List[Tuple[str, str]],
) -> Tuple[str, List[Tuple[str, str]]]:
    """Apply each (original, modified) replacement to base_text in order.
    Returns (new_text, unmatched) where unmatched lists replacements whose
    `original` was not found verbatim in the text at apply-time. Used by the
    SSE/NM merge step to produce pre_nam_text deterministically."""
    text = base_text or ""
    unmatched: List[Tuple[str, str]] = []
    for orig, mod in (replacements or []):
        orig_s = str(orig or "")
        mod_s = str(mod or "")
        if not _norm_text_loose(orig_s) or not _norm_text_loose(mod_s):
            continue
        if orig_s in text:
            text = text.replace(orig_s, mod_s, 1)
        else:
            unmatched.append((orig_s, mod_s))
    return text, unmatched


# ============================================================
# Style-vector payload routing + deterministic title actuator.
# The 5 TE / 5 NaM style features are the SIGNED deviation of a stylometric
# dimension vs the round-0 REAL original; their agent payload carries that
# original (title/body) as the comparison anchor. The 3 TITLE FORMATTING
# features additionally get a direction-aware deterministic post-edit: the
# LLM keeps doing the semantic work, then we enforce the formatting dimension
# so the feature provably moves (closing the actuability gap and avoiding
# no_effect soft-blacklist starvation). `direction` is the frozen-beta sign
# decision already made by the orchestrator — the agent never sees it.
# ============================================================
_TE_STYLE_PAYLOAD_FEATURES = {
    "TE_style_titlecase", "TE_style_allcaps", "TE_style_exclquest",
    "TE_style_title_len", "TE_style_nnp",
}
_NAM_STYLE_PAYLOAD_FEATURES = {
    "NaM_style_fk_grade", "NaM_style_ttr", "NaM_style_quote_rate",
    "NaM_style_punct_rate", "NaM_style_adjadv",
}
_TITLE_STYLE_ACTUATOR_FEATURES = {
    "TE_style_titlecase", "TE_style_allcaps", "TE_style_exclquest",
}
_STYLE_ACT_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def _to_sentence_case(title: str) -> str:
    """Minimal title-case ratio: lowercase everything, capitalize only the
    first alphabetic character (metric titlecase = capitalized_tokens / tokens)."""
    s = (title or "").lower()
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.upper() + s[i + 1:]
    return s


def _to_title_case(title: str) -> str:
    """Maximal title-case ratio: capitalize the first letter of every token."""
    return _STYLE_ACT_TOKEN_RE.sub(
        lambda m: m.group(0)[:1].upper() + m.group(0)[1:].lower(), title or ""
    )


def _strip_allcaps(title: str) -> str:
    """Minimal all-caps rate: lowercase every fully-uppercase token (len>=2)."""
    def fix(m):
        w = m.group(0)
        return w.lower() if (len(w) >= 2 and w.isupper()) else w
    return _STYLE_ACT_TOKEN_RE.sub(fix, title or "")


def _add_allcaps(title: str) -> str:
    """Raise all-caps rate: uppercase the longest non-all-caps token (>=3 chars)."""
    cands = [m for m in _STYLE_ACT_TOKEN_RE.finditer(title or "")
             if len(m.group(0)) >= 3 and not m.group(0).isupper()]
    if not cands:
        return title or ""
    t = max(cands, key=lambda m: len(m.group(0)))
    s = title or ""
    return s[:t.start()] + t.group(0).upper() + s[t.end():]


def _strip_excl_quest(title: str) -> str:
    return (title or "").replace("!", "").replace("?", "")


def _add_excl_quest(title: str) -> str:
    t = (title or "").rstrip()
    if t.endswith("!") or t.endswith("?"):
        return title or ""
    return t.rstrip(".") + "!"


def _apply_title_style_actuator(chosen_feature: str, direction: str, new_title: str) -> str:
    """Direction-aware deterministic enforcement of the 3 title FORMATTING
    features; no-op for every other feature. Runs on the LLM output so the
    agent's semantic work is preserved and only the formatting dimension is
    forced in the orchestrator-chosen direction."""
    cf = str(chosen_feature or "").strip()
    if cf not in _TITLE_STYLE_ACTUATOR_FEATURES:
        return new_title
    up = str(direction or "").strip().upper() == "INCREASE"
    t = new_title or ""
    if cf == "TE_style_titlecase":
        return _to_title_case(t) if up else _to_sentence_case(t)
    if cf == "TE_style_allcaps":
        return _add_allcaps(t) if up else _strip_allcaps(t)
    if cf == "TE_style_exclquest":
        return _add_excl_quest(t) if up else _strip_excl_quest(t)
    return t


def _nam_payload_pivot(
    objective: str,
    chosen_feature: str,
    original_article: str,
    current_article: str,
    key_sentences: List[str],
    key_numbers: List[str],
    trajectory_feedback: Optional[Dict[str, Any]] = None,
    previous_attempt: Optional[Dict[str, Any]] = None,
    output_root: Optional[str] = None,
    article_id: str = "",
) -> str:
    """NaM pivot payload (2026-05-20 revision).

    `previous_article` (= NaM output r-1) was removed: at the call site it
    was always identical to `current_article` by construction (NaM pivot
    re-runs starting from prev_nam_article, ignoring SSE/NM of round r —
    see pipeline_multiagent.py:3942). Showing both fields confused the
    agent. `current_article` now plays both roles: operando AND the last
    move to surpass.
    """
    cf = str(chosen_feature or "").strip()
    payload: Dict[str, Any] = {
        "objective": str(objective or "").strip(),
        "current_article": current_article or "",
    }
    if cf == "NaM_rewrite_distance":
        payload["original_article"] = original_article or ""
    elif cf == "NaM_contradiction_prob":
        ks0 = _load_ks0_for_id(output_root, article_id)
        if ks0:
            payload["reference_key_sentences"] = ks0
        elif key_sentences:
            payload["reference_key_sentences"] = list(key_sentences)
    elif cf in _NAM_STYLE_PAYLOAD_FEATURES:
        # Body style feature = signed deviation vs the round-0 REAL article;
        # the agent compares its rewrite against that original.
        payload["original_article"] = original_article or ""
    else:
        payload["original_article"] = original_article or ""
        if key_sentences:
            payload["key_sentences"] = list(key_sentences)
        if key_numbers:
            payload["key_numbers"] = list(key_numbers)
    payload["output_schema"] = {"modified_article": "...", "technique_used": "..."}
    return json.dumps(payload, ensure_ascii=False) + JSON_ONLY_SUFFIX

def _te_payload_round0(original_title: str, text_for_title: str) -> str:
    payload = {
        "objective": "Assign a journalistically plausible title to the article below.",
        "original_title": original_title or "",
        "current_article": text_for_title or "",
        "output_schema": {"modified_title": "..."},
    }
    return json.dumps(payload, ensure_ascii=False) + JSON_ONLY_SUFFIX

def _te_payload_pivot(objective: str, current_title: str, current_article: str,
                      chosen_feature: str = "", original_title: str = "",
                      trajectory_feedback: Optional[Dict[str, Any]] = None) -> str:
    """TE pivot payload (2026-05-20 revision; style anchor 2026-06-17).

    For the NLI features the metric is pure NLI(current_article,
    current_title) — no round-0 anchor enters the calculation. For the 5 TE
    style features the metric is a signed deviation vs the round-0 REAL title,
    so we additionally show `original_title` as the comparison anchor.
    current_title always represents the last move (TE r-1 output, or the
    unmodified title if TE was not pivot at r-1).
    """
    payload: Dict[str, Any] = {
        "objective": str(objective or "").strip(),
        "current_title": current_title or "",
        "current_article": current_article or "",
    }
    if str(chosen_feature or "").strip() in _TE_STYLE_PAYLOAD_FEATURES:
        payload["original_title"] = original_title or ""
    payload["output_schema"] = {"modified_title": "..."}
    return json.dumps(payload, ensure_ascii=False) + JSON_ONLY_SUFFIX


def _build_panel_round_strict(
    round_idx: int,
    target_ids: List[str],
    gen_csv_path: str,
    skipped_ids_round: Set[str],
    gen_dir: Optional[str] = None,  # Bug-4 fix: optional generation directory for extra CSVs
) -> pd.DataFrame:
    df_gen = pd.read_csv(gen_csv_path, encoding="utf-8") if os.path.exists(gen_csv_path) else pd.DataFrame(columns=["ID", "title", "text"])
    if not df_gen.empty:
        df_gen["ID"] = df_gen["ID"].astype(str)
    gen_map = df_gen.set_index("ID")[["title", "text"]].to_dict(orient="index") if not df_gen.empty else {}

    # Bug-4 fix: load auxiliary CSVs from gen_dir if provided
    sa_map: dict = {}
    sse_map: dict = {}
    nm_map: dict = {}
    rr_map: dict = {}
    if gen_dir and os.path.isdir(gen_dir):
        # SemanticAnalyzer.csv -> key_sentences, key_numbers
        _sa_csv = os.path.join(gen_dir, "SemanticAnalyzer.csv")
        if os.path.exists(_sa_csv):
            try:
                _dsa = pd.read_csv(_sa_csv, encoding="utf-8")
                _dsa["ID"] = _dsa["ID"].astype(str)
                sa_map = _dsa.set_index("ID").to_dict(orient="index")
            except Exception:
                pass

        # SalientSentenceEditor.csv -> count of SSE edits per article
        _sse_csv = os.path.join(gen_dir, "SalientSentenceEditor.csv")
        if os.path.exists(_sse_csv):
            try:
                _dsse = pd.read_csv(_sse_csv, encoding="utf-8")
                _dsse["ID"] = _dsse["ID"].astype(str)
                sse_map = _dsse.groupby("ID").size().to_dict()
            except Exception:
                pass

        # NumberModifier.csv -> count of NM edits per article
        _nm_csv = os.path.join(gen_dir, "NumberModifier.csv")
        if os.path.exists(_nm_csv):
            try:
                _dnm = pd.read_csv(_nm_csv, encoding="utf-8")
                _dnm["ID"] = _dnm["ID"].astype(str)
                nm_map = _dnm.groupby("ID").size().to_dict()
            except Exception:
                pass

        # Round_Results.csv -> pivot_agent, chosen_feature, pipeline_status, skip info
        _rr_csv = os.path.join(gen_dir, "Round_Results.csv")
        if os.path.exists(_rr_csv):
            try:
                _drr = pd.read_csv(_rr_csv, encoding="utf-8")
                _drr["ID"] = _drr["ID"].astype(str)
                for _col in ("pivot_agent", "chosen_feature", "status_pipeline", "skip_stage", "skip_reason"):
                    if _col not in _drr.columns:
                        _drr[_col] = ""
                rr_map = _drr.set_index("ID").to_dict(orient="index")
            except Exception:
                pass

    rows = []
    for _id in target_ids:
        _id = str(_id)
        _rr = rr_map.get(_id, {})
        _sa = sa_map.get(_id, {})
        if _id in gen_map:
            rows.append({
                "ID": _id,
                "round": int(round_idx),
                "title_round_r": gen_map[_id].get("title", ""),
                "text_round_r": gen_map[_id].get("text", ""),
                "pred_label_round_r": "",
                "status_round_r": "OK",
                "source_round": int(round_idx),
                # Bug-4 fix: enriched columns
                "pivot_agent": _rr.get("pivot_agent", ""),
                "chosen_feature": _rr.get("chosen_feature", ""),
                "pipeline_status": _rr.get("status_pipeline", ""),
                "skip_stage": _rr.get("skip_stage", ""),
                "skip_reason": _rr.get("skip_reason", ""),
                "key_sentences": _sa.get("key_sentences", ""),
                "key_numbers": _sa.get("key_numbers", ""),
                "sse_edits": int(sse_map.get(_id, 0)),
                "nm_edits": int(nm_map.get(_id, 0)),
            })
        else:
            rows.append({
                "ID": _id,
                "round": int(round_idx),
                "title_round_r": "",
                "text_round_r": "",
                "pred_label_round_r": "",
                "status_round_r": "SKIPPED" if _id in skipped_ids_round else "MISSING",
                "source_round": -1,
                # Bug-4 fix: enriched columns (empty for skipped/missing)
                "pivot_agent": _rr.get("pivot_agent", ""),
                "chosen_feature": _rr.get("chosen_feature", ""),
                "pipeline_status": _rr.get("status_pipeline", ""),
                "skip_stage": _rr.get("skip_stage", ""),
                "skip_reason": _rr.get("skip_reason", ""),
                "key_sentences": _sa.get("key_sentences", ""),
                "key_numbers": _sa.get("key_numbers", ""),
                "sse_edits": int(sse_map.get(_id, 0)),
                "nm_edits": int(nm_map.get(_id, 0)),
            })
    return pd.DataFrame(rows)


_AGENT_ORDER = {"SA": 0, "SSE": 1, "NM": 2, "NaM": 3, "TE": 4}

def _is_after_pivot(agent: str, pivot: str) -> bool:
    a = _normalize_agent_name(agent)
    p = _normalize_agent_name(pivot)
    return _AGENT_ORDER.get(a, 999) > _AGENT_ORDER.get(p, -1)

def _is_pivot(agent: str, pivot: str) -> bool:
    return _normalize_agent_name(agent) == _normalize_agent_name(pivot)


def _run_generation_pipeline_impl(
    pipeline: MultiAgentGenerationPipeline,
    df_input: pd.DataFrame,
    base_out_dir: str,
    sample_size: int | None = 100,
    sample_ids_path: str | None = None,
    resume: bool = True,
    process_only_missing_round0: bool = False,
    df_prev: pd.DataFrame | None = None,
    prev_round_dir: str | None = None,
    prev_generation_dir: str | None = None,
    prev_orch_dir: str | None = None,
    detector_name: str | None = None,
) -> pd.DataFrame:

    _ensure_dir(base_out_dir)
    checkpoint_path = os.path.join(base_out_dir, "LLM_STOP_CHECKPOINT.json")

    round_idx = _infer_round_index_from_path(base_out_dir)
    prev_round_idx = _get_prev_round_idx(round_idx)
    _warn_set_context(round=round_idx, detector=detector_name)

    if prev_generation_dir is None:
        prev_generation_dir = prev_round_dir
    if prev_orch_dir is None:
        prev_orch_dir = prev_round_dir

    control_enabled = (
        df_prev is not None and
        prev_generation_dir is not None and
        prev_orch_dir is not None and
        detector_name is not None
    )

    if "ID" not in df_input.columns:
        raise ValueError("df_input must contain column 'ID'")
    df_source_full = df_input.copy()
    df_source_full["ID"] = df_source_full["ID"].astype(str)

    if df_prev is not None:
        df_prev = df_prev.copy()
        if "ID" not in df_prev.columns:
            raise ValueError("df_prev must contain column 'ID'")
        df_prev["ID"] = df_prev["ID"].astype(str)

    sa_path = os.path.join(base_out_dir, "SemanticAnalyzer.csv")
    sse_path = os.path.join(base_out_dir, "SalientSentenceEditor.csv")
    nm_path = os.path.join(base_out_dir, "NumberModifier.csv")
    narr_path = os.path.join(base_out_dir, "NarrativeModifier.csv")
    title_path = os.path.join(base_out_dir, "TitleEditor.csv")
    final_output_path = os.path.join(base_out_dir, "Generated_Fake_News.csv")
    # Fase A — drift audit log for SSE/NM territorial violations
    drift_log_path = os.path.join(base_out_dir, "agent_drift_log.csv")

    # §4: agent snapshot paths
    _output_root = _find_output_root(base_out_dir)
    _snaps_base_dir: Optional[str] = (
        os.path.join(_output_root, "trajectory", detector_name, "agent_snapshots")
        if (_output_root and detector_name) else None
    )
    _snap_round_dir: Optional[str] = (
        os.path.join(_snaps_base_dir, f"round_{round_idx:03d}")
        if (_snaps_base_dir and round_idx != -999) else None
    )
    # §10: article snapshot paths
    _art_snaps_base_dir: Optional[str] = (
        os.path.join(_output_root, "trajectory", detector_name, "article_snapshots")
        if (_output_root and detector_name) else None
    )
    _art_snap_round_dir: Optional[str] = (
        os.path.join(_art_snaps_base_dir, f"round_{round_idx:03d}")
        if (_art_snaps_base_dir and round_idx != -999) else None
    )
    panel_path = os.path.join(base_out_dir, "panel_articles_round_r.parquet")

    skips_path = os.path.join(base_out_dir, "Skipped_IDs.csv")
    reuse_path = os.path.join(base_out_dir, "Reuse_Log.csv")
    round_results_path = os.path.join(base_out_dir, "Round_Results.csv")

    df_skips = pd.read_csv(skips_path, encoding="utf-8") if (resume and os.path.exists(skips_path)) else pd.DataFrame(columns=["ID", "stage", "reason"])
    df_reuse = pd.read_csv(reuse_path, encoding="utf-8") if (resume and os.path.exists(reuse_path)) else pd.DataFrame(columns=["ID", "what", "reason"])
    df_roundres = pd.read_csv(round_results_path, encoding="utf-8") if (resume and os.path.exists(round_results_path)) else pd.DataFrame(columns=[
        "ID", "pivot_agent", "chosen_feature", "objective_len",
        "status_pipeline", "skip_stage", "skip_reason",
    ])

    def _skipped_ids_current_round() -> Set[str]:
        return _skipped_ids_from_df(df_skips)

    ridx_for_panel = int(round_idx if round_idx != -999 else 999999)

    if not control_enabled:
        if sample_ids_path:
            try:
                df_ids = pd.read_csv(sample_ids_path, encoding="utf-8")
                sample_ids = df_ids["ID"].astype(str).tolist() if "ID" in df_ids.columns else df_source_full["ID"].astype(str).tolist()
            except Exception:
                sample_ids = df_source_full["ID"].astype(str).tolist()
        else:
            sample_ids = df_source_full["ID"].astype(str).tolist()

        done_ids = _read_ids_from_csv(final_output_path) if (resume and os.path.exists(final_output_path)) else set()
        skipped_ids = _skipped_ids_current_round() if resume else set()
        target_ids = sample_ids
        if resume:
            target_ids = [i for i in sample_ids if i not in skipped_ids]
        if process_only_missing_round0 and resume:
            target_ids = [i for i in target_ids if i not in done_ids]

        if resume and skipped_ids:
            print(f"[PIPE] Round0 resume: keeping skipped IDs skipped -> n={len(skipped_ids)}")

        rows_generated = []

        panel_target_ids = [str(x) for x in sample_ids]

        for idx, article_id in enumerate(tqdm(target_ids, desc="[PIPE] Round0 generating")):
            _hard_stop_if_tpd(
                "round0:loop_top", df_skips, skips_path, checkpoint_path, extra={"round": 0, "i": idx},
            )
            article_id = str(article_id)

            r = df_source_full[df_source_full["ID"] == article_id]
            if r.empty:
                _append_skip_and_flush(
                    df_skips=df_skips, article_id=article_id, stage="INPUT", reason="ID not found in df_input -> SKIP",
                    skips_path=skips_path,
                    df_reuse=df_reuse, reuse_path=reuse_path,
                    df_roundres=df_roundres, round_results_path=round_results_path,
                    panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                    panel_target_ids=panel_target_ids,
                    gen_csv_path=final_output_path,
                )
                continue
            r = r.iloc[0]
            title = str(r.get("title", ""))
            text = str(r.get("text", ""))

            if resume and ((article_id in done_ids) or (article_id in skipped_ids)):
                continue

            sa_msg = _sa_payload_round0(text)
            resp_sa = _chat_with_retries(
                "SemanticAnalyzer", SemanticAnalyzer, sa_msg,
                retries=3, backoff_base=2.0,
                verbose=True,
            )
            if resp_sa is None:
                _append_skip_and_flush(
                    df_skips=df_skips, article_id=article_id, stage="SA", reason="llm_failed_after_retries",
                    skips_path=skips_path,
                    df_reuse=df_reuse, reuse_path=reuse_path,
                    df_roundres=df_roundres, round_results_path=round_results_path,
                    panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                    panel_target_ids=panel_target_ids,
                    gen_csv_path=final_output_path,
                )
                continue
            sa_obj = _parse_with_local_repair(resp_sa)
            coerced = _coerce_sa(sa_obj) if isinstance(sa_obj, dict) else None
            if coerced is None:
                ok, ks, nums = check_semantic_analyzer_format(SemanticAnalyzer)
                if ok:
                    coerced = ([str(x) for x in ks], _sanitize_numbers([str(x) for x in nums]), {})
            if coerced is None:
                _append_skip_and_flush(
                    df_skips=df_skips, article_id=article_id, stage="SA", reason="invalid_json",
                    skips_path=skips_path,
                    df_reuse=df_reuse, reuse_path=reuse_path,
                    df_roundres=df_roundres, round_results_path=round_results_path,
                    panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                    panel_target_ids=panel_target_ids,
                    gen_csv_path=final_output_path,
                )
                continue
            key_sentences, key_numbers, n2s_map = coerced
            # Bug-5 fix: strip any numbers that are prompt-example placeholders or
            # not actually present in the source text.
            key_numbers = _sanitize_numbers_strict(key_numbers, text)
            n2s_map = {k: v for k, v in n2s_map.items() if k in set(key_numbers)}
            # Bugfix 2026-05-15: LLM frequently returns empty/incomplete
            # number_to_sentence_map (format="json" doesn't enforce schema, and
            # even with schema the field can be `{}`). Synthesize missing entries
            # from the article text so SSE tracking and NM sentence lookup work.
            if len(n2s_map) < len(key_numbers):
                _synth = _build_n2s_from_text(key_numbers, text)
                _added = 0
                for _k, _v in _synth.items():
                    if _k not in n2s_map:
                        n2s_map[_k] = _v
                        _added += 1
                if _added:
                    print(f"[SA] synthesized {_added}/{len(key_numbers)} n2s_map entries from text for ID={article_id}")

            # 2026-05-20 — Channel split (disjoint SSE / NM channels).
            # SSE works on key_sentences with NO key_number inside.
            # NM works on (number, sentence) pairs from n2s_map.
            # Hybrid sentences (key_sentence containing a key_number) belong
            # ONLY to NM. This eliminates the SSE-eats-NM-target failure mode.
            sse_targets_r0, nm_targets_r0 = _compute_sse_nm_targets(
                key_sentences, key_numbers, n2s_map,
            )
            sse_replacements_r0: List[Tuple[str, str]] = []
            nm_replacements_r0: List[Tuple[str, str]] = []

            _sse_snap_edits_r0: List[dict] = []
            if sse_targets_r0:
                for ks in sse_targets_r0:
                    target_sentence = _match_sentence_by_embedding(text, ks) or ks
                    sse_msg = _sse_payload_round0(target_sentence)
                    resp_sse = _chat_with_retries(
                        "SalientSentenceEditor", SalientSentenceEditor, sse_msg,
                        retries=3, backoff_base=2.0,
                        verbose=True,
                    )
                    if resp_sse is None:
                        continue
                    sse_obj = _parse_with_local_repair(resp_sse)
                    pair = _coerce_sse_nm(sse_obj) if isinstance(sse_obj, dict) else None
                    if pair is None:
                        ok, o, m = check_salient_sentence_editor_format(SalientSentenceEditor)
                        if ok:
                            pair = (str(o), str(m))
                    if pair is None:
                        continue

                    original_sentence, modified_sentence = pair[0], pair[1]
                    if not original_sentence.strip():
                        original_sentence = target_sentence

                    # 2026-05-20 channel-split: SSE works only on non-numeric
                    # sentences. _realign_numbers_post_sse / _repair_lost_numbers_sse
                    # are no longer needed for state recovery — they remain only
                    # as informational drift audit (SSE should never introduce numbers).
                    _nio, _nim, _lost, _renamed = _realign_numbers_post_sse(original_sentence, modified_sentence, n2s_map)

                    sse_replacements_r0.append((original_sentence, modified_sentence))
                    if target_sentence != original_sentence:
                        sse_replacements_r0.append((target_sentence, modified_sentence))

                    _append_sse_intermediate_exact(sse_path, article_id, original_sentence, modified_sentence,
                                                  numbers_in_original=_nio, numbers_in_modified=_nim,
                                                  lost_numbers=_lost, renamed_numbers=_renamed)
                    if _lost or _renamed:
                        _append_agent_drift_log(
                            drift_log_path, article_id, 0, "SSE",
                            [{"violation_type": "channel_leak",
                              "detail": f"lost={_lost} renamed={_renamed}"}],
                            "", original_sentence, modified_sentence,
                        )
                    _sse_snap_edits_r0.append({
                        "original_sentence": original_sentence,
                        "modified_sentence": modified_sentence,
                        "numbers_in_original": _nio,
                        "numbers_in_modified": _nim,
                        "lost_numbers": _lost,
                        "renamed_numbers": _renamed,
                    })

            # §4 snapshot — SSE round 0
            _save_agent_snapshot(_snap_round_dir, "SSE", article_id, round_idx, False, _sse_snap_edits_r0)

            # 2026-05-20 channel-split: NM iterates over nm_targets_r0 (disjoint
            # from sse_targets_r0). Both channels read from the SAME base text
            # (the REAL article) — NM does not see SSE's edits. Merge happens
            # after both loops via _apply_channel_replacements.
            _nm_snap_edits_r0: List[dict] = []
            for _nm_target in nm_targets_r0:
                _kn_orig_str = str(_nm_target.get("number", ""))
                _sent_current = str(_nm_target.get("sentence", ""))
                if not _sent_current:
                    _warn("NM_NO_SENTENCE", "no sentence found for number",
                          article_id=article_id, number=_kn_orig_str)
                    _append_nm_intermediate_exact(
                        nm_path, article_id, _kn_orig_str, "", "",
                        success=False, reason="no_sentence_found", modified_number="",
                    )
                    continue
                nm_msg = _nm_payload_round0(_sent_current, target_number=_kn_orig_str)
                resp_nm = _chat_with_retries(
                    "NumberModifier", NumberModifier, nm_msg,
                    retries=3, backoff_base=2.0,
                    verbose=True,
                )
                if resp_nm is None:
                    continue
                nm_obj = _parse_with_local_repair(resp_nm)
                pair = _coerce_sse_nm(nm_obj) if isinstance(nm_obj, dict) else None
                if pair is None:
                    ok, o, m = check_number_modifier_format(NumberModifier)
                    if ok:
                        pair = (str(o), str(m))
                if pair is None:
                    continue
                original_sentence, modified_sentence = pair[0], pair[1]
                if not original_sentence.strip():
                    original_sentence = _sent_current
                if _norm_text_loose(original_sentence) and _norm_text_loose(modified_sentence):
                    nm_replacements_r0.append((original_sentence, modified_sentence))
                    if _sent_current != original_sentence:
                        nm_replacements_r0.append((_sent_current, modified_sentence))
                _modified_number = _extract_number_from_sentence(modified_sentence, _kn_orig_str)
                _append_nm_intermediate_exact(
                    nm_path, article_id, _kn_orig_str, original_sentence, modified_sentence,
                    success=True, modified_number=_modified_number,
                )
                # Fase A — NM drift audit (no rejection, just logging)
                _nm_violations_r0 = _compute_nm_drift_violations(_kn_orig_str, original_sentence, modified_sentence)
                if _nm_violations_r0:
                    _append_agent_drift_log(
                        drift_log_path, article_id, 0, "NM", _nm_violations_r0,
                        _kn_orig_str, original_sentence, modified_sentence,
                    )
                    _warn("NM_DRIFT_VIOLATION", "NM violated edit constraints",
                          article_id=article_id, target=_kn_orig_str,
                          n_violations=len(_nm_violations_r0),
                          types=",".join(v["violation_type"] for v in _nm_violations_r0))
                _nm_snap_edits_r0.append({
                    "key_number_original_sa": _kn_orig_str,
                    "key_number_after_sse": _kn_orig_str,
                    "key_number_after_nm": _modified_number,
                    "sentence_after_sse": _sent_current,
                    "sentence_after_nm": modified_sentence,
                })

            # §4 snapshot — NM round 0
            _save_agent_snapshot(_snap_round_dir, "NM", article_id, round_idx, False, _nm_snap_edits_r0)

            # Merge: apply both channels' replacements on the same base text (REAL).
            # SSE and NM channels are disjoint by construction, so order is irrelevant.
            pre_nam_text_r0, _unmatched_sse_r0 = _apply_channel_replacements(text, sse_replacements_r0)
            pre_nam_text_r0, _unmatched_nm_r0 = _apply_channel_replacements(pre_nam_text_r0, nm_replacements_r0)
            if _unmatched_sse_r0 or _unmatched_nm_r0:
                _warn("CHANNEL_MERGE_UNMATCHED", "some replacements not applied verbatim",
                      article_id=article_id,
                      unmatched_sse=len(_unmatched_sse_r0),
                      unmatched_nm=len(_unmatched_nm_r0))

            nam_msg = _nam_payload_round0(pre_nam_text_r0)
            resp_nam = _chat_with_retries(
                "NarrativeModifier", NarrativeModifier, nam_msg,
                retries=3, backoff_base=2.0,
                verbose=True,
            )
            if resp_nam is None:
                _append_skip_and_flush(
                    df_skips=df_skips, article_id=article_id, stage="NaM", reason="llm_failed_after_retries",
                    skips_path=skips_path,
                    df_reuse=df_reuse, reuse_path=reuse_path,
                    df_roundres=df_roundres, round_results_path=round_results_path,
                    panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                    panel_target_ids=panel_target_ids,
                    gen_csv_path=final_output_path,
                )
                continue
            nam_obj = _parse_with_local_repair(resp_nam)
            _nam_coerced = _coerce_nam(nam_obj) if isinstance(nam_obj, dict) else None
            if _nam_coerced is None:
                ok, mod_article2 = check_narrative_modifier_format(NarrativeModifier)
                if ok:
                    mod_article = str(mod_article2)
                    nam_technique = ""
                else:
                    mod_article = None
                    nam_technique = ""
            else:
                mod_article, nam_technique = _nam_coerced
            if mod_article is None:
                _append_skip_and_flush(
                    df_skips=df_skips, article_id=article_id, stage="NaM", reason="invalid_json",
                    skips_path=skips_path,
                    df_reuse=df_reuse, reuse_path=reuse_path,
                    df_roundres=df_roundres, round_results_path=round_results_path,
                    panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                    panel_target_ids=panel_target_ids,
                    gen_csv_path=final_output_path,
                )
                continue

            # TE base (Round0 must receive ORIGINAL_TITLE)
            te_msg = _te_payload_round0(title, mod_article)
            resp_te = _chat_with_retries(
                "TitleEditor", TitleEditor, te_msg,
                retries=3, backoff_base=2.0,
                verbose=True,
            )
            if resp_te is None:
                _append_skip_and_flush(
                    df_skips=df_skips, article_id=article_id, stage="TE", reason="llm_failed_after_retries",
                    skips_path=skips_path,
                    df_reuse=df_reuse, reuse_path=reuse_path,
                    df_roundres=df_roundres, round_results_path=round_results_path,
                    panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                    panel_target_ids=panel_target_ids,
                    gen_csv_path=final_output_path,
                )
                continue
            te_obj = _parse_with_local_repair(resp_te)
            new_title = _coerce_te(te_obj) if isinstance(te_obj, dict) else None
            if new_title is None:
                ok, t2 = check_title_editor_format(TitleEditor)
                if ok:
                    new_title = str(t2)
            if new_title is None:
                _append_skip_and_flush(
                    df_skips=df_skips, article_id=article_id, stage="TE", reason="invalid_json",
                    skips_path=skips_path,
                    df_reuse=df_reuse, reuse_path=reuse_path,
                    df_roundres=df_roundres, round_results_path=round_results_path,
                    panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                    panel_target_ids=panel_target_ids,
                    gen_csv_path=final_output_path,
                )
                continue

            _safe_append_dedupe_csv(sa_path, pd.DataFrame([{
                "ID": article_id,
                "key_sentences": json.dumps(key_sentences, ensure_ascii=False),
                "key_numbers": json.dumps(key_numbers, ensure_ascii=False),
                "number_to_sentence_map": json.dumps(n2s_map, ensure_ascii=False),
                "sse_targets": json.dumps(sse_targets_r0, ensure_ascii=False),
                "nm_targets": json.dumps(nm_targets_r0, ensure_ascii=False),
            }]), subset_cols=["ID"], keep="last")

            _safe_append_dedupe_csv(narr_path, pd.DataFrame([{
                "ID": article_id,
                "original_article": pre_nam_text_r0,
                "modified_article": mod_article,
                "technique_used": nam_technique,
            }]), subset_cols=["ID"], keep="last")

            _safe_append_dedupe_csv(title_path, pd.DataFrame([{
                "ID": article_id,
                "original_title": title,
                "modified_title": new_title,
            }]), subset_cols=["ID"], keep="last")

            rows_generated.append({"ID": article_id, "title": new_title, "text": mod_article})

            # §10 — save article snapshot for round 0
            _save_article_snapshot(_art_snap_round_dir, article_id, mod_article)

            _safe_save_generated(final_output_path, pd.DataFrame([{
                "ID": article_id,
                "title": new_title,
                "text": mod_article,
            }]))

            try:
                if not df_skips.empty and "ID" in df_skips.columns:
                    df_skips = df_skips[df_skips["ID"].astype(str) != str(article_id)].copy()
                    df_skips.to_csv(skips_path, index=False, encoding="utf-8")
            except Exception:
                pass

            _flush_all_after_id(
                df_skips=df_skips, skips_path=skips_path,
                df_reuse=df_reuse, reuse_path=reuse_path,
                df_roundres=df_roundres, round_results_path=round_results_path,
                panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                panel_target_ids=panel_target_ids,
                gen_csv_path=final_output_path,
            )

            if (len(rows_generated) > 0) and (len(rows_generated) % FLUSH_EVERY_N_IDS == 0):
                _safe_save_generated(final_output_path, pd.DataFrame(rows_generated))
                rows_generated = []
                _flush_all_after_id(
                    df_skips=df_skips, skips_path=skips_path,
                    df_reuse=df_reuse, reuse_path=reuse_path,
                    df_roundres=df_roundres, round_results_path=round_results_path,
                    panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                    panel_target_ids=panel_target_ids,
                    gen_csv_path=final_output_path,
                )

        if rows_generated:
            _safe_save_generated(final_output_path, pd.DataFrame(rows_generated))

        _flush_all_after_id(
            df_skips=df_skips, skips_path=skips_path,
            df_reuse=df_reuse, reuse_path=reuse_path,
            df_roundres=df_roundres, round_results_path=round_results_path,
            panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
            panel_target_ids=panel_target_ids,
            gen_csv_path=final_output_path,
        )

        if os.path.exists(final_output_path):
            return pd.read_csv(final_output_path, encoding="utf-8")
        return pd.DataFrame(columns=["ID", "title", "text"])

    # ==============================
    # ROUND > 0
    # ==============================
    control_df = _load_orch_control_df(prev_orch_dir, detector_name)
    if control_df is None:
        return pd.DataFrame(columns=["ID", "title", "text"])

    active_ids = set(control_df.index.tolist())

    # Intersect active_ids with sample_ids_path (active_ids_in.csv written by
    # run_loop_rounds.py before each call). The orchestrator control_df from the previous
    # round may contain IDs that have since flipped to DONE_REAL or been marked ERROR;
    # run_loop_rounds.py writes only the current ACTIVE_FAKE subset to active_ids_in.csv.
    # Without this intersection, stale IDs reach df_input as misses → Skipped_IDs.csv noise.
    if sample_ids_path and os.path.exists(sample_ids_path):
        allowed_ids = _read_ids_from_csv(sample_ids_path)
        if allowed_ids:
            before = len(active_ids)
            active_ids = active_ids & allowed_ids
            filtered = before - len(active_ids)
            if filtered > 0:
                print(f"[PIPE] Round>0: filtered {filtered} DONE/excluded IDs via sample_ids_path "
                      f"({before} -> {len(active_ids)} active)")

    panel_target_ids = [str(x) for x in sorted(active_ids)]

    # Round>0: Generated_Fake_News.csv must NOT be cumulative across rounds.
    if resume and os.path.exists(final_output_path):
        try:
            df_existing = pd.read_csv(final_output_path, encoding="utf-8")
            if "ID" in df_existing.columns:
                df_existing["ID"] = df_existing["ID"].astype(str)
                df_existing = df_existing[df_existing["ID"].isin(active_ids)].copy()
                if set(["ID", "title", "text"]).issubset(df_existing.columns):
                    df_existing = df_existing[["ID", "title", "text"]]
                df_existing.to_csv(final_output_path, index=False, encoding="utf-8")
        except Exception:
            pass

    done_ids_this_round = _read_ids_from_csv(final_output_path) if (resume and os.path.exists(final_output_path)) else set()
    skipped_ids_this_round = _skipped_ids_current_round() if resume else set()
    missing_ids = list(sorted((active_ids - done_ids_this_round) - skipped_ids_this_round))

    print(f"[PIPE] Round>0: processing ONLY orchestrated IDs: n={len(active_ids)}")
    print(f"[PIPE] Round>0 resume: target={len(active_ids)} done={len(done_ids_this_round & active_ids)} missing={len(missing_ids)} skipped={len(active_ids & skipped_ids_this_round)}")

    # Bug 2 fix: guarantee Generated_Fake_News.csv exists BEFORE the generation loop.
    # If every ID ends up being skipped/failed, the outer run_loop_rounds.py would
    # hard-stop on the missing-file check.  An empty file lets it reach the softer
    # "len(generated_in_scope)==0" guard instead, which breaks cleanly.
    if not os.path.exists(final_output_path):
        _ensure_dir(os.path.dirname(final_output_path) or ".")
        pd.DataFrame(columns=["ID", "title", "text"]).to_csv(
            final_output_path, index=False, encoding="utf-8"
        )

    rows_generated = []

    for i, article_id in enumerate(tqdm(missing_ids, desc="[PIPE] Round>0 generating missing IDs")):
        _hard_stop_if_tpd(
            "round>0:loop_top", df_skips, skips_path, checkpoint_path, extra={"round": round_idx, "i": i},
        )
        article_id = str(article_id)

        src_row = df_source_full[df_source_full["ID"] == article_id]
        if src_row.empty:
            _append_skip_and_flush(
                df_skips=df_skips, article_id=article_id, stage="CTRL", reason="ID not found in df_input scope -> SKIP",
                skips_path=skips_path,
                df_reuse=df_reuse, reuse_path=reuse_path,
                df_roundres=df_roundres, round_results_path=round_results_path,
                panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                panel_target_ids=panel_target_ids,
                gen_csv_path=final_output_path,
            )
            continue
        src_row = src_row.iloc[0]

        prev_row = df_prev[df_prev["ID"] == article_id]
        if prev_row.empty:
            current_title = str(src_row.get("title", ""))
            current_text = str(src_row.get("text", ""))
        else:
            current_title = str(prev_row.iloc[0].get("title", "")) or str(src_row.get("title", ""))
            current_text = str(prev_row.iloc[0].get("text", "")) or str(src_row.get("text", ""))

        original_text = str(src_row.get("text", ""))

        row_ctrl = control_df.loc[article_id]
        pivot = _infer_pivot_agent(row_ctrl)
        chosen_feature = str(row_ctrl.get("chosen_feature", "") or "").strip()
        # Frozen-beta direction (INCREASE/DECREASE) — internal only, NEVER shown
        # to agents; used for the deterministic title-style actuator.
        _dir_raw = row_ctrl.get("direction", "")
        direction = "" if (_dir_raw is None or (isinstance(_dir_raw, float) and pd.isna(_dir_raw))) else str(_dir_raw).strip().upper()
        # Round-0 REAL title anchor for the TE style features' signed deviation.
        _te_orig_title = _load_title_orig_for_id(_output_root, str(article_id))
        objective = _extract_objective(row_ctrl)
        if not objective.strip():
            objective = "Revise the assigned content according to the instructions, preserving coherence, plausibility, and required output format."
        trajectory_feedback = _extract_trajectory_feedback(row_ctrl)

        # §5 — parse previous_attempt from orchestrator decisions
        _prev_attempt_raw = str(row_ctrl.get("previous_attempt_json", "") or "")
        previous_attempt: Optional[Dict[str, Any]] = None
        if _prev_attempt_raw:
            try:
                _pa = json.loads(_prev_attempt_raw)
                if isinstance(_pa, dict) and _pa:
                    previous_attempt = _pa
            except Exception:
                pass

        run_seq = _compute_run_from_pivot(pivot)
        run_SA, run_SSE, run_NM, run_NaM, run_TE = (
            run_seq["run_SA"], run_seq["run_SSE"], run_seq["run_NM"], run_seq["run_NaM"], run_seq["run_TE"]
        )

        rr = {
            "ID": article_id,
            "pivot_agent": pivot,
            "chosen_feature": chosen_feature,
            "objective_len": int(len(objective or "")),
            "status_pipeline": "OK",
            "skip_stage": "",
            "skip_reason": "",
        }

        # For pivot-effect comparisons ONLY (prev round outputs)
        prev_sse_out = _load_prev_sse_modified_sentences(prev_generation_dir, article_id)
        prev_nm_out = _load_prev_nm_modified_sentences(prev_generation_dir, article_id)

        # Load prev intermediates once (used by pivot SSE/NM, and NaM pivot inputs)
        prev_sse_df = _load_prev_sse_rows(prev_generation_dir, article_id)
        prev_nm_df = _load_prev_nm_rows(prev_generation_dir, article_id)

        prev_ks, prev_kn, prev_n2s_map = _load_prev_sa_output(prev_generation_dir, article_id)
        prev_nam_article = _load_prev_nam_article(prev_generation_dir, article_id)

        # ---------- Semantic anchors from previous round ----------
        key_sentences: List[str] = list(prev_ks)
        key_numbers: List[str] = list(prev_kn)

        if run_SA:
            _sa_pivot_ks: List[str] = list(prev_ks or [])
            _sa_pivot_kn: List[str] = list(prev_kn or [])
            if _is_pivot("SA", pivot):
                cf_norm = (chosen_feature or "").strip()
                if cf_norm == "SA_semantic_coverage":
                    _aligned_ks, _lost_ks = _realign_anchors_to_current_text(
                        prev_ks, current_text, threshold=0.65
                    )
                    if _aligned_ks:
                        _sa_pivot_ks = _aligned_ks
                    if _lost_ks:
                        _append_reuse_and_flush(
                            df_reuse=df_reuse, article_id=article_id,
                            what="SA_PIVOT_ANCHOR_LOST",
                            reason=f"feature={cf_norm}: {len(_lost_ks)} anchor(s) dropped after realignment to current_article",
                            df_skips=df_skips, skips_path=skips_path,
                            reuse_path=reuse_path,
                            df_roundres=df_roundres, round_results_path=round_results_path,
                            panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                            panel_target_ids=panel_target_ids,
                            gen_csv_path=final_output_path,
                        )
                elif cf_norm == "SA_numeric_coverage":
                    _filtered_kn = [n for n in (prev_kn or []) if str(n).strip() and str(n) in (current_text or "")]
                    if len(_filtered_kn) < len(prev_kn or []):
                        _lost_nums = [n for n in (prev_kn or []) if n not in _filtered_kn]
                        _append_reuse_and_flush(
                            df_reuse=df_reuse, article_id=article_id,
                            what="SA_PIVOT_NUMBER_LOST",
                            reason=f"feature={cf_norm}: {len(_lost_nums)} number(s) no longer present in current_article: {_lost_nums[:3]}",
                            df_skips=df_skips, skips_path=skips_path,
                            reuse_path=reuse_path,
                            df_roundres=df_roundres, round_results_path=round_results_path,
                            panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                            panel_target_ids=panel_target_ids,
                            gen_csv_path=final_output_path,
                        )
                    _sa_pivot_kn = _filtered_kn
            sa_msg = _sa_payload_round_gt0(
                objective, original_text, _sa_pivot_ks, _sa_pivot_kn,
                chosen_feature=chosen_feature,
                current_article=current_text,  # Bug-2 fix: use modified article text
                trajectory_feedback=trajectory_feedback,
            )
            resp_sa = _chat_with_retries(
                "SemanticAnalyzer", SemanticAnalyzer, sa_msg,
                retries=3, backoff_base=2.0,
                verbose=True,
            )
            if resp_sa is None:
                rr["status_pipeline"] = "SKIP_STRICT"
                rr["skip_stage"] = "SA"
                rr["skip_reason"] = "llm_failed_after_retries"
                _append_skip_and_flush(
                    df_skips=df_skips, article_id=article_id, stage="SA", reason="llm_failed_after_retries",
                    skips_path=skips_path,
                    df_reuse=df_reuse, reuse_path=reuse_path,
                    df_roundres=df_roundres, round_results_path=round_results_path,
                    panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                    panel_target_ids=panel_target_ids,
                    gen_csv_path=final_output_path,
                )
                _append_roundres_and_flush(
                    df_roundres=df_roundres, rr=rr,
                    df_skips=df_skips, skips_path=skips_path,
                    df_reuse=df_reuse, reuse_path=reuse_path,
                    round_results_path=round_results_path,
                    panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                    panel_target_ids=panel_target_ids,
                    gen_csv_path=final_output_path,
                )
                continue

            sa_obj = _parse_with_local_repair(resp_sa)
            coerced = _coerce_sa(sa_obj) if isinstance(sa_obj, dict) else None
            if coerced is None:
                ok, ks, nums = check_semantic_analyzer_format(SemanticAnalyzer)
                if ok:
                    coerced = ([str(x) for x in ks], _sanitize_numbers([str(x) for x in nums]), {})
            if coerced is None:
                rr["status_pipeline"] = "SKIP_STRICT"
                rr["skip_stage"] = "SA"
                rr["skip_reason"] = "invalid_json"
                _append_skip_and_flush(
                    df_skips=df_skips, article_id=article_id, stage="SA", reason="invalid_json",
                    skips_path=skips_path,
                    df_reuse=df_reuse, reuse_path=reuse_path,
                    df_roundres=df_roundres, round_results_path=round_results_path,
                    panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                    panel_target_ids=panel_target_ids,
                    gen_csv_path=final_output_path,
                )
                _append_roundres_and_flush(
                    df_roundres=df_roundres, rr=rr,
                    df_skips=df_skips, skips_path=skips_path,
                    df_reuse=df_reuse, reuse_path=reuse_path,
                    round_results_path=round_results_path,
                    panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                    panel_target_ids=panel_target_ids,
                    gen_csv_path=final_output_path,
                )
                continue

            new_ks, new_kn, n2s_map = coerced
            # Bug-5 fix: strip prompt-example placeholders and numbers absent from current_text
            new_kn = _sanitize_numbers_strict(new_kn, current_text)
            n2s_map = {k: v for k, v in n2s_map.items() if k in set(new_kn)}
            # Bugfix 2026-05-15: see round-0 path. Synthesize missing n2s_map
            # entries from current_text (the actively modified article).
            if len(n2s_map) < len(new_kn):
                _synth = _build_n2s_from_text(new_kn, current_text)
                _added = 0
                for _k, _v in _synth.items():
                    if _k not in n2s_map:
                        n2s_map[_k] = _v
                        _added += 1
                if _added:
                    print(f"[SA] synthesized {_added}/{len(new_kn)} n2s_map entries from current_text for ID={article_id}")

            def _norm_list(lst):
                return [re.sub(r"\s+", " ", (str(x) or "").strip()) for x in (lst or []) if str(x).strip()]

            anchors_changed = (sorted(_norm_list(prev_ks)) != sorted(_norm_list(new_ks))) or (sorted(_norm_list(prev_kn)) != sorted(_norm_list(new_kn)))

            key_sentences, key_numbers = new_ks, new_kn

        else:
            key_sentences, key_numbers = prev_ks, prev_kn
            n2s_map = prev_n2s_map
            # Realign prev anchors to current_text (mechanical, no LLM)
            aligned_ks, lost_ks = _realign_anchors_to_current_text(
                prev_ks, current_text, threshold=0.65
            )
            if aligned_ks:
                key_sentences = aligned_ks
            if lost_ks:
                _append_reuse_and_flush(
                    df_reuse=df_reuse, article_id=article_id,
                    what="SA_ANCHOR_LOST",
                    reason=f"run_SA=False -> {len(lost_ks)} anchor(s) lost in realignment: {lost_ks[:3]}",
                    df_skips=df_skips, skips_path=skips_path,
                    reuse_path=reuse_path,
                    df_roundres=df_roundres, round_results_path=round_results_path,
                    panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                    panel_target_ids=panel_target_ids,
                    gen_csv_path=final_output_path,
                )
            _append_reuse_and_flush(
                df_reuse=df_reuse, article_id=article_id, what="SA_PREV_OUTPUT",
                reason="run_SA=False -> reused SemanticAnalyzer.csv from round n-1 (with realignment)",
                df_skips=df_skips, skips_path=skips_path,
                reuse_path=reuse_path,
                df_roundres=df_roundres, round_results_path=round_results_path,
                panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                panel_target_ids=panel_target_ids,
                gen_csv_path=final_output_path,
            )

        # ---------- SSE ----------
        # 2026-05-20 channel-split: SSE works on SSE_TARGETS_0 (pivot) or on
        # sse_targets_curr = filter(key_sentences without key_numbers) (cascade).
        # SSE never sees numeric sentences. Replacements accumulate; merge to
        # pre_nam_text happens after both SSE+NM loops.
        sse_replacements_curr: List[Tuple[str, str]] = []
        sse_modified_sentences_curr: List[str] = []
        # Channel-split snapshot of current SA output (used by cascade branches).
        sse_targets_curr_round, nm_targets_curr_round = _compute_sse_nm_targets(
            key_sentences, key_numbers, n2s_map,
        )

        if run_SSE:
            # Pivot uses round-0 frozen channel; cascade uses current-round split.
            if _is_pivot("SSE", pivot):
                sse_targets_frozen = _load_sse_targets0_for_id(_output_root, article_id)
                if sse_targets_frozen:
                    sse_targets = list(sse_targets_frozen)
                else:
                    # Fallback: if SSE_TARGETS_0 missing (legacy run), recompute
                    # from current SA output. Less stable but better than nothing.
                    sse_targets = list(sse_targets_curr_round)
                    _append_reuse_and_flush(
                        df_reuse=df_reuse, article_id=article_id, what="SSE_PIVOT_FALLBACK",
                        reason="SSE_TARGETS_0 missing -> fallback to current channel-split",
                        df_skips=df_skips, skips_path=skips_path,
                        reuse_path=reuse_path,
                        df_roundres=df_roundres, round_results_path=round_results_path,
                        panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                        panel_target_ids=panel_target_ids,
                        gen_csv_path=final_output_path,
                    )
            else:
                sse_targets = list(sse_targets_curr_round)

            if (not _is_pivot("SSE", pivot)) or (len(sse_targets) == 0):
                if not sse_targets:
                    if _is_pivot("SSE", pivot):
                        # Bug-7 fix: SSE is pivot but channel is empty (no
                        # non-numeric key_sentences). Fall through — NaM/TE
                        # still run and the article gets generated.
                        pass
                    else:
                        # Cascade with empty SSE channel: nothing to do. Don't
                        # skip the whole article — fall through to NM/NaM/TE.
                        _append_reuse_and_flush(
                            df_reuse=df_reuse, article_id=article_id, what="SSE_CASCADE_EMPTY",
                            reason="cascade SSE: empty channel (no non-numeric key_sentences)",
                            df_skips=df_skips, skips_path=skips_path,
                            reuse_path=reuse_path,
                            df_roundres=df_roundres, round_results_path=round_results_path,
                            panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                            panel_target_ids=panel_target_ids,
                            gen_csv_path=final_output_path,
                        )

            _sse_snap_edits_curr: List[dict] = []
            for target_sentence in sse_targets:
                if _is_pivot("SSE", pivot):
                    # 2026-05-20 channel-split + anchor revision:
                    #   current_sentence (= _anchored_sentence): sentence in
                    #     current_text (post-NaM r-1) that best matches the
                    #     SSE_TARGETS_0 anchor. If score < 0.5 the channel
                    #     anchor is lost — skip this target silently for r.
                    #   original_sentence_anchor: sentence in original_text
                    #     (REAL round 0) cosine-matched against current_sentence
                    #     — this is what the metric SSE_semantic_drift /
                    #     lexical_stability uses in anchored mode.
                    _anchored_sentence, _align_score = _find_best_match_in_text(target_sentence, current_text)
                    if not _anchored_sentence:
                        _anchored_sentence = target_sentence
                        _align_score = 0.0
                    if _align_score < 0.5:
                        _append_reuse_and_flush(
                            df_reuse=df_reuse, article_id=article_id,
                            what="SSE_PIVOT_TARGET_LOST",
                            reason=f"sse_anchor_lost_after_nam:{_align_score:.3f}",
                            df_skips=df_skips, skips_path=skips_path,
                            reuse_path=reuse_path,
                            df_roundres=df_roundres, round_results_path=round_results_path,
                            panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                            panel_target_ids=panel_target_ids,
                            gen_csv_path=final_output_path,
                        )
                        continue
                    _original_sentence_anchor, _orig_score = _find_best_match_in_text(_anchored_sentence, original_text or "")
                    if not _original_sentence_anchor or _orig_score < 0.5:
                        # Fall back to current as best-effort if round-0 match is too weak.
                        _original_sentence_anchor = _anchored_sentence
                    sse_msg = _sse_payload_pivot(
                        objective,
                        current_sentence=_anchored_sentence,
                        original_sentence=_original_sentence_anchor,
                        round_num=int(round_idx if round_idx != -999 else 0),
                        alignment_score=float(_align_score),
                        trajectory_feedback=trajectory_feedback,
                    )
                else:
                    # Cascade SSE (after SA pivot): round0 payload, no objective
                    sse_msg = _sse_payload_round0(target_sentence)

                resp_sse = _chat_with_retries(
                    "SalientSentenceEditor", SalientSentenceEditor, sse_msg,
                    retries=3, backoff_base=2.0, verbose=True,
                )
                if resp_sse is None:
                    rr["status_pipeline"] = "SKIP_STRICT"
                    rr["skip_stage"] = "SSE"
                    rr["skip_reason"] = "llm_failed_after_retries"
                    _append_skip_and_flush(
                        df_skips=df_skips, article_id=article_id, stage="SSE", reason="llm_failed_after_retries",
                        skips_path=skips_path,
                        df_reuse=df_reuse, reuse_path=reuse_path,
                        df_roundres=df_roundres, round_results_path=round_results_path,
                        panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                        panel_target_ids=panel_target_ids,
                        gen_csv_path=final_output_path,
                    )
                    _append_roundres_and_flush(
                        df_roundres=df_roundres, rr=rr,
                        df_skips=df_skips, skips_path=skips_path,
                        df_reuse=df_reuse, reuse_path=reuse_path,
                        round_results_path=round_results_path,
                        panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                        panel_target_ids=panel_target_ids,
                        gen_csv_path=final_output_path,
                    )
                    break

                sse_obj = _parse_with_local_repair(resp_sse)
                pair = _coerce_sse_nm(sse_obj) if isinstance(sse_obj, dict) else None
                if pair is None:
                    ok, o, m = check_salient_sentence_editor_format(SalientSentenceEditor)
                    if ok:
                        pair = (str(o), str(m))
                if pair is None:
                    rr["status_pipeline"] = "SKIP_STRICT"
                    rr["skip_stage"] = "SSE"
                    rr["skip_reason"] = "invalid_json"
                    _append_skip_and_flush(
                        df_skips=df_skips, article_id=article_id, stage="SSE", reason="invalid_json",
                        skips_path=skips_path,
                        df_reuse=df_reuse, reuse_path=reuse_path,
                        df_roundres=df_roundres, round_results_path=round_results_path,
                        panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                        panel_target_ids=panel_target_ids,
                        gen_csv_path=final_output_path,
                    )
                    _append_roundres_and_flush(
                        df_roundres=df_roundres, rr=rr,
                        df_skips=df_skips, skips_path=skips_path,
                        df_reuse=df_reuse, reuse_path=reuse_path,
                        round_results_path=round_results_path,
                        panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                        panel_target_ids=panel_target_ids,
                        gen_csv_path=final_output_path,
                    )
                    break

                original_sentence, modified_sentence = pair[0], pair[1]
                if not original_sentence.strip():
                    original_sentence = target_sentence

                # 2026-05-20 channel-split: accumulate replacements, don't mutate
                # text in-place. The SSE/NM merge step at the end of this round
                # applies both channels' replacements on a single base (current_text).
                if _norm_text_loose(original_sentence) and _norm_text_loose(modified_sentence):
                    # Candidate keys for the deterministic replace step. We push
                    # several variants so _apply_channel_replacements can find a
                    # verbatim match even if SSE rewrote `original_sentence`.
                    if _is_pivot("SSE", pivot):
                        # Prefer the anchored current-text sentence first.
                        if _anchored_sentence and _anchored_sentence != original_sentence:
                            sse_replacements_curr.append((_anchored_sentence, modified_sentence))
                        # §10: sim-based alignment for pivot=SSE
                        if previous_attempt and previous_attempt.get("my_last_modified_sentence"):
                            _sim_target, _sim_score = _find_best_match_in_text(
                                previous_attempt["my_last_modified_sentence"], current_text
                            )
                            if _sim_target and _sim_target != _anchored_sentence:
                                sse_replacements_curr.append((_sim_target, modified_sentence))
                    sse_replacements_curr.append((original_sentence, modified_sentence))
                    if target_sentence != original_sentence:
                        sse_replacements_curr.append((target_sentence, modified_sentence))

                sse_modified_sentences_curr.append(modified_sentence)
                # Drift audit (channel leak: SSE should never touch numbers).
                _nio, _nim, _lost, _renamed = _realign_numbers_post_sse(original_sentence, modified_sentence, n2s_map)
                _append_sse_intermediate_exact(sse_path, article_id, original_sentence, modified_sentence,
                                              numbers_in_original=_nio, numbers_in_modified=_nim,
                                              lost_numbers=_lost, renamed_numbers=_renamed)
                if _lost or _renamed:
                    _append_agent_drift_log(
                        drift_log_path, article_id, round_idx, "SSE",
                        [{"violation_type": "channel_leak",
                          "detail": f"lost={_lost} renamed={_renamed}"}],
                        "", original_sentence, modified_sentence,
                    )
                _sse_snap_edits_curr.append({
                    "original_sentence": original_sentence,
                    "modified_sentence": modified_sentence,
                    "numbers_in_original": _nio,
                    "numbers_in_modified": _nim,
                    "lost_numbers": _lost,
                    "renamed_numbers": _renamed,
                })

            # §4 snapshot — SSE round >0
            _save_agent_snapshot(_snap_round_dir, "SSE", article_id, round_idx, _is_pivot("SSE", pivot), _sse_snap_edits_curr)

            if rr["status_pipeline"] != "OK":
                continue

        # ---------- NM ----------
        # 2026-05-20 channel-split: nm_text is finalised at the end of the NM
        # block by merging sse_replacements_curr + nm_replacements_curr on
        # current_text. Until then it stays = current_text as a safe fallback
        # (NM bypass / no-numbers paths).
        nm_text = current_text
        nm_replacements_curr: List[Tuple[str, str]] = []
        nm_modified_sentences_curr: List[str] = []
        nm_any_called = False
        nm_bypass_taken = False

        if run_NM:
            # Fix 1 — regex fallback when key_numbers is completely empty
            if not key_numbers:
                key_numbers_fresh = _extract_numbers_from_text(current_text)
                if key_numbers_fresh:
                    _warn("NM_KEY_NUMBERS_REGEX_FALLBACK",
                          "key_numbers empty from prev round; re-extracted via regex",
                          article_id=article_id, count=len(key_numbers_fresh))
                    key_numbers = key_numbers_fresh
                    _append_reuse_and_flush(
                        df_reuse=df_reuse, article_id=article_id,
                        what="NM_KEY_NUMBERS_REGEX_FALLBACK",
                        reason=(
                            f"key_numbers was empty; regex-extracted "
                            f"{len(key_numbers_fresh)} number(s) from current text"
                        ),
                        df_skips=df_skips, skips_path=skips_path,
                        reuse_path=reuse_path,
                        df_roundres=df_roundres, round_results_path=round_results_path,
                        panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                        panel_target_ids=panel_target_ids,
                        gen_csv_path=final_output_path,
                    )

                # Graceful bypass when no numbers are found in the article.
                # NM cannot run on a numberless article; fall through to NaM/TE.
                if not key_numbers:
                    nm_bypass_taken = True
                    _NUMBER_DEPENDENT_FEATURES: Set[str] = {
                        "NM_change_magnitude", "NM_edit_success_rate",
                        "SA_numeric_coverage", "SA_numeric_compression_ratio",
                    }
                    is_number_feature = chosen_feature in _NUMBER_DEPENDENT_FEATURES
                    _warn("NM_BYPASS_NO_NUMBERS",
                          "no numbers in article; NM bypassed",
                          article_id=article_id,
                          feature=chosen_feature,
                          will_mark_done=is_number_feature)
                    _append_reuse_and_flush(
                        df_reuse=df_reuse, article_id=article_id,
                        what="NM_BYPASSED_NO_NUMBERS",
                        reason=(
                            f"chosen_feature={chosen_feature}: "
                            f"no numbers in text; NM bypassed, pipeline continues"
                        ),
                        df_skips=df_skips, skips_path=skips_path,
                        reuse_path=reuse_path,
                        df_roundres=df_roundres, round_results_path=round_results_path,
                        panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                        panel_target_ids=panel_target_ids,
                        gen_csv_path=final_output_path,
                    )
                    # nm_text is finalised by the channel-merge step — fall through to NaM/TE.

            # 2026-05-20 channel-split — NM target source:
            #   - pivot=NM: iterate over NM_TARGETS_0 (= NUM2SENT_0 from
            #     baseline_references) using lineage-tracked current_number.
            #     The frozen anchor (target_number=KN_0) is the metric identity;
            #     current_number is what's actually in current_text.
            #   - cascade=NM: iterate over nm_targets_curr_round (split of the
            #     current SA output). current_number = key_number directly.
            nm_rows: List[Dict[str, Any]] = []
            if _is_pivot("NM", pivot):
                _num2sent0_pivot = _load_num2sent0_for_id(_output_root, article_id)
                _kn0_pivot = _load_kn0_for_id(_output_root, article_id)
                # Iterate over KN_0 (frozen identity universe). Resolve current_number
                # via lineage, then locate it in current_text. If missing -> skip.
                for _kn_orig_str in _kn0_pivot:
                    _kn_current = _lineage_current_number(
                        _output_root, detector_name or "", article_id,
                        _kn_orig_str, int(round_idx if round_idx != -999 else 0),
                    )
                    _round0_anchor = _num2sent0_pivot.get(_kn_orig_str, "")
                    _candidate_sents = _sentences_containing_number(current_text, _kn_current)
                    if not _candidate_sents:
                        nm_rows.append({
                            "kn_original_sa": _kn_orig_str,
                            "kn_current": _kn_current,
                            "sentence_current": "",
                            "round0_anchor_sentence": _round0_anchor,
                            "skip": True,
                            "skip_reason": "nm_target_missing_pre_nam",
                        })
                        continue
                    if len(_candidate_sents) == 1 or not _round0_anchor:
                        _sent_current = _candidate_sents[0]
                    else:
                        _best, _ = _find_best_match_in_text(_round0_anchor, " ".join(_candidate_sents))
                        _sent_current = _best if _best in _candidate_sents else _candidate_sents[0]
                    nm_rows.append({
                        "kn_original_sa": _kn_orig_str,
                        "kn_current": _kn_current,
                        "sentence_current": _sent_current,
                        "round0_anchor_sentence": _round0_anchor,
                        "skip": False,
                        "skip_reason": None,
                    })
            else:
                # Cascade NM (pivot is SA/SSE/NaM/TE): use current channel split.
                for _nm_t in nm_targets_curr_round:
                    _kn_orig_str = str(_nm_t.get("number", ""))
                    _sent_current = str(_nm_t.get("sentence", ""))
                    if not _kn_orig_str:
                        continue
                    if not _sent_current:
                        nm_rows.append({
                            "kn_original_sa": _kn_orig_str,
                            "kn_current": _kn_orig_str,
                            "sentence_current": "",
                            "round0_anchor_sentence": "",
                            "skip": True,
                            "skip_reason": "no_sentence_found",
                        })
                        continue
                    nm_rows.append({
                        "kn_original_sa": _kn_orig_str,
                        "kn_current": _kn_orig_str,
                        "sentence_current": _sent_current,
                        "round0_anchor_sentence": _sent_current,
                        "skip": False,
                        "skip_reason": None,
                    })

            _nm_snap_edits_curr: List[dict] = []
            # Stage-level skip: nothing to modify (no numbers extracted by SA, or all
            # targets unavailable — e.g. all lost_by_sse / no_sentence_found).
            _all_nm_skip = (len(nm_rows) == 0) or all(bool(r.get("skip")) for r in nm_rows)
            if _all_nm_skip and not nm_bypass_taken:
                if len(nm_rows) == 0:
                    _skip_details = "no_numbers_to_modify"
                else:
                    _reasons = sorted({str(r.get("skip_reason") or "unknown") for r in nm_rows})
                    _skip_details = f"all_targets_skipped({','.join(_reasons)})"
                if _is_pivot("NM", pivot):
                    nm_bypass_taken = True
                    rr["nm_stage_noop"] = True  # signal to orchestrator: NM produced no edit this round
                    _skip_details += "; pivot=NM bypassed (no actionable target)"
                _append_reuse_and_flush(
                    df_reuse=df_reuse, article_id=article_id, what="NM_SKIPPED",
                    reason=_skip_details,
                    df_skips=df_skips, skips_path=skips_path,
                    reuse_path=reuse_path,
                    df_roundres=df_roundres, round_results_path=round_results_path,
                    panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                    panel_target_ids=panel_target_ids,
                    gen_csv_path=final_output_path,
                )
                # Persist skip rows in NM CSV / snapshot for traceability.
                for _nm_row in nm_rows:
                    _append_nm_intermediate_exact(nm_path, article_id, _nm_row["kn_original_sa"], _nm_row["sentence_current"], "", success=False, reason=_nm_row["skip_reason"])
                    _nm_snap_edits_curr.append({
                        "key_number_original_sa": _nm_row["kn_original_sa"],
                        "key_number_after_sse": _nm_row["kn_current"],
                        "key_number_after_nm": _nm_row["kn_current"],
                        "sentence_after_sse": _nm_row["sentence_current"],
                        "sentence_after_nm": _nm_row["sentence_current"],
                    })
            else:
                for _nm_row in nm_rows:
                    if _nm_row["skip"]:
                        _append_nm_intermediate_exact(nm_path, article_id, _nm_row["kn_original_sa"], _nm_row["sentence_current"], "", success=False, reason=_nm_row["skip_reason"])
                        _nm_snap_edits_curr.append({
                            "key_number_original_sa": _nm_row["kn_original_sa"],
                            "key_number_after_sse": _nm_row["kn_current"],
                            "key_number_after_nm": _nm_row["kn_current"],
                            "sentence_after_sse": _nm_row["sentence_current"],
                            "sentence_after_nm": _nm_row["sentence_current"],
                        })
                        continue

                    _kn_orig_sa = str(_nm_row["kn_original_sa"])
                    _kn_current = str(_nm_row["kn_current"])
                    _sent_current = str(_nm_row["sentence_current"])

                    if _is_pivot("NM", pivot):
                        # 2026-05-20 channel-split — NM pivot payload:
                        #   target_number = KN_0[i] (frozen identity anchor)
                        #   current_number = lineage[-1] (what's in current_sentence,
                        #                                 = NM's most recent output)
                        #   current_sentence = pre-anchored sentence in current_text
                        # current_number already encodes "last move" — no need for a
                        # separate previous_modified_number field.
                        nm_msg = _nm_payload_pivot(
                            objective,
                            current_sentence=_sent_current,
                            target_number=_kn_orig_sa,
                            current_number=_kn_current,
                            alignment_score=1.0,
                            trajectory_feedback=trajectory_feedback,
                            chosen_feature=chosen_feature,
                        )
                    else:
                        # Cascade NM (after SA/SSE/NaM/TE pivot): round0 payload, no objective.
                        nm_msg = _nm_payload_round0(_sent_current, target_number=_kn_orig_sa)

                    resp_nm = _chat_with_retries(
                        "NumberModifier", NumberModifier, nm_msg,
                        retries=3, backoff_base=2.0, verbose=True,
                    )
                    if resp_nm is None:
                        rr["status_pipeline"] = "SKIP_STRICT"
                        rr["skip_stage"] = "NM"
                        rr["skip_reason"] = "llm_failed_after_retries"
                        _append_skip_and_flush(
                            df_skips=df_skips, article_id=article_id, stage="NM", reason="llm_failed_after_retries",
                            skips_path=skips_path,
                            df_reuse=df_reuse, reuse_path=reuse_path,
                            df_roundres=df_roundres, round_results_path=round_results_path,
                            panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                            panel_target_ids=panel_target_ids,
                            gen_csv_path=final_output_path,
                        )
                        _append_roundres_and_flush(
                            df_roundres=df_roundres, rr=rr,
                            df_skips=df_skips, skips_path=skips_path,
                            df_reuse=df_reuse, reuse_path=reuse_path,
                            round_results_path=round_results_path,
                            panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                            panel_target_ids=panel_target_ids,
                            gen_csv_path=final_output_path,
                        )
                        break

                    nm_any_called = True

                    nm_obj = _parse_with_local_repair(resp_nm)
                    pair = _coerce_sse_nm(nm_obj) if isinstance(nm_obj, dict) else None
                    if pair is None:
                        ok, o, m = check_number_modifier_format(NumberModifier)
                        if ok:
                            pair = (str(o), str(m))
                    if pair is None:
                        rr["status_pipeline"] = "SKIP_STRICT"
                        rr["skip_stage"] = "NM"
                        rr["skip_reason"] = "invalid_json"
                        _append_skip_and_flush(
                            df_skips=df_skips, article_id=article_id, stage="NM", reason="invalid_json",
                            skips_path=skips_path,
                            df_reuse=df_reuse, reuse_path=reuse_path,
                            df_roundres=df_roundres, round_results_path=round_results_path,
                            panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                            panel_target_ids=panel_target_ids,
                            gen_csv_path=final_output_path,
                        )
                        _append_roundres_and_flush(
                            df_roundres=df_roundres, rr=rr,
                            df_skips=df_skips, skips_path=skips_path,
                            df_reuse=df_reuse, reuse_path=reuse_path,
                            round_results_path=round_results_path,
                            panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                            panel_target_ids=panel_target_ids,
                            gen_csv_path=final_output_path,
                        )
                        break

                    original_sentence, modified_sentence = pair[0], pair[1]
                    if not original_sentence.strip():
                        original_sentence = _sent_current

                    # 2026-05-20 channel-split: accumulate replacements rather
                    # than mutate text in-place. Both SSE and NM channels merge
                    # onto a single base (current_text) at end of round.
                    if _norm_text_loose(original_sentence) and _norm_text_loose(modified_sentence):
                        if _is_pivot("NM", pivot):
                            # Try previous-round modified sentence first, in case the
                            # current sentence has drifted slightly across NaM.
                            prev_match = prev_nm_df[
                                prev_nm_df["target_number"].astype(str) == _kn_orig_sa
                            ] if not prev_nm_df.empty and "target_number" in prev_nm_df.columns else pd.DataFrame()
                            if not prev_match.empty:
                                _prev_mod_sent = str(prev_match.iloc[0].get("modified_sentence", "") or "")
                                if _prev_mod_sent and _prev_mod_sent != _sent_current:
                                    nm_replacements_curr.append((_prev_mod_sent, modified_sentence))
                            # §10: sim-based alignment for pivot=NM
                            if previous_attempt and previous_attempt.get("sentence_with_my_last_modification"):
                                _sim_target_nm, _sim_score_nm = _find_best_match_in_text(
                                    previous_attempt["sentence_with_my_last_modification"], current_text
                                )
                                if _sim_target_nm and _sim_target_nm != _sent_current:
                                    nm_replacements_curr.append((_sim_target_nm, modified_sentence))
                        nm_replacements_curr.append((_sent_current, modified_sentence))
                        if original_sentence != _sent_current:
                            nm_replacements_curr.append((original_sentence, modified_sentence))

                    nm_modified_sentences_curr.append(modified_sentence)
                    # target_number = round-0 KN_0 (immutable identity);
                    # modified_number extracted relative to _kn_current (= lineage[-1]),
                    # so an unchanged sentence yields "" and contributes 0 to magnitude.
                    _modified_number = _extract_number_from_sentence(modified_sentence, _kn_current)
                    _append_nm_intermediate_exact(
                        nm_path, article_id, _kn_orig_sa, original_sentence, modified_sentence,
                        success=True, modified_number=_modified_number,
                    )
                    # Fase A — NM drift audit (no rejection, just logging)
                    _nm_violations = _compute_nm_drift_violations(_kn_current, original_sentence, modified_sentence)
                    if _nm_violations:
                        _append_agent_drift_log(
                            drift_log_path, article_id, round_idx, "NM", _nm_violations,
                            _kn_current, original_sentence, modified_sentence,
                        )
                        _warn("NM_DRIFT_VIOLATION", "NM violated edit constraints",
                              article_id=article_id, target=_kn_current,
                              n_violations=len(_nm_violations),
                              types=",".join(v["violation_type"] for v in _nm_violations))
                    _nm_snap_edits_curr.append({
                        "key_number_original_sa": _kn_orig_sa,
                        "key_number_after_sse": _kn_current,
                        "key_number_after_nm": _modified_number,
                        "sentence_after_sse": _sent_current,
                        "sentence_after_nm": modified_sentence,
                    })

                # §4 snapshot — NM round >0
                _save_agent_snapshot(_snap_round_dir, "NM", article_id, round_idx, _is_pivot("NM", pivot), _nm_snap_edits_curr)

                if rr["status_pipeline"] != "OK":
                    continue

        # 2026-05-20 channel-split merge: build pre_nam_text by applying both
        # channels' replacements on a single base (current_text). SSE and NM
        # operate on disjoint sentence sets so order doesn't matter.
        _pre_nam_text, _unmatched_sse_rg0 = _apply_channel_replacements(current_text, sse_replacements_curr)
        _pre_nam_text, _unmatched_nm_rg0 = _apply_channel_replacements(_pre_nam_text, nm_replacements_curr)
        if _unmatched_sse_rg0 or _unmatched_nm_rg0:
            _warn("CHANNEL_MERGE_UNMATCHED", "some replacements not applied verbatim",
                  article_id=article_id,
                  unmatched_sse=len(_unmatched_sse_rg0),
                  unmatched_nm=len(_unmatched_nm_rg0))
        nm_text = _pre_nam_text


        # ---------- NaM ----------
        nam_input_article = nm_text  # default: cascade uses post-NM article
        nam_technique = ""  # default when NaM is skipped — propagates to CSV/snapshot below
        if run_NaM:
            # Cascade: NaM uses the most current article state (after NM).
            # Pivot NaM reads from prev_nam_article to rewrite from that baseline.
            if _is_pivot("NaM", pivot):
                nam_input_article = prev_nam_article or current_text
                # BUG-C fix: enrich previous_article from agent snapshot when CSV has no data
                if _snaps_base_dir:
                    _nam_snap = _load_agent_snapshot(_snaps_base_dir, "NaM", article_id, round_idx)
                    if _nam_snap and isinstance(_nam_snap.get("edits"), list) and _nam_snap["edits"]:
                        _snap_prev_nam = _nam_snap["edits"][0].get("modified_article", "")
                        if _snap_prev_nam:
                            prev_nam_article = _snap_prev_nam
                            nam_input_article = _snap_prev_nam
                            print(f"[NaM][PIVOT] loaded previous_article from snapshot round={_nam_snap.get('round')} ID={article_id}")
                ks_for_nam = key_sentences
                kn_for_nam = key_numbers
                if not prev_sse_df.empty:
                    ks_for_nam = prev_sse_df["modified_sentence"].astype(str).tolist()
                if str(chosen_feature or "").strip() in (
                    "NaM_contradiction_prob",
                ) and not prev_nm_df.empty:
                    kn_for_nam = prev_nm_df["target_number"].astype(str).tolist()

                nam_msg = _nam_payload_pivot(
                    objective=objective,
                    chosen_feature=chosen_feature,
                    original_article=original_text,
                    current_article=nam_input_article,
                    key_sentences=ks_for_nam,
                    key_numbers=kn_for_nam,
                    trajectory_feedback=trajectory_feedback,
                    previous_attempt=previous_attempt,
                    output_root=_output_root,
                    article_id=str(article_id),
                )
            else:
                # Cascade NaM (after SA/SSE/NM pivot): normal round0 edit on current article
                nam_msg = _nam_payload_round0(nm_text)

            resp_nam = _chat_with_retries(
                "NarrativeModifier", NarrativeModifier, nam_msg,
                retries=3, backoff_base=2.0,
                verbose=True,
            )
            if resp_nam is None:
                rr["status_pipeline"] = "SKIP_STRICT"
                rr["skip_stage"] = "NaM"
                rr["skip_reason"] = "llm_failed_after_retries"
                _append_skip_and_flush(
                    df_skips=df_skips, article_id=article_id, stage="NaM", reason="llm_failed_after_retries",
                    skips_path=skips_path,
                    df_reuse=df_reuse, reuse_path=reuse_path,
                    df_roundres=df_roundres, round_results_path=round_results_path,
                    panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                    panel_target_ids=panel_target_ids,
                    gen_csv_path=final_output_path,
                )
                _append_roundres_and_flush(
                    df_roundres=df_roundres, rr=rr,
                    df_skips=df_skips, skips_path=skips_path,
                    df_reuse=df_reuse, reuse_path=reuse_path,
                    round_results_path=round_results_path,
                    panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                    panel_target_ids=panel_target_ids,
                    gen_csv_path=final_output_path,
                )
                continue

            nam_obj = _parse_with_local_repair(resp_nam)
            _nam_coerced = _coerce_nam(nam_obj) if isinstance(nam_obj, dict) else None
            if _nam_coerced is None:
                ok, mod_article2 = check_narrative_modifier_format(NarrativeModifier)
                if ok:
                    mod_article = str(mod_article2)
                    nam_technique = ""
                else:
                    mod_article = None
                    nam_technique = ""
            else:
                mod_article, nam_technique = _nam_coerced
            if mod_article is None:
                rr["status_pipeline"] = "SKIP_STRICT"
                rr["skip_stage"] = "NaM"
                rr["skip_reason"] = "invalid_json"
                _append_skip_and_flush(
                    df_skips=df_skips, article_id=article_id, stage="NaM", reason="invalid_json",
                    skips_path=skips_path,
                    df_reuse=df_reuse, reuse_path=reuse_path,
                    df_roundres=df_roundres, round_results_path=round_results_path,
                    panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                    panel_target_ids=panel_target_ids,
                    gen_csv_path=final_output_path,
                )
                _append_roundres_and_flush(
                    df_roundres=df_roundres, rr=rr,
                    df_skips=df_skips, skips_path=skips_path,
                    df_reuse=df_reuse, reuse_path=reuse_path,
                    round_results_path=round_results_path,
                    panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                    panel_target_ids=panel_target_ids,
                    gen_csv_path=final_output_path,
                )
                continue

            # §4 snapshot — NaM round >0 (enables previous_attempt in next round)
            _save_agent_snapshot(
                _snap_round_dir, "NaM", article_id, round_idx,
                _is_pivot("NaM", pivot),
                [{"modified_article": mod_article, "technique_used": nam_technique}],
            )

        else:
            mod_article = nm_text

        # ---------- TE ----------
        if run_TE:
            if _is_pivot("TE", pivot):
                # 2026-05-20 revision — TE pivot payload no longer carries a
                # round-0 anchor or r-1 reference: the metric is pure
                # NLI(current_article, current_title) of the SAME round.
                # current_title already represents the last move (TE r-1
                # output, or the unmodified title if TE was not pivot at r-1).
                te_msg = _te_payload_pivot(
                    objective, current_title, mod_article,
                    chosen_feature=chosen_feature,
                    original_title=_te_orig_title,
                    trajectory_feedback=trajectory_feedback,
                )
            else:
                # Cascade TE (after SA/SSE/NM/NaM pivot): normal round0 edit
                te_msg = _te_payload_round0(current_title, mod_article)
            resp_te = _chat_with_retries(
                "TitleEditor", TitleEditor, te_msg,
                retries=3, backoff_base=2.0,
                verbose=True,
            )
            if resp_te is None:
                rr["status_pipeline"] = "SKIP_STRICT"
                rr["skip_stage"] = "TE"
                rr["skip_reason"] = "llm_failed_after_retries"
                _append_skip_and_flush(
                    df_skips=df_skips, article_id=article_id, stage="TE", reason="llm_failed_after_retries",
                    skips_path=skips_path,
                    df_reuse=df_reuse, reuse_path=reuse_path,
                    df_roundres=df_roundres, round_results_path=round_results_path,
                    panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                    panel_target_ids=panel_target_ids,
                    gen_csv_path=final_output_path,
                )
                _append_roundres_and_flush(
                    df_roundres=df_roundres, rr=rr,
                    df_skips=df_skips, skips_path=skips_path,
                    df_reuse=df_reuse, reuse_path=reuse_path,
                    round_results_path=round_results_path,
                    panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                    panel_target_ids=panel_target_ids,
                    gen_csv_path=final_output_path,
                )
                continue

            te_obj = _parse_with_local_repair(resp_te)
            new_title = _coerce_te(te_obj) if isinstance(te_obj, dict) else None
            if new_title is None:
                ok, t2 = check_title_editor_format(TitleEditor)
                if ok:
                    new_title = str(t2)
            if new_title is None:
                rr["status_pipeline"] = "SKIP_STRICT"
                rr["skip_stage"] = "TE"
                rr["skip_reason"] = "invalid_json"
                _append_skip_and_flush(
                    df_skips=df_skips, article_id=article_id, stage="TE", reason="invalid_json",
                    skips_path=skips_path,
                    df_reuse=df_reuse, reuse_path=reuse_path,
                    df_roundres=df_roundres, round_results_path=round_results_path,
                    panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                    panel_target_ids=panel_target_ids,
                    gen_csv_path=final_output_path,
                )
                _append_roundres_and_flush(
                    df_roundres=df_roundres, rr=rr,
                    df_skips=df_skips, skips_path=skips_path,
                    df_reuse=df_reuse, reuse_path=reuse_path,
                    round_results_path=round_results_path,
                    panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                    panel_target_ids=panel_target_ids,
                    gen_csv_path=final_output_path,
                )
                continue

            # Deterministic, direction-aware enforcement of the 3 title
            # FORMATTING style features (no-op for every other feature). Runs
            # on the validated LLM output so semantic work is preserved while
            # guaranteeing the chosen formatting feature actually moves.
            if _is_pivot("TE", pivot):
                new_title = _apply_title_style_actuator(chosen_feature, direction, new_title)

        else:
            new_title = current_title

        # ---------- write logs ----------
        ts = _now_iso()
        ridx_out = int(round_idx if round_idx != -999 else 999999)

        _safe_append_dedupe_csv(sa_path, pd.DataFrame([{
            "ID": article_id, "round_idx": ridx_out, "timestamp": ts,
            "key_sentences": json.dumps(key_sentences, ensure_ascii=False),
            "key_numbers": json.dumps(key_numbers, ensure_ascii=False),
            "number_to_sentence_map": json.dumps(n2s_map, ensure_ascii=False),
            "sse_targets": json.dumps(sse_targets_curr_round, ensure_ascii=False),
            "nm_targets": json.dumps(nm_targets_curr_round, ensure_ascii=False),
        }]), subset_cols=["ID", "round_idx"], keep="last")

        _safe_append_dedupe_csv(narr_path, pd.DataFrame([{
            "ID": article_id, "round_idx": ridx_out, "timestamp": ts,
            "input_article": nam_input_article if run_NaM else nm_text,
            "modified_article": mod_article,
            "technique_used": nam_technique,
        }]), subset_cols=["ID", "round_idx"], keep="last")

        _safe_append_dedupe_csv(title_path, pd.DataFrame([{
            "ID": article_id, "round_idx": ridx_out, "timestamp": ts,
            "original_title": current_title,
            "modified_title": new_title,
        }]), subset_cols=["ID", "round_idx"], keep="last")

        rows_generated.append({"ID": article_id, "title": new_title, "text": mod_article})

        # §10 — save article snapshot for round >0
        _save_article_snapshot(_art_snap_round_dir, article_id, mod_article)

        _safe_save_generated_subset(final_output_path, pd.DataFrame([{
            "ID": article_id,
            "title": new_title,
            "text": mod_article,
        }]), keep_ids=active_ids)

        try:
            if not df_skips.empty and "ID" in df_skips.columns:
                df_skips = df_skips[df_skips["ID"].astype(str) != str(article_id)].copy()
                df_skips.to_csv(skips_path, index=False, encoding="utf-8")
        except Exception:
            pass

        _append_roundres_and_flush(
            df_roundres=df_roundres, rr=rr,
            df_skips=df_skips, skips_path=skips_path,
            df_reuse=df_reuse, reuse_path=reuse_path,
            round_results_path=round_results_path,
            panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
            panel_target_ids=panel_target_ids,
            gen_csv_path=final_output_path,
        )

        _flush_all_after_id(
            df_skips=df_skips, skips_path=skips_path,
            df_reuse=df_reuse, reuse_path=reuse_path,
            df_roundres=df_roundres, round_results_path=round_results_path,
            panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
            panel_target_ids=panel_target_ids,
            gen_csv_path=final_output_path,
        )

        if (len(rows_generated) > 0) and (len(rows_generated) % FLUSH_EVERY_N_IDS == 0):
            _safe_save_generated_subset(final_output_path, pd.DataFrame(rows_generated), keep_ids=active_ids)
            rows_generated = []
            _flush_all_after_id(
                df_skips=df_skips, skips_path=skips_path,
                df_reuse=df_reuse, reuse_path=reuse_path,
                df_roundres=df_roundres, round_results_path=round_results_path,
                panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
                panel_target_ids=panel_target_ids,
                gen_csv_path=final_output_path,
            )

    if rows_generated:
        _safe_save_generated_subset(final_output_path, pd.DataFrame(rows_generated), keep_ids=active_ids)

    _flush_all_after_id(
        df_skips=df_skips, skips_path=skips_path,
        df_reuse=df_reuse, reuse_path=reuse_path,
        df_roundres=df_roundres, round_results_path=round_results_path,
        panel_path=panel_path, round_idx_for_panel=ridx_for_panel,
        panel_target_ids=panel_target_ids,
        gen_csv_path=final_output_path,
    )

    if os.path.exists(final_output_path):
        return pd.read_csv(final_output_path, encoding="utf-8")
    return pd.DataFrame(columns=["ID", "title", "text"])

