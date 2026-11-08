import os
import ast
import json
import re
import math
import time
from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
import evaluate
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)

# ============================================================
# Numeric config (NO new features, just robust behavior)
# ============================================================
IGNORE_YEARS_IN_SA_NUMERIC_METRICS = True
YEAR_MIN = 1900
YEAR_MAX = 2099

DEDUP_NUMBERS = True
DEDUP_DECIMALS = 6


# ============================================================
# Logging helpers
# ============================================================
def log(msg: str):
    print(msg, flush=True)

def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def sec(t0):
    return f"{(time.perf_counter() - t0):.2f}s"


# ============================================================
# Parsing helpers
# ============================================================
def safe_list(x) -> List[str]:
    if isinstance(x, list):
        return [str(z) for z in x if str(z).strip()]
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x).strip()
    if not s:
        return []
    for parser in (ast.literal_eval, json.loads):
        try:
            v = parser(s)
            if isinstance(v, (list, tuple)):
                return [str(z) for z in v if str(z).strip()]
        except Exception:
            pass
    for sep in ["||", "§", "###", ";", "|", "\n", ","]:
        if sep in s:
            parts = [t.strip() for t in s.split(sep) if t.strip()]
            if parts:
                return parts
    return [s]

def first_nonempty(*vals, default=""):
    for v in vals:
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        s = str(v)
        if s.strip():
            return s
    return default


# ============================================================
# Sentence splitting (lightweight)
# ============================================================
_SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+")

def split_sentences(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    sents = _SENT_SPLIT_RE.split(text.strip())
    sents = [s.strip() for s in sents if len(s.strip()) >= 3]
    return sents


# ============================================================
# Numeric extraction & normalization (ROBUST)
# ============================================================
_NUM_TOKEN_RE = re.compile(
    r"(?<!\w)[+\-]?\d{1,3}(?:[.,\s]?\d{3})*(?:[.,]\d+)?(?:[eE][+\-]?\d+)?%?(?!\w)"
)

_THOUSAND_COMMA_RE = re.compile(r"^[+\-]?\d{1,3}(?:,\d{3})+(?:%?)$")
_THOUSAND_DOT_RE   = re.compile(r"^[+\-]?\d{1,3}(?:\.\d{3})+(?:%?)$")
_DECIMAL_COMMA_RE  = re.compile(r"^[+\-]?\d+,\d+(?:%?)$")
_DECIMAL_DOT_RE    = re.compile(r"^[+\-]?\d+\.\d+(?:%?)$")

def _normalize_numeric_token(tok: str) -> Optional[float]:
    if not tok:
        return None

    s = tok.strip().replace("\u00A0", " ").strip()
    if not s:
        return None

    s = s.replace(" ", "")

    perc = s.endswith("%")
    if perc:
        s = s[:-1]

    if ("e" in s.lower()) and any(ch.isdigit() for ch in s):
        try:
            return float(s)
        except Exception:
            return None

    if "," in s and "." in s:
        last_comma, last_dot = s.rfind(","), s.rfind(".")
        dec = "," if last_comma > last_dot else "."
        thou = "." if dec == "," else ","
        s = s.replace(thou, "").replace(dec, ".")
    elif "," in s:
        if _THOUSAND_COMMA_RE.match(s):
            s = s.replace(",", "")
        elif _DECIMAL_COMMA_RE.match(s):
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "") if s.count(",") > 1 else s.replace(",", ".")
    elif "." in s:
        if _THOUSAND_DOT_RE.match(s):
            s = s.replace(".", "")
        else:
            pass

    try:
        v = float(s)
        if math.isinf(v) or math.isnan(v):
            return None
        return v
    except Exception:
        return None

def _is_year(v: float, year_min=YEAR_MIN, year_max=YEAR_MAX) -> bool:
    if v is None or math.isnan(v) or math.isinf(v):
        return False
    iv = int(round(v))
    return abs(v - iv) < 1e-9 and (year_min <= iv <= year_max)

def _dedupe_numbers(nums: List[float], decimals: int = DEDUP_DECIMALS) -> List[float]:
    if not nums:
        return []
    seen = set()
    out = []
    for v in nums:
        if v is None or math.isnan(v) or math.isinf(v):
            continue
        key = round(float(v), decimals)
        if key in seen:
            continue
        seen.add(key)
        out.append(float(v))
    return out

def extract_numbers(text: str, ignore_years: bool = False) -> List[float]:
    if not isinstance(text, str) or not text.strip():
        return []
    nums: List[float] = []
    for m in _NUM_TOKEN_RE.finditer(text):
        v = _normalize_numeric_token(m.group(0))
        if v is None:
            continue
        if ignore_years and _is_year(v):
            continue
        nums.append(float(v))
    if DEDUP_NUMBERS:
        nums = _dedupe_numbers(nums)
    return nums

def _greedy_match_indices(pred_nums, ref_nums, rel_tol=0.01, abs_tol=1e-6) -> Dict[int, int]:
    if not pred_nums or not ref_nums:
        return {}
    ref_used = [False] * len(ref_nums)
    matches = {}
    for i, p in enumerate(pred_nums):
        best_j, best_err = None, None
        for j, r in enumerate(ref_nums):
            if ref_used[j]:
                continue
            err = abs(p - r)
            if best_err is None or err < best_err:
                best_err, best_j = err, j
        if best_j is not None:
            r = ref_nums[best_j]
            thresh = max(rel_tol * max(abs(r), 1e-12), abs_tol)
            if best_err <= thresh:
                matches[i] = best_j
                ref_used[best_j] = True
    return matches

def num_recall(pred_nums, ref_nums, rel_tol=0.01, abs_tol=1e-6) -> float:
    if len(ref_nums) == 0:
        return 1.0
    if len(pred_nums) == 0:
        return 0.0
    matches = _greedy_match_indices(pred_nums, ref_nums, rel_tol, abs_tol)
    return len(set(matches.values())) / len(ref_nums)


# ============================================================
# NLI scorer
# ============================================================
class NLIScorer:
    def __init__(self, model_name="textattack/roberta-base-MNLI", device=None, max_len=384, fp16=True):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.fp16 = fp16 and (self.device == "cuda")

        log(f"[NLI] loading {model_name} on {self.device} (fp16={self.fp16})")
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        log(f"[NLI] loaded in {sec(t0)}")

        self.contrad_idx = 0
        self.neutral_idx = 1
        self.entail_idx = 2

    @torch.no_grad()
    def batch_probs(self, pairs: List[Tuple[str, str]], batch_size: int = 32, show_progress: bool = True) -> List[Dict[str, float]]:
        out = []
        rng = range(0, len(pairs), batch_size)
        it = tqdm(rng, desc="NLI batch", disable=not show_progress)
        for i in it:
            chunk = pairs[i:i+batch_size]
            premises = [p for p, _ in chunk]
            hyps = [h for _, h in chunk]
            tok = self.tokenizer(
                premises, hyps,
                truncation=True,
                max_length=self.max_len,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            if self.fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(**tok).logits
            else:
                logits = self.model(**tok).logits

            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            for p in probs:
                out.append({
                    "contradiction": float(p[self.contrad_idx]),
                    "neutral": float(p[self.neutral_idx]),
                    "entailment": float(p[self.entail_idx]),
                })
        return out


# ============================================================
# Fluency / Perplexity (PER-EXAMPLE FIXED)
# ============================================================
class PerplexityScorer:
    def __init__(self, model_name="distilgpt2", device=None, max_len=256):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        log(f"[PPL] loading {model_name} on {self.device}")
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        log(f"[PPL] loaded in {sec(t0)}")

    @torch.no_grad()
    def batch_perplexity(self, texts: List[str], batch_size=16, show_progress=True) -> List[float]:
        out: List[float] = []
        rng = range(0, len(texts), batch_size)
        it = tqdm(rng, desc="PPL batch", disable=not show_progress)

        for i in it:
            chunk = texts[i:i+batch_size]
            tok = self.tokenizer(
                chunk,
                truncation=True,
                max_length=self.max_len,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            input_ids = tok["input_ids"]
            attn = tok["attention_mask"]

            outputs = self.model(input_ids=input_ids, attention_mask=attn)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_attn = attn[:, 1:].contiguous()

            vocab = shift_logits.size(-1)
            loss_tok = F.cross_entropy(
                shift_logits.view(-1, vocab),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_labels.size())

            loss_tok = loss_tok * shift_attn
            tok_counts = shift_attn.sum(dim=1).clamp(min=1)
            nll = loss_tok.sum(dim=1) / tok_counts

            ppl = torch.exp(nll).detach().cpu().numpy().tolist()
            out.extend([float(p) for p in ppl])

        return out


# ============================================================
# Metric building blocks
# ============================================================
def token_count(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(text.split())

def mean_cosine_distance_from_pairs(emb_model: SentenceTransformer, pairs: List[Tuple[str, str]], batch_size=32) -> float:
    if not pairs:
        return np.nan
    a = [o if o is not None else "" for o, _ in pairs]
    b = [m if m is not None else "" for _, m in pairs]
    ea = emb_model.encode(a, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    eb = emb_model.encode(b, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    sims = np.sum(ea * eb, axis=1)
    dists = 1.0 - sims
    return float(np.mean(dists))

def semantic_coverage_sentence_level(
    emb_model: SentenceTransformer,
    original_text: str,
    key_sentences: List[str],
    batch_size=64
) -> float:
    sents = split_sentences(original_text)
    ks = [s for s in key_sentences if isinstance(s, str) and s.strip()]
    if not sents or not ks:
        return np.nan

    emb_s = emb_model.encode(sents, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    emb_k = emb_model.encode(ks, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    sim = cosine_similarity(emb_s, emb_k)
    max_sim = sim.max(axis=1)
    return float(np.mean(max_sim))

def _norm_edit_distance(a: str, b: str) -> float:
    if not isinstance(a, str):
        a = "" if a is None else str(a)
    if not isinstance(b, str):
        b = "" if b is None else str(b)
    if not a and not b:
        return 0.0
    return float(1.0 - SequenceMatcher(None, a, b).ratio())

def nm_text_edit_success_rate(nm_rows: List[Dict[str, str]],
                              kn0: Optional[List[str]] = None) -> float:
    """NM edit success rate (2026-05-20 revision).

    If `kn0` (round-0 key_numbers from baseline_references.json) is provided,
    the denominator is FIXED at |KN_0| — independent of how many SA extracted
    this round, of NaM-induced losses, or of skips. The numerator counts
    distinct KN_0 values that have at least one nm_row with success=True
    AND a non-empty modified_number (or modified_sentence != original_sentence
    as fallback) for the same target_number.

    If `kn0` is None or empty, falls back to the legacy intra-round formula
    (successi / |nm_rows considered|) for backward compatibility with code
    paths that haven't been updated.
    """
    # ---- New formula: anchored to KN_0 ----
    if kn0:
        denom = sum(1 for x in kn0 if str(x).strip())
        if denom == 0:
            return np.nan
        # Build a dict { target_number: success_bool } from nm_rows.
        tn_success: Dict[str, bool] = {}
        for r in nm_rows or []:
            tn_raw = r.get("target_number") or r.get("original_number") or ""
            tn = str(tn_raw).strip()
            if not tn:
                continue
            ok = False
            if r.get("has_success_field"):
                ok = bool(r.get("success"))
            else:
                o = (r.get("original") or "").strip()
                m = (r.get("modified") or "").strip()
                ok = bool(o and m and o != m)
            # Extra guard: a row marked success=True but with an empty
            # modified_number / unchanged sentence is not a real success.
            mn = str(r.get("modified_number") or "").strip()
            if ok and "modified_number" in r and not mn:
                ok = False
            # OR over multiple rows for same target_number (any success counts).
            tn_success[tn] = tn_success.get(tn, False) or ok
        successes = sum(1 for kn in kn0 if str(kn).strip() and tn_success.get(str(kn).strip(), False))
        return float(successes / denom)

    # ---- Legacy fallback (no KN_0 provided) ----
    if not nm_rows:
        return np.nan
    total = 0
    successes = 0
    for r in nm_rows:
        o = (r.get("original") or "").strip()
        m = (r.get("modified") or "").strip()
        if not o and not m and not r.get("has_success_field"):
            continue
        total += 1
        if r.get("has_success_field"):
            successes += 1 if r.get("success") else 0
        else:
            successes += 1 if o != m else 0
    if total == 0:
        return np.nan
    return float(successes / total)

def nm_text_change_magnitude(nm_rows: List[Dict[str, str]]) -> float:
    """NM change magnitude (2026-05-20 revision).

    Skips rows that represent NO real edit attempt:
      - success=False rows (lost_by_sse, anchor_lost, target_missing) — would
        otherwise contribute max edit-distance (modified="") and inflate the
        metric for cases where NM never had a chance to act.
      - rows where both original and modified are empty.

    For valid attempts, computes normalised edit-distance between
    `original_number` and `modified_number` (when CSV has those columns) or
    between the sentence pair (legacy fallback). Mean across attempts.
    """
    if not nm_rows:
        return np.nan
    dists = []
    for r in nm_rows:
        if r.get("has_success_field") and not r.get("success"):
            continue
        o = (r.get("original") or "").strip()
        m = (r.get("modified") or "").strip()
        if not o and not m:
            continue
        dists.append(_norm_edit_distance(o, m))
    if not dists:
        return np.nan
    return float(np.mean(dists))


# ============================================================
# Residualization helpers (coverage ~ compression)
# Split into fit / apply so the regression parameters fitted at round 0
# can be persisted and re-applied unchanged in later rounds.
# ============================================================
def _fit_residualizer(y: np.ndarray, x: np.ndarray) -> Tuple[float, float, bool]:
    """
    Fit y = a*x + b on finite (y, x) pairs.
    Returns (a, b, fitted). fitted=False signals the caller to skip subtraction
    (mask too small, x near-constant, or polyfit failed).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 5:
        return 0.0, 0.0, False
    if np.nanstd(x[mask]) < 1e-12:
        return 0.0, 0.0, False
    try:
        a, b = np.polyfit(x[mask], y[mask], 1)
        return float(a), float(b), True
    except Exception:
        return 0.0, 0.0, False


def _apply_residualizer(y: np.ndarray, x: np.ndarray, a: float, b: float, fitted: bool) -> np.ndarray:
    """
    Apply pre-fitted (a, b) to (y, x). Mirrors the fallback semantics of the
    original _residualize: if not fitted, return y on finite (y, x) positions
    and NaN elsewhere; if fitted, return y - (a*x + b) on finite positions.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.full_like(y, np.nan, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    if not fitted:
        out[mask] = y[mask]
        return out
    yhat = a * x + b
    out[mask] = y[mask] - yhat[mask]
    return out


def _residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Backward-compatible: fit and apply in one step (no persistence)."""
    a, b, fitted = _fit_residualizer(y, x)
    return _apply_residualizer(y, x, a, b, fitted)


def _load_residualizer(path: str) -> Optional[Dict[str, Dict[str, Any]]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"[RESIDUALIZER][WARN] failed to load {path}: {e}")
        return None


def _save_residualizer(
    path: str,
    sem: Tuple[float, float, bool],
    num: Tuple[float, float, bool],
    n_samples: int,
) -> None:
    payload = {
        "sem": {"a": float(sem[0]), "b": float(sem[1]), "fitted": bool(sem[2])},
        "num": {"a": float(num[0]), "b": float(num[1]), "fitted": bool(num[2])},
        "fitted_on_n_samples": int(n_samples),
        "fitted_at_round": 0,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ============================================================
# Baseline references writer (round 0 freeze of SA outputs)
# ============================================================
def write_baseline_references(
    sa_csv_path: str,
    df_orig: pd.DataFrame,
    out_path: str,
    overwrite: bool = False,
) -> bool:
    """
    Build Output/meta_frozen/baseline_references.json from the round-0
    SemanticAnalyzer.csv and the dataset DataFrame.

    Per-ID payload: {"KS_0": [...], "KN_0": [...], "title_orig": "...", "NUM2SENT_0": {...}}.

    Idempotent: if `out_path` exists and `overwrite=False`, returns False
    without touching the file (consistent with --resume semantics).

    Returns True iff the file was written by this call.
    """
    if not overwrite and os.path.exists(out_path):
        log(f"[BASELINE_REFS] found {out_path} -> skip")
        return False

    if not os.path.exists(sa_csv_path):
        log(f"[BASELINE_REFS][WARN] SA csv missing: {sa_csv_path} -> cannot build references")
        return False

    sa = pd.read_csv(sa_csv_path, encoding="utf-8")
    if "ID" not in sa.columns:
        log(f"[BASELINE_REFS][WARN] SA csv has no 'ID' column: {sa_csv_path}")
        return False
    sa["ID"] = sa["ID"].astype(str)

    df = df_orig.copy()
    if "ID" not in df.columns:
        log("[BASELINE_REFS][WARN] df_orig has no 'ID' column")
        return False
    df["ID"] = df["ID"].astype(str)

    title_col = "title" if "title" in df.columns else ("original_title" if "original_title" in df.columns else None)
    title_by_id: Dict[str, str] = {}
    if title_col is not None:
        title_by_id = {str(r["ID"]): str(r.get(title_col, "") or "") for _, r in df.iterrows()}

    payload: Dict[str, Dict[str, Any]] = {}
    for _, r in sa.iterrows():
        article_id = str(r["ID"])
        ks0 = safe_list(r.get("key_sentences", ""))
        kn0_raw = first_nonempty(r.get("key_numbers", None), r.get("numbers", None), default="")
        kn0 = safe_list(kn0_raw)
        raw_n2s = r.get("number_to_sentence_map", None) if "number_to_sentence_map" in sa.columns else None
        try:
            num2sent_0 = json.loads(raw_n2s or "{}") if raw_n2s is not None else {}
            if not isinstance(num2sent_0, dict):
                num2sent_0 = {}
        except Exception:
            num2sent_0 = {}

        # 2026-05-20 channel-split: persist SSE_TARGETS_0 and NM_TARGETS_0.
        # Prefer the columns written by the SA writer; fall back to deriving
        # them from KS_0/KN_0/NUM2SENT_0 for legacy round-0 CSVs.
        raw_sse_t0 = r.get("sse_targets", None) if "sse_targets" in sa.columns else None
        try:
            sse_targets_0 = json.loads(raw_sse_t0) if raw_sse_t0 else []
        except Exception:
            sse_targets_0 = []
        if not sse_targets_0:
            _nums_low = [re.sub(r"\s+", " ", str(n)).strip() for n in kn0 if str(n).strip()]
            for _s in ks0:
                _s_norm = re.sub(r"\s+", " ", str(_s)).strip()
                if not _s_norm:
                    continue
                _has_num = any(_n and _n in _s_norm for _n in _nums_low)
                if not _has_num:
                    sse_targets_0.append(_s_norm)

        raw_nm_t0 = r.get("nm_targets", None) if "nm_targets" in sa.columns else None
        try:
            nm_targets_0 = json.loads(raw_nm_t0) if raw_nm_t0 else []
        except Exception:
            nm_targets_0 = []
        if not nm_targets_0 and num2sent_0:
            nm_targets_0 = [{"number": str(k), "sentence": str(v)}
                            for k, v in num2sent_0.items() if str(v).strip()]

        payload[article_id] = {
            "KS_0": [str(s) for s in ks0],
            "KN_0": [str(s) for s in kn0],
            "title_orig": title_by_id.get(article_id, ""),
            "NUM2SENT_0": {str(k): str(v) for k, v in num2sent_0.items()},
            "SSE_TARGETS_0": [str(s) for s in sse_targets_0 if str(s).strip()],
            "NM_TARGETS_0": [
                {"number": str(t.get("number", "")), "sentence": str(t.get("sentence", ""))}
                for t in nm_targets_0 if isinstance(t, dict) and t.get("number")
            ],
        }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log(f"[BASELINE_REFS] wrote {out_path} (n_ids={len(payload)})")
    return True


# ============================================================
# Style vector (literature-grounded stylometric features)
# Horne & Adali 2017 (ICWSM); Chakraborty 2016; Potthast 2018; ISOT.
# Each registered feature is the per-dimension SIGNED deviation
# current - original (title for TE, body for NaM). The helpers below
# return the per-text raw style value; pure regex + optional nltk POS.
# ============================================================
_STYLE_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_STYLE_ALPHA_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_STYLE_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")
_STYLE_QUOTE_CHARS = "\"'“”‘’«»"
_STYLE_PUNCT_CHARS = set(".,;:!?\"'()[]{}-—–…“”‘’")

# POS tagging is optional: if the nltk tagger is unavailable (e.g. not
# pre-downloaded on an offline node) the two POS features degrade to NaN
# and get imputed downstream — never a hard failure.
_POS_AVAILABLE: Optional[bool] = None
_POS_CACHE: Dict[str, Any] = {}

def _style_pos_tags(text: str):
    """(token, POS) list via nltk averaged_perceptron_tagger; None if unavailable.
    Cached per-text so the fixed REAL anchor is not re-tagged every round."""
    global _POS_AVAILABLE
    t = str(text or "")
    if not t.strip() or _POS_AVAILABLE is False:
        return None
    if t in _POS_CACHE:
        return _POS_CACHE[t]
    try:
        import nltk
        from nltk import pos_tag, word_tokenize
        tags = pos_tag(word_tokenize(t))
        _POS_AVAILABLE = True
    except Exception:
        _POS_AVAILABLE = False
        tags = None
    if len(_POS_CACHE) < 8192:
        _POS_CACHE[t] = tags
    return tags

def _style_count_syllables(word: str) -> int:
    w = word.lower()
    n = len(_STYLE_VOWEL_GROUP_RE.findall(w))
    if w.endswith("e") and n > 1:
        n -= 1
    return max(n, 1)

def _style_title_raw(title: str) -> Dict[str, float]:
    t = str(title or "").strip()
    toks = _STYLE_ALPHA_TOKEN_RE.findall(t)
    words = _STYLE_WORD_RE.findall(t)
    n = len(toks)
    if n == 0:
        return {"titlecase": np.nan, "allcaps": np.nan, "exclquest": np.nan,
                "title_len": float(len(words)), "nnp": np.nan}
    cap = sum(1 for w in toks if w[:1].isupper())
    allcaps = sum(1 for w in toks if len(w) >= 2 and w.isupper())
    nq = t.count("!") + t.count("?")
    out = {
        "titlecase": cap / n,
        "allcaps": allcaps / n,
        "exclquest": nq / max(len(words), 1),
        "title_len": float(len(words)),
        "nnp": np.nan,
    }
    tags = _style_pos_tags(t)
    if tags:
        alpha = [tg for w, tg in tags if any(c.isalpha() for c in w)]
        if alpha:
            out["nnp"] = sum(1 for tg in alpha if tg in ("NNP", "NNPS")) / len(alpha)
    return out

def _style_body_raw(text: str) -> Dict[str, float]:
    t = str(text or "").strip()
    words = _STYLE_WORD_RE.findall(t)
    nw = len(words)
    if nw == 0:
        return {"fk_grade": np.nan, "ttr": np.nan, "quote_rate": np.nan,
                "punct_rate": np.nan, "adjadv": np.nan}
    alpha_words = [w for w in words if w.isalpha()]
    n_alpha = max(len(alpha_words), 1)
    ns = max(len(split_sentences(t)), 1)
    syll = sum(_style_count_syllables(w) for w in alpha_words)
    fk = 0.39 * (nw / ns) + 11.8 * (syll / n_alpha) - 15.59
    ttr = len(set(w.lower() for w in alpha_words)) / n_alpha
    n_quotes = sum(t.count(c) for c in _STYLE_QUOTE_CHARS)
    n_punct = sum(1 for ch in t if ch in _STYLE_PUNCT_CHARS)
    out = {
        "fk_grade": float(fk),
        "ttr": float(ttr),
        "quote_rate": n_quotes / nw,
        "punct_rate": n_punct / nw,
        "adjadv": np.nan,
    }
    tags = _style_pos_tags(t)
    if tags:
        alpha = [tg for w, tg in tags if any(c.isalpha() for c in w)]
        if alpha:
            out["adjadv"] = sum(1 for tg in alpha if tg.startswith("JJ") or tg.startswith("RB")) / len(alpha)
    return out

def _style_signed_deviation(cur: Dict[str, float], orig: Dict[str, float], keys) -> Dict[str, float]:
    res: Dict[str, float] = {}
    for k in keys:
        cv = cur.get(k); ov = orig.get(k)
        if (cv is None or ov is None
                or (isinstance(cv, float) and (math.isnan(cv) or math.isinf(cv)))
                or (isinstance(ov, float) and (math.isnan(ov) or math.isinf(ov)))):
            res[k] = np.nan
        else:
            res[k] = float(cv) - float(ov)
    return res

# Raw style key -> registry metric_id (the registered feature is the
# SIGNED deviation current - original on that dimension).
_TE_STYLE_KEYS = {
    "titlecase": "TE_style_titlecase",
    "allcaps": "TE_style_allcaps",
    "exclquest": "TE_style_exclquest",
    "title_len": "TE_style_title_len",
    "nnp": "TE_style_nnp",
}
_NAM_STYLE_KEYS = {
    "fk_grade": "NaM_style_fk_grade",
    "ttr": "NaM_style_ttr",
    "quote_rate": "NaM_style_quote_rate",
    "punct_rate": "NaM_style_punct_rate",
    "adjadv": "NaM_style_adjadv",
}

def _style_features_title(cur_title: str, orig_title: str) -> Dict[str, float]:
    dev = _style_signed_deviation(_style_title_raw(cur_title), _style_title_raw(orig_title), _TE_STYLE_KEYS.keys())
    return {_TE_STYLE_KEYS[k]: v for k, v in dev.items()}

def _style_features_body(cur_body: str, orig_body: str) -> Dict[str, float]:
    dev = _style_signed_deviation(_style_body_raw(cur_body), _style_body_raw(orig_body), _NAM_STYLE_KEYS.keys())
    return {_NAM_STYLE_KEYS[k]: v for k, v in dev.items()}


# ============================================================
# Registry (NO direction here)
# ============================================================
@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    agent_id: str
    description: str
    raw_to_feature: Callable[[float], float]

def _identity(x: float) -> float:
    return float(x)

def _log1p_safe(x: float) -> float:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return float("nan")
    return float(np.log1p(max(float(x), 0.0)))

METRICS_REGISTRY: Dict[str, MetricSpec] = {
    "SA_semantic_coverage": MetricSpec("SA_semantic_coverage", "SA", "Sentence-level semantic coverage (RESIDUALIZED vs compression)", _identity),
    "SA_text_compression_ratio": MetricSpec("SA_text_compression_ratio", "SA", "Tokens(key_sentences)/Tokens(original)", _identity),
    "SA_numeric_coverage": MetricSpec("SA_numeric_coverage", "SA", "Numeric recall (RESIDUALIZED vs numeric compression)", _identity),
    "SA_numeric_compression_ratio": MetricSpec("SA_numeric_compression_ratio", "SA", "Count(key_numbers)/Count(original_numbers)", _identity),

    "SSE_semantic_drift": MetricSpec("SSE_semantic_drift", "SSE", "Mean cosine distance (orig_sentence, mod_sentence)", _identity),
    "SSE_lexical_stability": MetricSpec("SSE_lexical_stability", "SSE", "Mean ROUGE-L between orig and modified salient sentences", _identity),
    "SSE_fluency_ppl": MetricSpec("SSE_fluency_ppl", "SSE", "Perplexity on modified salient sentences (per-example)", _log1p_safe),

    "NM_change_magnitude": MetricSpec("NM_change_magnitude", "NM", "Mean normalized edit distance (original_number vs modified_number)", _identity),
    "NM_edit_success_rate": MetricSpec("NM_edit_success_rate", "NM", "Rate of changed sentences (modified_number != original_number)", _identity),

    "NaM_rewrite_distance": MetricSpec("NaM_rewrite_distance", "NaM", "1 - cosine_similarity(emb(original_text), emb(modified_article))", _identity),
    "NaM_contradiction_prob": MetricSpec("NaM_contradiction_prob", "NaM", "NLI contradiction prob between (modified_article, key_sentences)", _identity),

    "TE_title_entailment": MetricSpec("TE_title_entailment", "TE", "NLI entailment prob between (article_text, modified_title)", _identity),
    "TE_title_contradiction_prob": MetricSpec("TE_title_contradiction_prob", "TE", "NLI contradiction prob between (article_text, modified_title)", _identity),

    # ---- Style vector — TITLE (TE): signed deviation current_title - original_title ----
    "TE_style_titlecase": MetricSpec("TE_style_titlecase", "TE", "Title-case ratio deviation (capitalized words / words) vs original title", _identity),
    "TE_style_allcaps": MetricSpec("TE_style_allcaps", "TE", "ALL-CAPS token-rate deviation vs original title", _identity),
    "TE_style_exclquest": MetricSpec("TE_style_exclquest", "TE", "Exclamation/question per-word rate deviation vs original title", _identity),
    "TE_style_title_len": MetricSpec("TE_style_title_len", "TE", "Title length (words) deviation vs original title", _identity),
    "TE_style_nnp": MetricSpec("TE_style_nnp", "TE", "Proper-noun (NNP) rate deviation vs original title", _identity),

    # ---- Style vector — BODY (NaM): signed deviation current_article - original_article ----
    "NaM_style_fk_grade": MetricSpec("NaM_style_fk_grade", "NaM", "Flesch-Kincaid grade deviation vs original article", _identity),
    "NaM_style_ttr": MetricSpec("NaM_style_ttr", "NaM", "Type-token ratio (lexical diversity) deviation vs original article", _identity),
    "NaM_style_quote_rate": MetricSpec("NaM_style_quote_rate", "NaM", "Quote-mark per-word rate deviation vs original article", _identity),
    "NaM_style_punct_rate": MetricSpec("NaM_style_punct_rate", "NaM", "Total punctuation per-word rate deviation vs original article", _identity),
    "NaM_style_adjadv": MetricSpec("NaM_style_adjadv", "NaM", "Adjective+adverb density deviation vs original article", _identity),
}


# ============================================================
# MAIN: compute_agent_metrics
# ============================================================
def compute_agent_metrics(
    df_orig: pd.DataFrame,
    base_out_dir: str,
    ids_mode: str = "csv_in_dataset",
    out_csv_name: str = "agent_metrics.csv",

    only_ids: Optional[List[str]] = None,
    append_if_exists: bool = False,

    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    nli_model_name: str = "textattack/roberta-base-MNLI",
    nli_max_len: int = 384,

    compute_fluency: bool = True,
    ppl_model_name: str = "distilgpt2",

    output_mode: str = "feat",
    include_applicability_flags: bool = True,

    residualizer_path: Optional[str] = None,
    baseline_references_path: Optional[str] = None,
):
    try:
        os.makedirs(base_out_dir, exist_ok=True)
    except FileExistsError:
        # FileExistsError spurio di makedirs su Lustre sotto carico:
        # se e' gia' una directory e' un successo, rilanciamo solo su vero non-dir.
        if not os.path.isdir(base_out_dir):
            raise
    out_csv = os.path.join(base_out_dir, out_csv_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[{now()}] device={device}")
    if device == "cuda":
        log(f"[{now()}] GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    log(f"[{now()}] Loading ROUGE metric…")
    t0 = time.perf_counter()
    rouge_metric = evaluate.load("rouge")
    log(f"[OK] ROUGE loaded in {sec(t0)}")

    df_orig = df_orig.copy()
    if "ID" not in df_orig.columns:
        raise ValueError("df_orig must contain column 'ID'")
    df_orig["ID"] = df_orig["ID"].astype(str)

    if "text" not in df_orig.columns and "original_text" in df_orig.columns:
        df_orig["text"] = df_orig["original_text"]
    if "title" not in df_orig.columns and "original_title" in df_orig.columns:
        df_orig["title"] = df_orig["original_title"]

    text_by_id = dict(zip(df_orig["ID"], df_orig.get("text", pd.Series([""] * len(df_orig)))))
    title_by_id = dict(zip(df_orig["ID"], df_orig.get("title", pd.Series([""] * len(df_orig)))))

    def read_csv_safe(path, default_cols):
        if os.path.exists(path):
            return pd.read_csv(path, encoding="utf-8")
        return pd.DataFrame(columns=default_cols)

    sa = read_csv_safe(os.path.join(base_out_dir, "SemanticAnalyzer.csv"), ["ID", "key_sentences", "numbers", "key_numbers", "number_to_sentence_map"])
    sse = read_csv_safe(os.path.join(base_out_dir, "SalientSentenceEditor.csv"), ["ID", "original_sentence", "modified_sentence", "numbers_in_original", "numbers_in_modified", "lost_numbers", "renamed_numbers"])
    nm  = read_csv_safe(
        os.path.join(base_out_dir, "NumberModifier.csv"),
        ["ID", "original_sentence", "modified_sentence", "target_number", "original_number", "modified_number"]
    )
    nam = read_csv_safe(os.path.join(base_out_dir, "NarrativeModifier.csv"), ["ID", "original_text", "modified_text", "modified_article"])
    te  = read_csv_safe(os.path.join(base_out_dir, "TitleEditor.csv"), ["ID", "original_title", "modified_title", "assigned_title"])

    _use_anchored_refs = (
        baseline_references_path is not None
        and os.path.exists(baseline_references_path)
    )
    if baseline_references_path is not None and not _use_anchored_refs:
        log(f"[WARN] baseline_references_path set but not found: {baseline_references_path} → legacy metric computation")

    _baseline_refs: Dict[str, Any] = {}
    _gfn_text_by_id: Dict[str, str] = {}
    _gfn_title_by_id: Dict[str, str] = {}
    _use_anchored_nam_te = False
    if _use_anchored_refs:
        with open(baseline_references_path, "r", encoding="utf-8") as _f:
            _baseline_refs = json.load(_f)
        _gfn_path = os.path.join(base_out_dir, "Generated_Fake_News.csv")
        if os.path.exists(_gfn_path):
            try:
                _gfn_df = pd.read_csv(_gfn_path, encoding="utf-8")
                _gfn_df["ID"] = _gfn_df["ID"].astype(str)
                if "text" in _gfn_df.columns:
                    _gfn_text_by_id = dict(zip(_gfn_df["ID"], _gfn_df["text"].fillna("")))
                if "title" in _gfn_df.columns:
                    _gfn_title_by_id = dict(zip(_gfn_df["ID"], _gfn_df["title"].fillna("")))
                _use_anchored_nam_te = True
            except Exception as _e:
                log(f"[WARN] Cannot load Generated_Fake_News.csv ({_e}) → NaM/TE legacy computation")
        else:
            log(f"[WARN] Generated_Fake_News.csv not found ({_gfn_path}) → NaM/TE legacy computation")

    for d in (sa, sse, nm, nam, te):
        if not d.empty and "ID" in d.columns:
            d["ID"] = d["ID"].astype(str)

    sa_key_sents = {r["ID"]: safe_list(r.get("key_sentences", "")) for _, r in sa.iterrows()}
    sa_key_nums  = {
        r["ID"]: safe_list(first_nonempty(r.get("numbers", None), r.get("key_numbers", None), default=""))
        for _, r in sa.iterrows()
    }

    if not sse.empty and {"original_sentence", "modified_sentence"}.issubset(sse.columns):
        sse_by_id = (
            sse.groupby("ID")[["original_sentence", "modified_sentence"]]
            .apply(lambda g: [(str(o or ""), str(m or "")) for o, m in zip(g["original_sentence"], g["modified_sentence"])
                              if str(o or "").strip() or str(m or "").strip()])
            .to_dict()
        )
    else:
        sse_by_id = {}

    nm_rows_by_id: Dict[str, List[Dict[str, str]]] = {}
    if not nm.empty:
        cols = set(nm.columns)
        _nm_has_success = "success" in cols
        _nm_has_target = "target_number" in cols
        # 2026-05-20 fix: pandas reads empty CSV cells as NaN (float). Coerce
        # to "" string up front so downstream str() doesn't produce "nan".
        for _c in ("original_number", "modified_number",
                   "original_sentence", "modified_sentence", "target_number"):
            if _c in cols:
                nm[_c] = nm[_c].fillna("").astype(str)
        if {"original_number", "modified_number"}.issubset(cols):
            # 2026-05-20 — preferred path: change_magnitude is now edit-distance
            # between (original_number, modified_number), cumulative from round 0.
            for _, r in nm.iterrows():
                entry: Dict[str, Any] = {
                    "original": str(r.get("original_number", "") or "").strip(),
                    "modified": str(r.get("modified_number", "") or "").strip(),
                    "original_number": str(r.get("original_number", "") or "").strip(),
                    "modified_number": str(r.get("modified_number", "") or "").strip(),
                }
                if _nm_has_target:
                    entry["target_number"] = str(r.get("target_number", "") or "").strip()
                if _nm_has_success:
                    raw_s = r.get("success", True)
                    entry["success"] = bool(str(raw_s).strip().lower() not in ("false", "0", ""))
                    entry["has_success_field"] = True
                nm_rows_by_id.setdefault(str(r["ID"]), []).append(entry)
        elif {"original_sentence", "modified_sentence"}.issubset(cols):
            # Legacy CSV (pre 2026-05-20): change_magnitude falls back to
            # sentence-level edit-distance. Still attach target_number so
            # the new edit_success_rate can use it.
            for _, r in nm.iterrows():
                entry = {
                    "original": str(r.get("original_sentence", "") or "").strip(),
                    "modified": str(r.get("modified_sentence", "") or "").strip(),
                }
                if _nm_has_target:
                    entry["target_number"] = str(r.get("target_number", "") or "").strip()
                if _nm_has_success:
                    raw_s = r.get("success", True)
                    entry["success"] = bool(str(raw_s).strip().lower() not in ("false", "0", ""))
                    entry["has_success_field"] = True
                nm_rows_by_id.setdefault(str(r["ID"]), []).append(entry)

    nam_by_id = {}
    for _, r in nam.iterrows():
        id_ = r["ID"]
        orig = str(r.get("original_text", "") or "")
        mod  = first_nonempty(r.get("modified_article", None), r.get("modified_text", None), default="")
        nam_by_id[id_] = (orig, str(mod or ""))

    te_by_id = {}
    for _, r in te.iterrows():
        id_ = r["ID"]
        o = str(r.get("original_title", "") or "")
        m = first_nonempty(r.get("assigned_title", None), r.get("modified_title", None), default="")
        te_by_id[id_] = (o, str(m or ""))

    dataset_ids = set(text_by_id.keys())
    csv_ids = set(sa_key_sents.keys()) | set(sse_by_id.keys()) | set(nm_rows_by_id.keys()) | set(nam_by_id.keys()) | set(te_by_id.keys())

    if ids_mode == "dataset":
        all_ids = sorted(dataset_ids)
    elif ids_mode == "union":
        all_ids = sorted(dataset_ids | csv_ids)
    else:
        all_ids = sorted(dataset_ids & csv_ids)

    if only_ids is not None:
        only_set = set(str(x) for x in only_ids)
        all_ids = [i for i in all_ids if str(i) in only_set]

    log(f"[INFO] IDs to evaluate: {len(all_ids)} (mode={ids_mode})")

    log("[LOAD] Embedding model…")
    t0 = time.perf_counter()
    emb_model = SentenceTransformer(embed_model_name, device=device)
    log(f"[OK] Embeddings loaded in {sec(t0)}")

    log("[LOAD] NLI model…")
    t0 = time.perf_counter()
    nli = NLIScorer(model_name=nli_model_name, device=device, max_len=nli_max_len, fp16=True)
    log(f"[OK] NLI ready in {sec(t0)}")

    ppl = None
    if compute_fluency:
        try:
            ppl = PerplexityScorer(model_name=ppl_model_name, device=device, max_len=192)
        except Exception as e:
            log(f"[WARN] Could not load PPL model ({ppl_model_name}): {e}")
            ppl = None

    if _use_anchored_refs:
        import nltk
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        # POS tagger for the style-vector NNP / adj-adv features (offline-safe:
        # no-op if already cached; _style_pos_tags degrades to NaN if missing).
        nltk.download("averaged_perceptron_tagger", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        from nltk.tokenize import sent_tokenize as _sent_tokenize
    else:
        _sent_tokenize = None

    nam_rewrite_rows: List[int] = []
    nam_rewrite_pairs: List[Tuple[str, str]] = []

    nam_nli_rows: List[int] = []
    nam_nli_pairs: List[Tuple[str, str]] = []

    te_nli_rows: List[int] = []
    te_nli_pairs: List[Tuple[str, str]] = []

    # ===========================
    # ✅ FIX: SSE PPL per-sentence + aggregate by MEDIAN per ID
    # ===========================
    sse_ppl_row_to_indices: Dict[int, List[int]] = {}
    sse_ppl_texts_flat: List[str] = []

    rows: List[Dict[str, Any]] = []

    t0_loop = time.perf_counter()
    for id_ in tqdm(all_ids, desc="Metrics per ID"):
        orig_text = str(text_by_id.get(id_, "") or "")
        orig_title = str(title_by_id.get(id_, "") or "")

        # ---------- SA (4) ----------
        key_sents = sa_key_sents.get(id_, [])
        key_nums_strs = sa_key_nums.get(id_, [])

        SA_semantic_coverage = np.nan
        if orig_text and key_sents:
            SA_semantic_coverage = semantic_coverage_sentence_level(
                emb_model, orig_text, key_sents,
                batch_size=(96 if device == "cuda" else 32)
            )

        SA_text_compression_ratio = np.nan
        if orig_text:
            denom = max(token_count(orig_text), 1)
            numer = token_count(" ".join(key_sents)) if key_sents else 0
            SA_text_compression_ratio = float(numer / denom)

        ignore_years = bool(IGNORE_YEARS_IN_SA_NUMERIC_METRICS)
        ref_nums = extract_numbers(orig_text, ignore_years=ignore_years) if orig_text else []
        SA_has_numbers = float(1.0 if len(ref_nums) > 0 else 0.0)

        SA_numeric_coverage = np.nan
        SA_numeric_compression_ratio = np.nan
        if orig_text and len(ref_nums) > 0:
            key_nums = extract_numbers(" ".join(key_nums_strs), ignore_years=ignore_years) if key_nums_strs else []
            SA_numeric_coverage = float(num_recall(key_nums, ref_nums))
            SA_numeric_compression_ratio = float(len(key_nums) / max(len(ref_nums), 1))

        # ---------- SSE (3) ----------
        sse_pairs = sse_by_id.get(id_, [])
        SSE_semantic_drift = np.nan
        SSE_lexical_stability = np.nan
        SSE_fluency_ppl = np.nan

        if _use_anchored_refs:
            # Anchored semantics: compare KS_r vs sent_tokenize(article_orig)
            _KS_r = [s for s in key_sents if s.strip()]
            _orig_sents = _sent_tokenize(orig_text) if (orig_text or "").strip() else []
            if _KS_r and _orig_sents:
                _bs = 64 if device == "cuda" else 16
                _emb_kr = emb_model.encode(
                    _KS_r, batch_size=_bs, show_progress_bar=False,
                    convert_to_numpy=True, normalize_embeddings=True,
                )
                _emb_orig_s = emb_model.encode(
                    _orig_sents, batch_size=_bs, show_progress_bar=False,
                    convert_to_numpy=True, normalize_embeddings=True,
                )
                _sim_mat = cosine_similarity(_emb_kr, _emb_orig_s)  # (n_KS_r, n_orig)

                # ROUGE-L all-pairs matrix for stability (one row per KS_r sentence)
                _n_orig = len(_orig_sents)
                _rougeL_mat = np.zeros((len(_KS_r), _n_orig), dtype=float)
                for _i, _ks in enumerate(_KS_r):
                    _rl = rouge_metric.compute(
                        predictions=[_ks] * _n_orig,
                        references=_orig_sents,
                        rouge_types=["rougeL"],
                        use_aggregator=False,
                    )["rougeL"]
                    _rougeL_mat[_i, :] = _rl

                # Greedy best-cosine matching → drift
                _used_d: set = set()
                _drift_vals: List[float] = []
                for _i in range(len(_KS_r)):
                    _avail = [j for j in range(_n_orig) if j not in _used_d]
                    if _avail:
                        _j = max(_avail, key=lambda j: _sim_mat[_i, j])
                        _best = float(_sim_mat[_i, _j])
                        if _best >= 0.5:
                            _drift_vals.append(1.0 - _best)
                            _used_d.add(_j)
                        else:
                            _drift_vals.append(1.0)
                    else:
                        _drift_vals.append(1.0)
                SSE_semantic_drift = float(np.mean(_drift_vals))

                # Greedy best-ROUGE-L matching → stability
                _used_s: set = set()
                _stab_vals: List[float] = []
                for _i in range(len(_KS_r)):
                    _avail = [j for j in range(_n_orig) if j not in _used_s]
                    if _avail:
                        _j = max(_avail, key=lambda j: _rougeL_mat[_i, j])
                        _best = float(_rougeL_mat[_i, _j])
                        if _best >= 0.5:
                            _stab_vals.append(_best)
                            _used_s.add(_j)
                        else:
                            _stab_vals.append(0.0)
                    else:
                        _stab_vals.append(0.0)
                SSE_lexical_stability = float(np.mean(_stab_vals))

            # SSE_fluency_ppl: still uses modified sentences from sse_pairs
            if ppl is not None and sse_pairs:
                mods = [m for _, m in sse_pairs if (m or "").strip()]
                if mods:
                    row_idx = len(rows)
                    idxs: List[int] = []
                    for m in mods:
                        idxs.append(len(sse_ppl_texts_flat))
                        sse_ppl_texts_flat.append(m)
                    sse_ppl_row_to_indices[row_idx] = idxs

        else:
            # Legacy: compare modified sentence vs original sentence from sse_pairs
            if sse_pairs:
                SSE_semantic_drift = mean_cosine_distance_from_pairs(
                    emb_model, sse_pairs, batch_size=(64 if device == "cuda" else 16)
                )

                rouge_vals = []
                for o, m in sse_pairs:
                    if (o or "").strip() and (m or "").strip():
                        r = rouge_metric.compute(predictions=[m], references=[o], rouge_types=["rougeL"])
                        rouge_vals.append(float(r.get("rougeL", np.nan)))
                if rouge_vals:
                    SSE_lexical_stability = float(np.nanmean(rouge_vals))

                # ✅ enqueue per-sentence PPL (not concatenated)
                if ppl is not None:
                    mods = [m for _, m in sse_pairs if (m or "").strip()]
                    if mods:
                        row_idx = len(rows)  # row index that will be appended below
                        idxs: List[int] = []
                        for m in mods:
                            idxs.append(len(sse_ppl_texts_flat))
                            sse_ppl_texts_flat.append(m)
                        sse_ppl_row_to_indices[row_idx] = idxs

        # ---------- NM (2) ----------
        nm_rows = nm_rows_by_id.get(id_, [])
        NM_change_magnitude = nm_text_change_magnitude(nm_rows)
        # 2026-05-20 — denominator anchored to round-0 |KN_0| when available.
        _kn0_for_id = _baseline_refs.get(id_, {}).get("KN_0", []) if _use_anchored_refs else []
        NM_edit_success = nm_text_edit_success_rate(nm_rows, kn0=_kn0_for_id)

        # ---------- NaM (2) ----------
        _, nam_mod = nam_by_id.get(id_, (orig_text, ""))
        NaM_rewrite_distance = np.nan
        NaM_contradiction_prob = np.nan

        # NaM_rewrite_distance: always legacy (measures how much NaM changed the article)
        if (nam_mod or "").strip() and (orig_text or "").strip():
            nam_rewrite_rows.append(len(rows))
            nam_rewrite_pairs.append((orig_text, nam_mod))

        # NaM_contradiction_prob
        if _use_anchored_nam_te:
            # Anchored: premise=article_rr, hypothesis=" ".join(KS_0)
            _ks0 = _baseline_refs.get(id_, {}).get("KS_0", [])
            _hyp_nam = " ".join([str(s) for s in _ks0 if s]).strip()
            _art_rr = str(_gfn_text_by_id.get(id_, "") or orig_text)
            if not (_art_rr or "").strip():
                _art_rr = orig_text
            if _hyp_nam and (_art_rr or "").strip():
                nam_nli_rows.append(len(rows))
                nam_nli_pairs.append((_art_rr, _hyp_nam))
        else:
            # Legacy: only when NaM ran; hypothesis from modified or original key sentences
            if (nam_mod or "").strip():
                mod_key_sents = [m for _, m in sse_pairs if (m or "").strip()] if sse_pairs else []
                if not mod_key_sents:
                    mod_key_sents = [s for s in key_sents if (s or "").strip()]
                hyp = " ".join(mod_key_sents).strip()
                if hyp:
                    nam_nli_rows.append(len(rows))
                    nam_nli_pairs.append((nam_mod, hyp))

        # ---------- TE (2) ----------
        _, te_mod_title = te_by_id.get(id_, (orig_title, ""))
        TE_title_entailment = np.nan
        TE_title_contradiction_prob = np.nan

        if _use_anchored_nam_te:
            # Anchored: premise=article_rr, hypothesis=title_rr from Generated_Fake_News.csv
            _art_rr = str(_gfn_text_by_id.get(id_, "") or orig_text)
            if not (_art_rr or "").strip():
                _art_rr = orig_text
            _title_rr = str(_gfn_title_by_id.get(id_, "") or "")
            if (_art_rr or "").strip() and (_title_rr or "").strip():
                te_nli_rows.append(len(rows))
                te_nli_pairs.append((_art_rr, _title_rr))
        else:
            # Legacy: text_for_title = nam_mod or orig_text; hypothesis = te_mod_title
            text_for_title = nam_mod if (nam_mod or "").strip() else orig_text
            if (text_for_title or "").strip() and (te_mod_title or "").strip():
                te_nli_rows.append(len(rows))
                te_nli_pairs.append((text_for_title, te_mod_title))

        # ---------- Style vector (TE title + NaM body) — SIGNED deviation vs round-0 original ----------
        _sv_cur_title = (_gfn_title_by_id.get(id_, "") if _use_anchored_nam_te else te_mod_title) or ""
        _sv_cur_body = (_gfn_text_by_id.get(id_, "") if _use_anchored_nam_te else nam_mod) or ""
        _style_title_feats = _style_features_title(_sv_cur_title, orig_title)
        _style_body_feats = _style_features_body(_sv_cur_body, orig_text)

        row = {
            "ID": id_,

            "SA_semantic_coverage": SA_semantic_coverage,
            "SA_text_compression_ratio": SA_text_compression_ratio,
            "SA_numeric_coverage": SA_numeric_coverage,
            "SA_numeric_compression_ratio": SA_numeric_compression_ratio,

            "SSE_semantic_drift": SSE_semantic_drift,
            "SSE_lexical_stability": SSE_lexical_stability,
            "SSE_fluency_ppl": SSE_fluency_ppl,  # filled post-batch (median)

            "NM_change_magnitude": NM_change_magnitude,
            "NM_edit_success_rate": NM_edit_success,

            "NaM_rewrite_distance": NaM_rewrite_distance,
            "NaM_contradiction_prob": NaM_contradiction_prob,

            "TE_title_entailment": TE_title_entailment,
            "TE_title_contradiction_prob": TE_title_contradiction_prob,

            **_style_title_feats,
            **_style_body_feats,
        }

        if include_applicability_flags:
            row["SA_has_numbers"] = SA_has_numbers

        rows.append(row)

    log(f"[OK] main loop completed in {sec(t0_loop)}")
    log(f"[INFO] queued NaM rewrite pairs: {len(nam_rewrite_pairs)} | NaM NLI pairs: {len(nam_nli_pairs)} | TE NLI pairs: {len(te_nli_pairs)} | SSE PPL sentences: {len(sse_ppl_texts_flat)}")

    # ---------- Batch rewrite distance for NaM ----------
    if nam_rewrite_pairs:
        log("[NaM] computing rewrite distance (embedding cosine) …")
        t0 = time.perf_counter()
        batch_size = 128 if device == "cuda" else 32
        dists = []
        for i in range(0, len(nam_rewrite_pairs), batch_size):
            chunk = nam_rewrite_pairs[i:i+batch_size]
            origs = [a for a, _ in chunk]
            mods  = [b for _, b in chunk]
            e1 = emb_model.encode(origs, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
            e2 = emb_model.encode(mods,  batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
            sims = np.sum(e1 * e2, axis=1)
            dists.extend((1.0 - sims).tolist())
        for row_idx, dist in zip(nam_rewrite_rows, dists):
            rows[row_idx]["NaM_rewrite_distance"] = float(dist)
        log(f"[OK] NaM rewrite distance done in {sec(t0)}")

    # ---------- Batch NLI for NaM ----------
    if nam_nli_pairs:
        log("[NaM] running NLI batch…")
        t0 = time.perf_counter()
        batch_size = 64 if device == "cuda" else 16
        probs = nli.batch_probs(nam_nli_pairs, batch_size=batch_size, show_progress=True)
        for row_idx, pr in zip(nam_nli_rows, probs):
            rows[row_idx]["NaM_contradiction_prob"] = pr["contradiction"]
        log(f"[OK] NaM NLI done in {sec(t0)}")

    # ---------- Batch NLI for TE ----------
    if te_nli_pairs:
        log("[TE] running NLI batch…")
        t0 = time.perf_counter()
        batch_size = 64 if device == "cuda" else 16
        probs = nli.batch_probs(te_nli_pairs, batch_size=batch_size, show_progress=True)
        for row_idx, pr in zip(te_nli_rows, probs):
            rows[row_idx]["TE_title_entailment"] = pr["entailment"]
            rows[row_idx]["TE_title_contradiction_prob"] = pr["contradiction"]
        log(f"[OK] TE NLI done in {sec(t0)}")

    # ---------- Batch PPL for SSE (MEDIAN per ID) ----------
    if ppl is not None and sse_ppl_texts_flat:
        log("[SSE] running perplexity batch (per-sentence)…")
        t0 = time.perf_counter()
        batch_size = 16 if device == "cuda" else 8

        vals_flat = ppl.batch_perplexity(
            sse_ppl_texts_flat,
            batch_size=batch_size,
            show_progress=True,
        )

        for row_idx, idxs in sse_ppl_row_to_indices.items():
            ps = [float(vals_flat[j]) for j in idxs if j < len(vals_flat)]
            rows[row_idx]["SSE_fluency_ppl"] = float(np.median(ps)) if ps else np.nan

        log(f"[OK] SSE PPL done in {sec(t0)}")

    # ---------- Build df_raw ----------
    df_raw = pd.DataFrame(rows)

    # residualize coverages w.r.t. compressions (same column names)
    # Three-branch policy:
    #   residualizer_path is None              -> fit and apply in-memory (legacy).
    #   residualizer_path set, file exists     -> load (a, b, fitted) and apply.
    #   residualizer_path set, file missing    -> fit, save, apply (round-0 freeze).
    if not df_raw.empty:
        y_sem = pd.to_numeric(df_raw["SA_semantic_coverage"], errors="coerce").to_numpy(dtype=float)
        x_txt = pd.to_numeric(df_raw["SA_text_compression_ratio"], errors="coerce").to_numpy(dtype=float)
        y_num = pd.to_numeric(df_raw["SA_numeric_coverage"], errors="coerce").to_numpy(dtype=float)
        x_num = pd.to_numeric(df_raw["SA_numeric_compression_ratio"], errors="coerce").to_numpy(dtype=float)

        loaded = _load_residualizer(residualizer_path) if residualizer_path else None
        if loaded is not None:
            sem_p = loaded.get("sem", {}) or {}
            num_p = loaded.get("num", {}) or {}
            sem_a = float(sem_p.get("a", 0.0)); sem_b = float(sem_p.get("b", 0.0)); sem_fit = bool(sem_p.get("fitted", False))
            num_a = float(num_p.get("a", 0.0)); num_b = float(num_p.get("b", 0.0)); num_fit = bool(num_p.get("fitted", False))
            log(f"[RESIDUALIZER] loaded {residualizer_path} (sem.fitted={sem_fit}, num.fitted={num_fit})")
        else:
            sem_a, sem_b, sem_fit = _fit_residualizer(y_sem, x_txt)
            num_a, num_b, num_fit = _fit_residualizer(y_num, x_num)
            if residualizer_path:
                _save_residualizer(
                    residualizer_path,
                    sem=(sem_a, sem_b, sem_fit),
                    num=(num_a, num_b, num_fit),
                    n_samples=int(len(df_raw)),
                )
                log(f"[RESIDUALIZER] fitted at round 0 and saved -> {residualizer_path} "
                    f"(n={len(df_raw)}, sem.fitted={sem_fit}, num.fitted={num_fit})")

        df_raw["SA_semantic_coverage"] = _apply_residualizer(y_sem, x_txt, sem_a, sem_b, sem_fit)
        df_raw["SA_numeric_coverage"] = _apply_residualizer(y_num, x_num, num_a, num_b, num_fit)

    core_metrics = list(METRICS_REGISTRY.keys())
    for m in core_metrics:
        if m not in df_raw.columns:
            df_raw[m] = np.nan

    if output_mode not in ("feat", "raw_and_feat"):
        raise ValueError("output_mode must be 'feat' or 'raw_and_feat'")

    df_feat = df_raw[["ID"]].copy()
    for m in core_metrics:
        spec = METRICS_REGISTRY[m]
        df_feat[m] = df_raw[m].apply(lambda x: spec.raw_to_feature(x) if x is not None else np.nan)

    if output_mode == "feat":
        df_out = df_feat
    else:
        df_out = df_raw[["ID"]].copy()
        for m in core_metrics:
            df_out[f"{m}_raw"] = df_raw[m]
            df_out[f"{m}_feat"] = df_feat[m]
        if include_applicability_flags and "SA_has_numbers" in df_raw.columns:
            df_out["SA_has_numbers"] = df_raw["SA_has_numbers"]

    out_csv = os.path.join(base_out_dir, out_csv_name)

    if append_if_exists and os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        try:
            prev = pd.read_csv(out_csv, encoding="utf-8")
            if "ID" in prev.columns:
                prev["ID"] = prev["ID"].astype(str)
            df_out["ID"] = df_out["ID"].astype(str)
            df_out = pd.concat([prev, df_out], ignore_index=True)
            if "ID" in df_out.columns:
                df_out = df_out.drop_duplicates(subset=["ID"], keep="last")
        except Exception:
            pass

    df_out.to_csv(out_csv, index=False, encoding="utf-8")
    log(f"[SAVE] metrics CSV -> {out_csv}")
    log(f"[PATH] {os.path.abspath(out_csv)}")

    return df_out