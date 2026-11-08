"""Persistent warning logger.

Centralizes ad-hoc print warnings to a single log file at
``Output/logs/warnings.log`` while keeping the existing terminal output.

Usage::

    from .warn_logger import warn, set_context

    set_context(round=1, detector="bert")
    warn("REALIGN_LOW", "similarity below threshold",
         article_id="11", similarity=0.387, query=query[:60])

Each line is plain text, greppable::

    2026-05-19T14:23:11 round=1 det=bert ID=11 code=REALIGN_LOW similarity=0.387 query='...' msg='similarity below threshold'

Override the log directory with ``WARN_LOG_DIR`` if needed (default
``Output/logs``).
"""
from __future__ import annotations

import os
import threading
from datetime import datetime
from typing import Any

_LOCK = threading.Lock()
_CONTEXT: dict = {}
_LOG_FILE_PATH: str | None = None


def _resolve_log_path() -> str:
    global _LOG_FILE_PATH
    if _LOG_FILE_PATH is not None:
        return _LOG_FILE_PATH
    base = os.environ.get("WARN_LOG_DIR") or os.path.join("Output", "logs")
    os.makedirs(base, exist_ok=True)
    _LOG_FILE_PATH = os.path.join(base, "warnings.log")
    return _LOG_FILE_PATH


def set_context(**kwargs: Any) -> None:
    """Update the global context prefixing every warning. Pass None to remove a key."""
    with _LOCK:
        for k, v in kwargs.items():
            if v is None:
                _CONTEXT.pop(k, None)
            else:
                _CONTEXT[k] = v


def clear_context(*keys: str) -> None:
    """Drop selected keys; with no args, clear everything."""
    with _LOCK:
        if not keys:
            _CONTEXT.clear()
        else:
            for k in keys:
                _CONTEXT.pop(k, None)


def _fmt_value(v: Any) -> str:
    if isinstance(v, str):
        s = v.replace("\n", " ").replace("\r", " ")
        return repr(s) if (" " in s or "=" in s) else s
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1000 else f"{v:g}"
    return str(v)


def _format(code: str, msg: str, ctx: dict) -> str:
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    merged = {**_CONTEXT, **ctx}
    parts = [ts]
    for key in ("round", "detector", "article_id"):
        if key in merged:
            tag = {"round": "round", "detector": "det", "article_id": "ID"}[key]
            parts.append(f"{tag}={_fmt_value(merged.pop(key))}")
    parts.append(f"code={code}")
    for k, v in merged.items():
        parts.append(f"{k}={_fmt_value(v)}")
    if msg:
        parts.append(f"msg={_fmt_value(msg)}")
    return " ".join(parts)


def warn(code: str, msg: str = "", **ctx: Any) -> None:
    """Emit a warning to stdout AND to Output/logs/warnings.log.

    File I/O errors are swallowed: stdout always wins, persistence is best-effort.
    """
    line = _format(code, msg, ctx)
    print(line)
    try:
        path = _resolve_log_path()
        with _LOCK, open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
