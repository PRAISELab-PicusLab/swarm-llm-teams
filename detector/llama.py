"""
Llama 3.3:70b as a fake-news detector via Ollama.

Same model as the generator (llama3.3:70b). Calls Ollama /api/chat directly
(no AutoGen overhead). Returns REAL/FAKE + confidence (always 1.0 since
the LLM gives a binary answer, no probability).

Much slower than HuggingFace detectors (~5-30s per article vs ~10ms).

Environment variables (shared with pipeline):
  OLLAMA_BASE_URL       (default http://localhost:11434)
  OLLAMA_NUM_CTX        (default 32768)
  OLLAMA_TEMPERATURE    (default 0.5)
  OLLAMA_REQUEST_TIMEOUT (default 600)
"""

import json
import os
import re
import urllib.request
from typing import Optional, Tuple

MODEL = os.getenv("OLLAMA_MODEL", "llama3.3:70b")

SYSTEM_PROMPT = (
    "You are an AI fact checker trained to detect fake news. "
    "Your task is to read the following news text carefully and "
    "determine if it is real or fake news.\n\n"
    "Provide your final answer by responding ONLY with either "
    "'REAL' or 'FAKE' (without quotes or any additional text).\n\n"
    "Your output should be a JSON object containing only your "
    "final answer. Do not include any explanations or additional "
    "text. Use this format:\n\n"
    '{\n    "final_answer": "Your final answer here (REAL or FAKE)"\n}'
)

_MAX_RETRIES = 3


def _get_config():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    base_url = re.sub(r"/v1/?$", "", base_url.rstrip("/"))
    return dict(
        base_url=base_url,
        temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.5")),
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "32768")),
        timeout=int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "600")),
    )


def _call_ollama(prompt: str) -> str:
    cfg = _get_config()
    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "format": "json",
        "options": {
            "temperature": cfg["temperature"],
            "num_ctx": cfg["num_ctx"],
        },
    }
    req = urllib.request.Request(
        cfg["base_url"] + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=cfg["timeout"]) as resp:
        body = resp.read().decode("utf-8", errors="replace")

    data = json.loads(body)
    if isinstance(data, dict):
        msg = data.get("message")
        if isinstance(msg, dict):
            return str(msg.get("content", "") or "")
    return ""


def _parse_response(raw: str) -> Optional[str]:
    if not raw:
        return None

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            answer = str(obj.get("final_answer", "")).strip().upper()
            if answer in ("REAL", "FAKE"):
                return answer
    except (json.JSONDecodeError, ValueError):
        pass

    m = re.search(r"\{[^}]*\}", raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            answer = str(obj.get("final_answer", "")).strip().upper()
            if answer in ("REAL", "FAKE"):
                return answer
        except (json.JSONDecodeError, ValueError):
            pass

    raw_upper = raw.upper()
    has_real = "REAL" in raw_upper
    has_fake = "FAKE" in raw_upper
    if has_real and not has_fake:
        return "REAL"
    if has_fake and not has_real:
        return "FAKE"

    return None


def detect_fake_news_llama(
    title: str, text: str, device: Optional[str] = None
) -> Tuple[str, float]:
    """Classify article as REAL or FAKE using Llama 3.3:70b via Ollama.

    *device* is accepted for interface compatibility but ignored
    (Ollama manages GPU allocation).
    """
    prompt = (
        "Determine if the following is real or fake news:\n\n"
        f"Title: {title or ''}\n\n{text or ''}"
    )

    for attempt in range(_MAX_RETRIES):
        try:
            raw = _call_ollama(prompt)
            label = _parse_response(raw)
            if label:
                return label, 1.0
        except Exception as e:
            if attempt == _MAX_RETRIES - 1:
                print(f"[llama detector] ERRORE dopo {_MAX_RETRIES} tentativi: {e}")

    return "FAKE", 0.0
