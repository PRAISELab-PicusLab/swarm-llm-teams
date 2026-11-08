from __future__ import annotations

from typing import Tuple, List, Any, Dict, Optional

import json
import re

from .agents import call_ollama_structured, get_agent_json_schema, get_last_agent_message, get_utils
from .prompts import (
    check_semantic_analyzer_format_prompt,
    check_salient_sentence_editor_format_prompt,
    check_number_modifier_format_prompt,
    check_narrative_modifier_format_prompt,
    check_title_editor_format_prompt,
    check_llm_fake_news_detector_format_prompt,
)

NUM_ATTEMPT = 3


def _local_repair_json(raw: str) -> Optional[dict]:
    """Best-effort JSON parsing/repair.

    This mirrors (in a lightweight way) the repair strategy used inside
    pipeline_multiagent.py so that format checks and pipeline coercers are
    consistent.
    """
    if raw is None:
        return None

    s = str(raw).strip()
    if not s:
        return None

    # 1) direct parse
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # 2) try to extract a JSON object block
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        cand = m.group(0)
        try:
            obj = json.loads(cand)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    # 3) try json_repair if available
    try:
        from json_repair import repair_json  # type: ignore

        repaired = repair_json(s)
        obj = json.loads(repaired)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _sanitize_numbers(nums: Any) -> List[str]:
    if not isinstance(nums, list):
        return []
    out: List[str] = []
    for x in nums:
        sx = str(x)
        if re.search(r"\d", sx):
            out.append(sx)
    return out


def _to_json(text: str) -> Dict[str, Any]:
    # Prefer local repair (more robust) and fall back to the original util.
    repaired = _local_repair_json(text)
    if isinstance(repaired, dict):
        return repaired
    u = get_utils()
    obj = u.from_string_to_json(text)
    return obj if isinstance(obj, dict) else {}


def check_semantic_analyzer_format(agent) -> Tuple[bool, List[str], List[str]]:
    """
    STRICT (no extra keys), coerente con pipeline:
    - accetta ESATTAMENTE:
        1) {"key_sentences", "key_numbers"}
        2) {"key_sentences", "numbers"}  (fallback legacy)
    - ritorna (key_sentences, key_numbers) come liste di stringhe
    """
    for attempt in range(NUM_ATTEMPT):
        answer = get_last_agent_message(agent)["content"]
        json_answer = _to_json(answer)

        if isinstance(json_answer, dict):
            keys = set(json_answer.keys())

            # schema 1: key_numbers (superset match — number_to_sentence_map allowed as extra key)
            if {"key_sentences", "key_numbers"}.issubset(keys):
                if (
                    isinstance(json_answer["key_sentences"], list)
                    and isinstance(json_answer["key_numbers"], list)
                ):
                    ks = [str(x) for x in json_answer["key_sentences"]]
                    kn = _sanitize_numbers(json_answer["key_numbers"])
                    return True, ks, kn

            # schema 2: numbers (legacy, superset match)
            if {"key_sentences", "numbers"}.issubset(keys):
                if (
                    isinstance(json_answer["key_sentences"], list)
                    and isinstance(json_answer["numbers"], list)
                ):
                    ks = [str(x) for x in json_answer["key_sentences"]]
                    kn = _sanitize_numbers(json_answer["numbers"])
                    return True, ks, kn

        if attempt < NUM_ATTEMPT - 1:
            call_ollama_structured(agent, check_semantic_analyzer_format_prompt + answer, schema=get_agent_json_schema(agent))

    return False, [], []


def check_salient_sentence_editor_format(agent) -> Tuple[bool, str, str]:
    """
    STRICT: accetta ESATTAMENTE {"original_sentence","modified_sentence"}
    """
    for attempt in range(NUM_ATTEMPT):
        answer = get_last_agent_message(agent)["content"]
        json_answer = _to_json(answer)

        if (
            isinstance(json_answer, dict)
            and set(json_answer.keys()) == {"original_sentence", "modified_sentence"}
            and isinstance(json_answer["original_sentence"], str)
            and isinstance(json_answer["modified_sentence"], str)
        ):
            return True, json_answer["original_sentence"], json_answer["modified_sentence"]

        if attempt < NUM_ATTEMPT - 1:
            call_ollama_structured(agent, check_salient_sentence_editor_format_prompt + answer, schema=get_agent_json_schema(agent))

    return False, "ERROR", "ERROR"


def check_number_modifier_format(agent) -> Tuple[bool, str, str]:
    """
    STRICT: accetta ESATTAMENTE {"original_sentence","modified_sentence"}
    """
    for attempt in range(NUM_ATTEMPT):
        answer = get_last_agent_message(agent)["content"]
        json_answer = _to_json(answer)

        if (
            isinstance(json_answer, dict)
            and set(json_answer.keys()) == {"original_sentence", "modified_sentence"}
            and isinstance(json_answer["original_sentence"], str)
            and isinstance(json_answer["modified_sentence"], str)
        ):
            return True, json_answer["original_sentence"], json_answer["modified_sentence"]

        if attempt < NUM_ATTEMPT - 1:
            call_ollama_structured(agent, check_number_modifier_format_prompt + answer, schema=get_agent_json_schema(agent))

    return False, "ERROR", "ERROR"


def check_narrative_modifier_format(agent) -> Tuple[bool, str]:
    """
    STRICT (no extra keys), coerente con pipeline:
    - accetta ESATTAMENTE {"modified_article", "technique_used"} (formato nuovo)
    - accetta ESATTAMENTE {"modified_article"} (back-compat)
    - accetta ESATTAMENTE {"modified_text"} (legacy)
    """
    for attempt in range(NUM_ATTEMPT):
        answer = get_last_agent_message(agent)["content"]
        json_answer = _to_json(answer)

        if isinstance(json_answer, dict):
            keys = set(json_answer.keys())

            if keys == {"modified_article", "technique_used"} \
               and isinstance(json_answer.get("modified_article"), str) \
               and isinstance(json_answer.get("technique_used"), str):
                return True, json_answer["modified_article"]

            if keys == {"modified_article"} and isinstance(json_answer["modified_article"], str):
                return True, json_answer["modified_article"]

            if keys == {"modified_text"} and isinstance(json_answer["modified_text"], str):
                return True, json_answer["modified_text"]

        if attempt < NUM_ATTEMPT - 1:
            call_ollama_structured(agent, check_narrative_modifier_format_prompt + answer, schema=get_agent_json_schema(agent))

    return False, "ERROR"


def check_title_editor_format(agent) -> Tuple[bool, str]:
    """
    STRICT (no extra keys), coerente con pipeline:
    - accetta ESATTAMENTE {"assigned_title"} oppure ESATTAMENTE {"modified_title"}
    """
    for attempt in range(NUM_ATTEMPT):
        answer = get_last_agent_message(agent)["content"]
        json_answer = _to_json(answer)

        if isinstance(json_answer, dict):
            keys = set(json_answer.keys())

            if keys == {"assigned_title"} and isinstance(json_answer["assigned_title"], str):
                return True, json_answer["assigned_title"]

            if keys == {"modified_title"} and isinstance(json_answer["modified_title"], str):
                return True, json_answer["modified_title"]

        if attempt < NUM_ATTEMPT - 1:
            call_ollama_structured(agent, check_title_editor_format_prompt + answer, schema=get_agent_json_schema(agent))

    return False, "ERROR"


def check_llm_format(agent) -> Tuple[bool, str]:
    """
    STRICT: accetta ESATTAMENTE {"final_answer"}
    """
    for attempt in range(NUM_ATTEMPT):
        answer = get_last_agent_message(agent)["content"]
        json_answer = _to_json(answer)

        if (
            isinstance(json_answer, dict)
            and set(json_answer.keys()) == {"final_answer"}
            and isinstance(json_answer["final_answer"], str)
        ):
            return True, json_answer["final_answer"]

        if attempt < NUM_ATTEMPT - 1:
            call_ollama_structured(agent, check_llm_fake_news_detector_format_prompt + answer, schema=get_agent_json_schema(agent))

    return False, "ERROR"