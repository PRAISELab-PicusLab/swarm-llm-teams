import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Literal, Optional, Tuple

import autogen

from . import utils as utils_default
from . import utils_local as utils_local
from .prompts import (
    llm_fake_news_detector_system_prompt,
    narrative_modifier_system_prompt,
    number_modifier_system_prompt,
    salient_sentence_editor_system_prompt,
    semantic_analyzer_system_prompt,
    title_editor_system_prompt,
)

UtilsMode = Literal["default", "local"]

_JSON_ARRAY_STR_SCHEMA = {"type": "array", "items": {"type": "string"}, "minItems": 0}

AGENT_JSON_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "SemanticAnalyzer": {
        "type": "object",
        "properties": {
            "key_sentences": _JSON_ARRAY_STR_SCHEMA,
            "key_numbers": _JSON_ARRAY_STR_SCHEMA,
            "number_to_sentence_map": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "minProperties": 0,
            },
        },
        "required": ["key_sentences", "key_numbers", "number_to_sentence_map"],
        "additionalProperties": False,
    },
    "SalientSentenceEditor": {
        "type": "object",
        "properties": {
            "original_sentence": {"type": "string"},
            "modified_sentence": {"type": "string"},
        },
        "required": ["original_sentence", "modified_sentence"],
        "additionalProperties": False,
    },
    "NumberModifier": {
        "type": "object",
        "properties": {
            "original_sentence": {"type": "string"},
            "modified_sentence": {"type": "string"},
        },
        "required": ["original_sentence", "modified_sentence"],
        "additionalProperties": False,
    },
    "NarrativeModifier": {
        "type": "object",
        "properties": {
            "modified_article": {"type": "string"},
            "technique_used": {
                "type": "string",
                "enum": [
                    "Presenting Irrelevant Data",
                    "Misrepresentation of Position",
                    "Whataboutism",
                    "Causal Oversimplification",
                    "Obfuscation",
                    "Appeal to Authority",
                    "Black-and-white Fallacy",
                    "Loaded Language",
                    "Exaggeration or Minimisation",
                ],
            },
        },
        "required": ["modified_article", "technique_used"],
        "additionalProperties": False,
    },
    "TitleEditor": {
        "type": "object",
        "properties": {"modified_title": {"type": "string"}},
        "required": ["modified_title"],
        "additionalProperties": False,
    },
    "LlamaFakeNewsDetector": {
        "type": "object",
        "properties": {"final_answer": {"type": "string"}},
        "required": ["final_answer"],
        "additionalProperties": False,
    },
    "GemmaFakeNewsDetector": {
        "type": "object",
        "properties": {"final_answer": {"type": "string"}},
        "required": ["final_answer"],
        "additionalProperties": False,
    },
    "Detector": {
        "type": "object",
        "properties": {"feedback": {"type": "string"}},
        "required": ["feedback"],
        "additionalProperties": False,
    },
    "EvaluatorAgent": {
        "type": "object",
        "properties": {
            "feedback": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "agent": {"type": "string"},
                        "message": {"type": "string"},
                    },
                    "required": ["agent", "message"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["feedback"],
        "additionalProperties": False,
    },
    "SalientTextRewriter": {
        "type": "object",
        "properties": {"modified_text": {"type": "string"}},
        "required": ["modified_text"],
        "additionalProperties": False,
    },
    "UniversalAgent": {
        "type": "object",
        "properties": {"modified_text": {"type": "string"}},
        "required": ["modified_text"],
        "additionalProperties": False,
    },
}


@lru_cache(maxsize=4)
def _get_ollama_version(base_url: str) -> Optional[Tuple[int, int, int]]:
    """Query Ollama /api/version. Returns (major, minor, patch) or None if unreachable."""
    url = _ensure_native_base_url(base_url) + "/api/version"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        ver_str = str(data.get("version", "")).strip()
        m = re.match(r"^v?(\d+)\.(\d+)\.(\d+)", ver_str)
        if m:
            return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except Exception:
        pass
    return None


def _ollama_supports_schema(base_url: str) -> bool:
    """
    True se la versione Ollama supporta format=<JSON Schema>. Versioni < 0.3.0
    rispondono HTTP 500 o eco dello schema. Fallback conservativo (True) se
    /api/version non raggiungibile — replica il comportamento storico.
    Override con OLLAMA_FORCE_FORMAT_JSON=1.
    """
    if os.getenv("OLLAMA_FORCE_FORMAT_JSON", "").strip().lower() in ("1", "true", "yes"):
        return False
    ver = _get_ollama_version(base_url)
    if ver is None:
        return True
    return ver >= (0, 3, 0)


def _supports_full_schema(model_name: str, base_url: Optional[str] = None) -> bool:
    """
    True se il modello supporta constrained generation con JSON Schema completo.
    Euristica: modelli >= 7b supportano lo schema completo nel parametro format=.
    Modelli < 7b (1b, 3b) usano format="json" semplice per evitare HTTP 500.
    Se la dimensione non è determinabile dal nome, assume supporto completo (safe default per modelli grandi).
    Inoltre interroga la versione di Ollama: server < 0.3.0 non supportano schemas.
    """
    if not _ollama_supports_schema(base_url or os.getenv("OLLAMA_BASE_URL", "")):
        return False
    if not model_name:
        return True
    match = re.search(r":(\d+)b", model_name.lower())
    if match:
        return int(match.group(1)) >= 7
    return True


def get_utils(mode: Optional[UtilsMode] = None):
    mode = (mode or os.getenv("PIPELINE_UTILS", "default")).strip().lower()
    if mode == "local":
        return utils_local
    return utils_default


def make_user_proxy() -> autogen.UserProxyAgent:
    return autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
        default_auto_reply="default_auto_reply",
    )


def _ensure_native_base_url(base_url: str) -> str:
    url = (base_url or "http://localhost:11434").rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


def _extract_agent_config(agent: Any) -> Dict[str, Any]:
    llm_config = getattr(agent, "llm_config", None) or {}
    if isinstance(llm_config, dict):
        cfg_list = llm_config.get("config_list") or []
        if cfg_list and isinstance(cfg_list[0], dict):
            return dict(cfg_list[0])
    return {}


def get_agent_json_schema(agent: Any) -> Optional[Dict[str, Any]]:
    return AGENT_JSON_SCHEMAS.get(str(getattr(agent, "name", "")), None)


def _store_last_agent_message(agent: Any, content: str) -> None:
    try:
        setattr(agent, "_last_ollama_message", {"content": str(content or "")})
    except Exception:
        pass


def get_last_agent_message(agent: Any) -> Dict[str, str]:
    msg = getattr(agent, "_last_ollama_message", None)
    if isinstance(msg, dict) and "content" in msg:
        return {"content": str(msg.get("content", "") or "")}
    try:
        return user_proxy.last_message(agent)
    except Exception:
        return {"content": ""}


def _build_grounded_message(message: str, schema: Optional[Dict[str, Any]]) -> str:
    msg = str(message or "").strip()
    if not schema:
        return msg
    schema_txt = json.dumps(schema, ensure_ascii=False)
    grounding = (
        "\n\nSTRICT OUTPUT CONTRACT:\n"
        "Return exactly one JSON object that matches this JSON Schema.\n"
        f"Schema: {schema_txt}\n"
        "Do not add commentary, markdown, code fences, or extra keys."
    )
    if schema_txt in msg:
        return msg
    return msg + grounding


def call_ollama_structured(
    agent: Any,
    message: str,
    schema: Optional[Dict[str, Any]] = None,
    timeout: Optional[int] = None,
) -> Dict[str, str]:
    if timeout is None:
        timeout = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "600"))
    cfg = _extract_agent_config(agent)
    base_url = _ensure_native_base_url(
        str(cfg.get("base_url") or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434")
    )
    endpoint = base_url + "/api/chat"
    model = str(cfg.get("model") or os.getenv("OLLAMA_MODEL") or "gpt-oss:20b")
    system_message = str(getattr(agent, "system_message", "") or "")
    schema = schema or get_agent_json_schema(agent)
    grounded_message = _build_grounded_message(message, schema)

    payload: Dict[str, Any] = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": grounded_message},
        ],
        "options": {
            "temperature": float(cfg.get("temperature", 0.5)),
            "num_ctx": int(cfg.get("num_ctx", 32768)),
        },
    }
    _think = getattr(agent, "think", None)  # swarm coordinator may disable thinking; None = payload unchanged
    if _think is not None:
        payload["think"] = bool(_think)
    if schema:
        # Modelli < 7b non supportano constrained generation con JSON Schema completo.
        # Anche server Ollama < 0.3.0 non supportano format=<JSON Schema> (HTTP 500 o eco).
        # In quei casi usa format="json" semplice; il grounding nel messaggio guida l'output.
        payload["format"] = schema if _supports_full_schema(model, base_url) else "json"

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        print(f"[LLM][ERROR] HTTP {exc.code} from Ollama (model={model}, num_ctx={payload.get('options',{}).get('num_ctx')}): {err_body[:200]}")
        raise
    data = json.loads(body)

    content = ""
    if isinstance(data, dict):
        msg = data.get("message")
        if isinstance(msg, dict):
            content = str(msg.get("content", "") or "")
        elif "response" in data:
            content = str(data.get("response", "") or "")

    _store_last_agent_message(agent, content)
    return {"content": content}


@dataclass(frozen=True)
class AgentBundle:
    user_proxy: autogen.UserProxyAgent
    SemanticAnalyzer: autogen.AssistantAgent
    SalientSentenceEditor: autogen.AssistantAgent
    NumberModifier: autogen.AssistantAgent
    NarrativeModifier: autogen.AssistantAgent
    TitleEditor: autogen.AssistantAgent
    LlamaFakeNewsDetector: autogen.AssistantAgent
    GemmaFakeNewsDetector: autogen.AssistantAgent
    Detector: autogen.AssistantAgent
    EvaluatorAgent: autogen.AssistantAgent
    SalientTextRewriter: autogen.AssistantAgent
    UniversalAgent: autogen.AssistantAgent


def build_agents(mode: Optional[UtilsMode] = None) -> AgentBundle:
    u = get_utils(mode)
    proxy = make_user_proxy()

    SemanticAnalyzer = autogen.AssistantAgent(
        name="SemanticAnalyzer",
        system_message=semantic_analyzer_system_prompt,
        llm_config=u.llama3,
    )
    SalientSentenceEditor = autogen.AssistantAgent(
        name="SalientSentenceEditor",
        system_message=salient_sentence_editor_system_prompt,
        llm_config=u.llama3,
    )
    NumberModifier = autogen.AssistantAgent(
        name="NumberModifier",
        system_message=number_modifier_system_prompt,
        llm_config=u.llama3,
    )
    NarrativeModifier = autogen.AssistantAgent(
        name="NarrativeModifier",
        system_message=narrative_modifier_system_prompt,
        llm_config=u.llama3,
    )
    TitleEditor = autogen.AssistantAgent(
        name="TitleEditor",
        system_message=title_editor_system_prompt,
        llm_config=u.llama3,
    )
    LlamaFakeNewsDetector = autogen.AssistantAgent(
        name="LlamaFakeNewsDetector",
        system_message=llm_fake_news_detector_system_prompt,
        llm_config=u.llama3,
    )
    GemmaFakeNewsDetector = autogen.AssistantAgent(
        name="GemmaFakeNewsDetector",
        system_message=llm_fake_news_detector_system_prompt,
        llm_config=u.gemma,
    )
    Detector = autogen.AssistantAgent(
        name="Detector",
        system_message=(
            "You are a fake news generator. Analyze the text you receive as input and a series of words and phrases "
            "that signal its falsity. Your goals are: "
            "1) For each word or phrase you receive, provide a detailed analysis that explains why the news is fake. "
            "2) Provide suggestions on how to modify the text to maintain a false message, making it credible and plausible. "
            "The rewritten text must look real and must be classified as real news. "
            "3) You must be able to change the expected label from false to true. "
            'Return ONLY valid JSON: { "feedback": "..." }'
        ),
        llm_config=u.llama3,
    )
    EvaluatorAgent = autogen.AssistantAgent(
        name="EvaluatorAgent",
        system_message=(
            "You are an expert text evaluator specializing in disinformation and persuasive writing. "
            "You evaluate a modified text against its original version and suggest improvements to make the fake news more credible. "
            "You are provided with original_text, modified_text, evaluation_metrics (BLEU, ROUGE, readability, BERTScore), and target_agents. "
            'Return ONLY one-line valid JSON: { "feedback": [ { "agent": "...", "message": "..." }, ... ] }.'
        ),
        llm_config=u.llama3,
    )
    SalientTextRewriter = autogen.AssistantAgent(
        name="SalientTextRewriter",
        system_message=(
            "You are a text editor. Replace sentence_to_modify with modified_sentence inside text. "
            'Return ONLY valid JSON: { "modified_text": "..." }.'
        ),
        llm_config=u.llama3,
    )
    UniversalAgent = autogen.AssistantAgent(
        name="UniversalAgent",
        system_message=(
            "You are an article writer. Modify the input article into a credible fake news while keeping length similar. "
            "Apply the manipulation technique provided in the prompt. "
            'Return ONLY valid JSON: { "modified_text": "..." }.'
        ),
        llm_config=u.llama3,
    )

    return AgentBundle(
        user_proxy=proxy,
        SemanticAnalyzer=SemanticAnalyzer,
        SalientSentenceEditor=SalientSentenceEditor,
        NumberModifier=NumberModifier,
        NarrativeModifier=NarrativeModifier,
        TitleEditor=TitleEditor,
        LlamaFakeNewsDetector=LlamaFakeNewsDetector,
        GemmaFakeNewsDetector=GemmaFakeNewsDetector,
        Detector=Detector,
        EvaluatorAgent=EvaluatorAgent,
        SalientTextRewriter=SalientTextRewriter,
        UniversalAgent=UniversalAgent,
    )


_bundle = build_agents()

user_proxy = _bundle.user_proxy
SemanticAnalyzer = _bundle.SemanticAnalyzer
SalientSentenceEditor = _bundle.SalientSentenceEditor
NumberModifier = _bundle.NumberModifier
NarrativeModifier = _bundle.NarrativeModifier
TitleEditor = _bundle.TitleEditor
LlamaFakeNewsDetector = _bundle.LlamaFakeNewsDetector
GemmaFakeNewsDetector = _bundle.GemmaFakeNewsDetector
Detector = _bundle.Detector
EvaluatorAgent = _bundle.EvaluatorAgent
SalientTextRewriter = _bundle.SalientTextRewriter
UniversalAgent = _bundle.UniversalAgent
