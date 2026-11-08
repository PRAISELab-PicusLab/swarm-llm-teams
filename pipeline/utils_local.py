# utils_local.py
import json
import os
import re
import numpy as np
import pandas as pd
import autogen
import csv
import argparse
from tqdm import tqdm

"""
Versione locale per test su Ollama con modelli leggeri.

"""


llama3 = {
    "config_list": [
        {
            "model": os.getenv("OLLAMA_LLAMA_MODEL", "llama3.2:1b"),
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "api_key": "ollama",
            "api_type": "ollama",
            "native_format": True,
            "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.5")),
            "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "4096")),
        }
    ],
    "cache_seed": 42,
}

gemma = {
    "config_list": [
        {
            "model": os.getenv("OLLAMA_GEMMA_MODEL", "gemma2:2b"),
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "api_key": "ollama",
            "api_type": "ollama",
            "native_format": True,
            "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.5")),
            "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "4096")),
        }
    ],
    "cache_seed": 42,
}

def read_from_file(filename, folder):
    file_path = os.path.join(folder, filename)
    if not os.path.exists(file_path):
        return "File not found."
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def measure_agent_time(agent_name, function, *args, **kwargs):
    response = function(*args, **kwargs)
    return response

def from_string_to_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'{(.*?)}', text, re.DOTALL)
        if match:
            json_text = match.group(1).strip()
            try:
                return json.loads("{" + json_text + "}")
            except json.JSONDecodeError as e:
                print(f"JSON string parsing error: {e}")
    return {}
