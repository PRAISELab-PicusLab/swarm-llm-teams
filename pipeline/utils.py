import json
import os
import re
import numpy as np
import pandas as pd
import autogen
import csv
import argparse
from tqdm import tqdm

# llama3 = {
#     "config_list": [
#         {
#             "model": "llama3.1:8b",
#             "base_url": "http://localhost:11434/v1",
#             "api_key": "ollama",
#         }
#     ],
#     "cache_seed": 42,  # Disable caching.
# }

llama3 = {
    "config_list": [
        {
            "model": os.getenv("OLLAMA_MODEL", "llama3.3:70b"),  # swarm runs set llama3.1:8b / gemma4:31b
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "api_key": "ollama",
            "api_type": "ollama",
            "native_format": True,
            "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.5")),
            # OLLAMA_NUM_CTX default 32768 ok per articoli medi. Per articoli lunghi
            # (previous_article + current_article + trajectory_feedback) consigliato 65536.
            # llama3.3:70b supporta fino a 128k.
            "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "32768")),
        }
    ],
    "cache_seed": None,  # Disable caching (None = truly disabled).
}

gemma = {
    "config_list": [
        {
            "model": "gemma3:27b",
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "api_key": "ollama",
            "api_type": "ollama",
            "native_format": True,
            "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.5")),
            # Vedi commento su llama3.num_ctx — opzionale 65536 per payload grandi.
            "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "32768")),
        }
    ],
    "cache_seed": None,  # Disable caching (None = truly disabled).
}


def read_from_file(filename, folder):
    # Componi il percorso completo del file
    file_path = os.path.join(folder, filename)
    # print(f"Reading from file: {file_path}")

    if not os.path.exists(file_path):
        return "File not found."  # Il file non esiste

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def measure_agent_time(agent_name, function, *args, **kwargs):
    response = function(*args, **kwargs)
    return response


def from_string_to_json(text):
    try:
        # Prova a caricare la stringa JSON utilizzando il modulo json
        json_data = json.loads(text)
        return json_data
    except json.JSONDecodeError:
        # Se il parsing JSON fallisce, prova a estrarre il primo blocco JSON
        match = re.search(r'{(.*?)}', text, re.DOTALL)
        if match:
            json_text = match.group(1)
            json_text = json_text.strip()
            try:
                # Tentativo di caricare il testo JSON estratto e formattarlo con i doppi apici
                json_data = json.loads("{" + json_text + "}")
                return json_data
            except json.JSONDecodeError as e:
                print(f"JSON string parsing error: {e}")

    # Se tutto fallisce, restituisci un dizionario vuoto
    return {}
