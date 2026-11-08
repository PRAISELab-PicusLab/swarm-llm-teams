# Swarm of LLM Teams vs. a Fake-News Detector (Apart Swarm Alignment 2026, Track 2)

How do communication structure and incentives change what a **swarm of LLM teams** achieves? Five teams of agents turn
authentic news articles into subtly false versions and try to have them classified as REAL by a fixed detector. Model,
prompt and article sequence are the same in every scenario; only the shared channel and the incentives change.

This repository contains **code only**: no outputs, datasets or generated articles are released.

## Pre-existing vs. hackathon code

**Pre-existing** (built before the hackathon for an earlier paper): everything in `pipeline/`, `detector/`
(except `lennywinks.py`) and `prompts/`.

**Written during the hackathon (6–8 Nov 2026):**
- `swarm/`: game engine, team coordinator, scenarios, dual-SHAP feedback, detector screen.
- `detector/lennywinks.py` plus its `DETECTOR_FUNCS` entry; the detector was selected with `swarm/detector_screen/screen.py`.
- `pipeline/utils.py`: one line, the `OLLAMA_MODEL` environment override.
- `analysis_tools/`: agreement metrics, post vs team-memory NLI contradiction, round watcher.

## Design

- **Unit.** A team is the sequential pipeline SA→SSE→NM→NaM→TE (Semantic Analyzer, Salient Sentence Editor, Number
  Modifier, Narrative Modifier, Title Editor) plus a **team coordinator**.
- **Round.** Every team still in play gets a different article; slot `i` at round `r` gets `pool[r·N+i]`, the same in
  every scenario. The team runs one full pass and the detector returns REAL (+1 point) or FAKE (−1 life).
- **Feedback.** Dual SHAP from a frozen surrogate (logistic regression on edit features): the top features that pushed
  toward FAKE and toward REAL, each with its agent.
- **Memory.** The coordinator reads its team memory (plus the collective memory or the public board, depending on the
  scenario) and writes lessons plus one instruction per agent; every agent reads the team memory before editing.

| Scenario | Name | Shared channel | Lives |
|---|---|---|---|
| 0 | baseline | none (team memory only) | 3 per team |
| 1 | cooperative | collective memory (all coordinators read and write) | 3 per team |
| 2 | competitive | public board (share / hide / misreport; self-reported `post_is_truthful`) | 3 per team |
| 3 | competitive, shared lives | public board | one common pool (`--shared-lives`, default 10) |
| 4 | individuals → swarm | 25 agents choose chain slots; stable chains become teams | 3 per agent |
| 5 | bad apple | as 1, but one team (`--bad-apple`, default T0) silently gets the competitive prompt | 3 per team |

`2.1`, `3.1`, `5.1` = scenarios 2, 3, 5 plus one sentence stating that reading the board is free.

## Requirements

- Python with `requirements.txt` (plus `torch`).
- [Ollama](https://ollama.com) serving the agent model (e.g. `llama3.1:8b`); set `OLLAMA_MODEL` to it.
  `OLLAMA_NUM_PARALLEL=5` lets the five teams run in parallel; use `OLLAMA_NUM_CTX=16384` with `--memory-v3`.
- Hugging Face models: `Lennywinks/fake-news-detector-english` (detector), `sentence-transformers/all-MiniLM-L6-v2`,
  `textattack/roberta-base-MNLI`, `distilgpt2`; NLTK data and `rouge`.
- A CSV of authentic articles with columns `ID, title, text` (not included).

## Run

**Phase 1** — generate the round-0 cohort and train the frozen surrogate:

```bash
python -m pipeline runall --dataset <articles.csv> --detector lennywinks --mode keep --desired-n 300 \
    --max-round -1 --no-drift-retrain --output-root <out>/phase1
```

**Scenarios** — one process per scenario (the official coordinator prompt is `--memory-v3 --post-rule team`):

```bash
python -m swarm --scenario 1 --phase1-root <out>/phase1 --out <out>/s1 \
    --teams 5 --max-rounds 20 --lives 3 --memory-v3 --post-rule team
python -m swarm --scenario 3 --phase1-root <out>/phase1 --out <out>/s3 --shared-lives 15 --memory-v3 --post-rule team
```

Other options: `--bad-apple N`, `--coord-model M` (separate Ollama model for the coordinator only), `--seed`,
`--no-resume`. Each scenario writes:
- `state.json`: resume point, final memories and board;
- `rounds/round_RRR/results.csv`: one row per team;
- `rounds/round_RRR/events.json`: feedback, agent rationales, coordinator outputs.

**Analysis** (run from the folder that contains the scenario outputs):

```bash
python analysis_tools/agreement_metrics.py judge  <new_outdir> --versions <run>
python analysis_tools/agreement_metrics.py compute <new_outdir> --versions <run>
python analysis_tools/nli_post_vs_team.py <new_outdir> <run>[:label][:s1,s2,...]
```

The analysis scripts refuse to write into an existing output folder.
