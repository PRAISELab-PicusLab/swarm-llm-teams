"""One batched line per newly finished round of a set of swarm scenarios, plus job failures. Re-arm safe (state file).
Usage: watch_rounds.py ROOT SEEN_FILE LABEL scen=jobid [scen=jobid ...]
  ROOT = e.g. .../Output_Swarm/llama3_1_8b/official_coord-llama8b ; SEEN_FILE = where already-reported rounds are kept."""
import csv, glob, json, os, subprocess, sys, time

ROOT, SEEN, LABEL = sys.argv[1], sys.argv[2], sys.argv[3]
JOBS = dict(x.split("=") for x in sys.argv[4:])
SCEN = list(JOBS)
seen = json.load(open(SEEN)) if os.path.exists(SEEN) else {}
jstate = {}


def jl(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def line(s, r):
    rows = list(csv.DictReader(open(f"{ROOT}/{s}/rounds/round_{r:03d}/results.csv")))
    ev = jl(f"{ROOT}/{s}/rounds/round_{r:03d}/events.json") or []
    real = sum(x["label"] == "REAL" for x in rows); fake = sum(x["label"] == "FAKE" for x in rows)
    posts = [(e["unit"], (e["coordinator"].get("collective_note") or e["coordinator"].get("public_post") or "").strip()) for e in ev]
    empty = sum(1 for e in ev if not e["coordinator"].get("lessons"))
    out = f"[{LABEL} {s}] r{r}: {real}R/{fake}F | post {sum(bool(t) for _, t in posts)}/{len(rows)}"
    if empty:
        out += f" | OUTPUT COORDINATORE VUOTO {empty}/{len(rows)}"
    if s in ("s5", "s5.1"):
        t0 = next((t for u, t in posts if u == "T0"), None)
        out += " | T0 " + ("eliminata" if not any(x["unit"] == "T0" for x in rows) else ("post" if t0 else "SILENZIO"))
    out += " | punti " + " ".join(f"{x['unit']}:{x['score']}" for x in rows)
    out += (f" | vite condivise {rows[-1]['shared_lives']}" if s.startswith("s3") else " | vite " + " ".join(f"{x['unit']}:{x['lives']}" for x in rows))
    return out


while True:
    batch = []
    for s in SCEN:
        rs = sorted(int(p.split("round_")[1][:3]) for p in glob.glob(f"{ROOT}/{s}/rounds/round_*/results.csv"))
        for r in rs:
            if r > seen.get(s, -1):
                batch.append(line(s, r)); seen[s] = r
        st = jl(f"{ROOT}/{s}/state.json")
        if st and st.get("finished") and f"{s}_final" not in seen:
            batch.append(f"[{LABEL} {s}] TERMINATO ({st.get('end_reason')}) dopo {st['round']} round | punti finali "
                         + " ".join(f"{u}:{v}" for u, v in st["scores"].items())
                         + (f" | vite condivise {st['shared_lives']}" if s.startswith("s3") else " | vite " + " ".join(f"{u}:{v}" for u, v in st["lives"].items())))
            seen[f"{s}_final"] = True
    out = subprocess.run(["sacct", "-n", "-X", "-P", "-j", ",".join(JOBS.values()), "-o", "JobID,JobName,State,Elapsed"],
                         universal_newlines=True, stdout=subprocess.PIPE).stdout
    for l in out.strip().splitlines():
        jid, name, state, el = (l.split("|") + ["", "", "", ""])[:4]
        state = state.split()[0] if state else "?"
        if jstate.get(jid) != state:
            jstate[jid] = state
            if state in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"):
                batch.append(f"!! job {jid} {name} -> {state} (elapsed {el})")
            elif state == "RUNNING":
                batch.append(f"job {jid} -> RUNNING")
    if batch:
        print("\n".join(batch), flush=True)
    json.dump(seen, open(SEEN, "w"))
    if all(seen.get(f"{s}_final") for s in SCEN):
        print(f"TUTTI GLI SCENARI {LABEL} TERMINATI", flush=True)
        break
    time.sleep(60)
