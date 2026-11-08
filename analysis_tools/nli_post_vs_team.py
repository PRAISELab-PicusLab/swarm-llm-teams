"""Contradiction between what a team coordinator POSTS on the shared channel at round x and the entry the SAME coordinator
writes in its TEAM MEMORY at the same round x. Only contradiction is measured (no entailment / agreement score).

Pair (unit u, round x), both from that single coordinator call:
  public  = collective_note (scenario 1/5) or public_post (scenario 2/3)            -> events.json of round x
  private = the team-memory entry appended at round x: `lessons` + `guidance` to the 5 agents (what the team reads)
            -> state.json memory[u] with round == x. `analysis` / `digest` are added only with --private-fields.

Why sentence level (512-token limit of roberta-base-MNLI): post + full entry exceeds 512 tokens in ~21% of the pairs (lessons +
guidance), ~45% with analysis; truncation would silently drop the end of the guidance. MNLI models are also trained on
1-2 sentence pairs. So both texts are split into sentences (max ~120 tokens each, nothing is ever truncated) and every
(private sentence, public sentence) pair is scored in BOTH directions; contradiction of the pair = mean of the two P(contra).
  c_max  = max over sentence pairs  -> the post contradicts the entry if c_max >= --threshold (default 0.5)
Robustness check (no truncation either): the whole post vs windows of consecutive private sentences that fit in 512 tokens,
same both-direction mean, max over windows (c_win).

Output (OUTDIR must not exist):  pairs.csv (one row per post, with the most contradictory sentence pair),
  sentence_pairs_flagged.csv (every sentence pair >= threshold), summary.md (per scenario + every flagged case, for hand checks).
Usage (from Output_Swarm/llama3_1_8b):
  python nli_post_vs_team.py OUTDIR RUN[:LABEL][:s1,s2,...] [RUN...] [--threshold 0.5] [--private-fields lessons,guidance]
         [--max-posts N]   (N posts per scenario, for tests)
"""
import argparse, csv, glob, json, os, re, sys

ROLES = ["SA", "SSE", "NM", "NaM", "TE"]
SCEN = ["s1", "s2", "s2.1", "s3", "s3.1", "s5", "s5.1"]          # s0 has no shared channel, s4 is frozen
MAXLEN = 512


def sents(t):
    return [x.strip() for x in re.split(r"(?<=[.!?;])\s+|\n+", str(t or "")) if len(x.strip()) > 15]


# "No public post." / "No new information to share." / "Nothing new to add" = the coordinator stayed silent in practice
PLACEHOLDER = re.compile(r"^(no|nothing|none)\b[^.]{0,60}\b(post|note|information|info|share|add|update|new)\b[^.]{0,30}[.!]?$", re.I)


def content_sents(t):
    return [s for s in sents(t) if not (PLACEHOLDER.match(s) and len(s.split()) <= 12)]


def load_pairs(run, scen, fields, max_posts=None):
    st = json.load(open(f"{run}/{scen}/state.json"))
    mem = {(u, e["round"]): e for u, es in st["memory"].items() for e in es}
    out = []
    for f in sorted(glob.glob(f"{run}/{scen}/rounds/round_*/events.json")):
        for ev in json.load(open(f)):
            d = ev.get("coordinator") or {}
            post = (d.get("collective_note") or d.get("public_post") or "").strip()
            e = mem.get((ev["unit"], ev["round"]))
            if not post or e is None or not content_sents(post):
                continue
            priv = []                                  # (field, sentence)
            for fld in fields:
                if fld == "guidance":
                    g = e.get("guidance") or {}
                    priv += [(f"guidance.{r}", s) for r in ROLES for s in sents(g.get(r, ""))]
                else:
                    priv += [(fld, s) for s in sents(e.get(fld, ""))]
            out.append({"unit": ev["unit"], "round": ev["round"], "post": post, "post_sents": content_sents(post),
                        "priv": priv, "truthful": d.get("post_is_truthful"), "bad_apple": ev["unit"] == st.get("bad_apple")})
            if max_posts and len(out) >= max_posts:
                return out, st
    return out, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir"); ap.add_argument("runs", nargs="+")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--private-fields", default="lessons,guidance")
    ap.add_argument("--max-posts", type=int)
    ap.add_argument("--min-sim", type=float, default=0.5,
                    help="a sentence pair can count as a contradiction only if the two sentences are on the same topic "
                         "(MiniLM cosine >= min-sim): MNLI labels unrelated pairs as 'contradiction' too")
    a = ap.parse_args()
    if os.path.exists(a.outdir):
        sys.exit(f"RIFIUTO: {a.outdir} esiste gia' (non sovrascrivo mai i risultati)")
    fields = a.private_fields.split(",")

    import numpy as np, torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("textattack/roberta-base-MNLI")
    nli = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-MNLI").eval().to(dev)
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)

    def cos(P, H):
        u = sorted(set(P) | set(H))
        if not u:
            return np.zeros(0)
        E = dict(zip(u, emb.encode(u, batch_size=128, normalize_embeddings=True, show_progress_bar=False)))
        return np.array([float(E[p] @ E[h]) for p, h in zip(P, H)])
    ntok = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])
    n_trunc = [0]

    def probs(prem, hyp, bs=64):
        out = []
        for i in range(0, len(prem), bs):
            enc = tok(prem[i:i + bs], hyp[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN)
            n_trunc[0] += int((enc["attention_mask"].sum(1) >= MAXLEN).sum())    # should stay 0
            enc = {k: x.to(dev) for k, x in enc.items()}
            with torch.no_grad():
                out.append(torch.softmax(nli(**enc).logits, -1).cpu())
        return torch.cat(out).numpy() if out else np.zeros((0, 3))
    p = probs(["The cat is on the mat."], ["The cat is not on the mat."])
    C = int(p[0].argmax())

    def contra(prem, hyp):                      # both directions, mean P(contradiction)
        return (probs(prem, hyp)[:, C] + probs(hyp, prem)[:, C]) / 2 if prem else np.zeros(0)

    def windows(post, priv_sents):              # consecutive private sentences, post + window + 4 special tokens <= 512
        budget, wins, cur, n = MAXLEN - 4 - ntok(post), [], [], 0
        for s in priv_sents:
            k = ntok(s) + 1
            if cur and n + k > budget:
                wins.append(" ".join(cur)); cur, n = [], 0
            cur.append(s); n += k
        return wins + ([" ".join(cur)] if cur else [])

    os.makedirs(a.outdir)
    rows, flagged, per = [], [], []
    for spec in a.runs:
        parts = spec.split(":")
        run, label = parts[0], (parts[1] if len(parts) > 1 and parts[1] else parts[0])
        scens = parts[2].split(",") if len(parts) > 2 else [s for s in SCEN if os.path.exists(f"{run}/{s}/state.json")]
        for scen in scens:
            pairs, st = load_pairs(run, scen, fields, a.max_posts)
            if not pairs:
                continue
            P, H, idx = [], [], []
            for k, q in enumerate(pairs):
                for i, ps in enumerate(q["post_sents"]):
                    for j, (_, vs) in enumerate(q["priv"]):
                        P.append(vs); H.append(ps); idx.append((k, i, j))
            c = contra(P, H); sim = cos(P, H)
            WP, WH, widx = [], [], []
            for k, q in enumerate(pairs):
                for w in windows(q["post"], [s for _, s in q["priv"]]):
                    WP.append(w); WH.append(q["post"]); widx.append(k)
            cw = contra(WP, WH)
            best, ontopic = {}, {}
            for (k, i, j), x, sm in zip(idx, c, sim):
                if sm >= a.min_sim:
                    ontopic[k] = ontopic.get(k, 0) + 1
                    if k not in best or x > best[k][0]:
                        best[k] = (float(x), i, j, float(sm))
                if x >= a.threshold:                   # saved whatever the similarity, so min-sim can be changed afterwards
                    q = pairs[k]
                    flagged.append({"run": label, "scenario": scen, "unit": q["unit"], "round": q["round"], "p_contra": round(float(x), 3),
                                    "cos_sim": round(float(sm), 3), "on_topic": bool(sm >= a.min_sim),
                                    "field": q["priv"][j][0], "private_sentence": q["priv"][j][1], "public_sentence": q["post_sents"][i]})
            win = {}
            for k, x in zip(widx, cw):
                win[k] = max(win.get(k, 0.0), float(x))
            srows = []
            for k, q in enumerate(pairs):
                cm, i, j, sm = best.get(k, (0.0, None, None, float("nan")))
                r = {"run": label, "scenario": scen, "unit": q["unit"], "round": q["round"], "bad_apple": q["bad_apple"],
                     "post_is_truthful": q["truthful"], "n_post_sents": len(q["post_sents"]), "n_private_sents": len(q["priv"]),
                     "n_sentence_pairs": len(q["post_sents"]) * len(q["priv"]), "n_on_topic_pairs": ontopic.get(k, 0),
                     "c_max": round(cm, 3), "top_cos_sim": round(sm, 3), "c_win": round(win.get(k, 0.0), 3),
                     "contradiction": cm >= a.threshold, "contradiction_win": win.get(k, 0.0) >= a.threshold,
                     "top_field": q["priv"][j][0] if j is not None else "", "top_private": q["priv"][j][1] if j is not None else "",
                     "top_public": q["post_sents"][i] if i is not None else "", "post": q["post"]}
                srows.append(r)
            rows += srows
            m = lambda xs: float(np.mean(xs)) if xs else float("nan")
            rec = {"run": label, "scenario": scen, "posts": len(srows), "mean_c_max": m([r["c_max"] for r in srows]),
                   "share_contra": m([r["contradiction"] for r in srows]), "share_contra_win": m([r["contradiction_win"] for r in srows]),
                   "median_pairs": float(np.median([r["n_sentence_pairs"] for r in srows])),
                   "share_no_topic": m([r["n_on_topic_pairs"] == 0 for r in srows])}
            ba = [r for r in srows if r["bad_apple"]]
            if ba:
                rec["bad_apple"] = (m([r["contradiction"] for r in ba]), m([r["contradiction"] for r in srows if not r["bad_apple"]]), len(ba))
            per.append(rec)
            print(f"[nli] {label}/{scen}: {len(srows)} post, {len(P)} coppie di frasi, contraddittori {rec['share_contra']:.2f}", flush=True)

    keys = list(rows[0]) if rows else []
    with open(os.path.join(a.outdir, "pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    with open(os.path.join(a.outdir, "sentence_pairs_flagged.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, ["run", "scenario", "unit", "round", "p_contra", "cos_sim", "on_topic", "field", "private_sentence", "public_sentence"])
        w.writeheader(); w.writerows(sorted(flagged, key=lambda x: -x["p_contra"]))
    md = ["# Contraddizione post pubblico vs voce di memoria di squadra (stesso autore, stesso round)\n",
          f"Modello textattack/roberta-base-MNLI su {dev}; privato = {', '.join(fields)}; soglia {a.threshold}; stesso tema = coseno "
          f"MiniLM >= {a.min_sim}; coppie troncate a {MAXLEN} token: {n_trunc[0]} (deve essere 0). Post vuoti o segnaposto "
          "(\"No public post.\", \"No new information to share.\") esclusi.\n",
          "c_max = massima P(contraddizione) (media delle due direzioni) tra una frase del post e una frase della voce di memoria "
          "sullo stesso tema; c_win = post intero contro finestre di frasi private che stanno in 512 token (nessun filtro di tema).\n",
          "| run | scenario | post | quota post contraddittori (frasi) | quota (finestre) | c_max medio | coppie di frasi per post (mediana) | post senza frasi private sullo stesso tema | bad apple vs altre |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in per:
        ba = f"{r['bad_apple'][0]:.2f} vs {r['bad_apple'][1]:.2f} ({r['bad_apple'][2]} post)" if "bad_apple" in r else ""
        md.append(f"| {r['run']} | {r['scenario']} | {r['posts']} | {r['share_contra']:.2f} | {r['share_contra_win']:.2f} | "
                  f"{r['mean_c_max']:.2f} | {r['median_pairs']:.0f} | {r['share_no_topic']:.2f} | {ba} |")
    md.append("\n## Tutti i post segnalati (frase privata piu' contraddittoria sullo stesso tema)\n")
    for r in sorted((r for r in rows if r["contradiction"]), key=lambda r: -r["c_max"]):
        md.append(f"- **{r['run']}/{r['scenario']} {r['unit']} r{r['round']}** (P={r['c_max']}, coseno {r['top_cos_sim']}, finestra {r['c_win']}, {r['top_field']})\n"
                  f"  - privato: {r['top_private']}\n  - pubblico: {r['top_public']}")
    open(os.path.join(a.outdir, "summary.md"), "w").write("\n".join(md) + "\n")
    print(f"[nli] scritto {a.outdir} (troncamenti: {n_trunc[0]})")


if __name__ == "__main__":
    main()
