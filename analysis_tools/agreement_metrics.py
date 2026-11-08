"""Non-lexical agreement metrics for the swarm runs (v2, v3). Never overwrites: everything goes into OUTDIR, which must be new
(the `judge` step may resume inside the same OUTDIR).

  judge   OUTDIR   metric 1 - an LLM judge (Ollama, default qwen3:30b-a3b-instruct) labels the strategy each coordinator gives
                   its five agents, with a fixed taxonomy -> OUTDIR/strategy_labels.jsonl
  compute OUTDIR   metrics 1 (agreement on the judge labels), 2 (behaviour: the 23 per-article edit metrics),
                   3 (NLI agreement between teams' posts), 5 (semantic influence), 6 (private-public consistency)
                   -> OUTDIR/*.csv + OUTDIR/summary.md

Run from Output_Swarm/llama3_1_8b (paths v2/<scenario>, v3/<scenario>). Options: --only v3/s5,v3/s1 (subset, for tests),
--max-rounds N (tests).
"""
import argparse, collections, glob, json, math, os, random, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

VERSIONS = ["v2", "v3"]   # overridden by --versions (e.g. official_coord-llama8b)
SCENARIOS = ["s0", "s1", "s2", "s2.1", "s3", "s3.1", "s5", "s5.1"]
ROLES = ["SA", "SSE", "NM", "NaM", "TE"]
TAXONOMY = {
    "SA": {"preserve_original_selection": "extract the original key sentences/numbers faithfully, keep them as they are",
           "expand_or_add_items": "add or select more key sentences/numbers, or new ones not in the original",
           "reduce_or_omit_items": "select fewer key sentences/numbers or drop some",
           "unspecified": "no editing direction given"},
    "SSE": {"no_change": "leave the salient sentences unchanged",
            "neutral_rephrase": "rephrase with neutral wording, keeping the meaning",
            "subtle_meaning_change": "alter the meaning subtly (shift emphasis, nuance, implication)",
            "emotional_or_loaded": "add emotional, sensational or loaded wording",
            "unspecified": "no editing direction given"},
    "NM": {"no_number_change": "do not change numbers (or only if strictly necessary)",
           "small_plausible_change": "small, plausible changes to existing numbers",
           "large_or_new_numbers": "large changes or new numbers",
           "unspecified": "no editing direction given"},
    "NaM": {"no_additions_preserve": "add nothing new, preserve the article as it is",
            "neutral_rephrase": "rephrase neutrally without adding claims",
            "loaded_or_emotional_language": "use loaded, emotional or sensational language",
            "new_claims_or_context": "add new claims, causes, context or details not in the original",
            "named_persuasion_technique": "apply a named persuasion technique (e.g. causal oversimplification, whataboutism)",
            "unspecified": "no editing direction given"},
    "TE": {"keep_title_identical": "keep the title identical or almost identical",
           "minor_reword_same_structure": "minor rewording keeping length, capitalization and structure",
           "substantial_rewrite_or_sensational": "substantial rewrite, change of length/structure, or a more sensational title",
           "unspecified": "no editing direction given"},
}
INTENSITY = ["none", "minimal", "moderate", "aggressive"]


# ----------------------------------------------------------------------------------------------- data
def selected(args):
    if args.only:
        return [tuple(x.split("/")) for x in args.only.split(",")]
    return [(v, s) for v in VERSIONS for s in SCENARIOS if os.path.exists(f"{v}/{s}/state.json")]


def events(v, s, max_rounds=None):
    out = []
    for f in sorted(glob.glob(f"{v}/{s}/rounds/round_*/events.json")):
        r = int(f.split("round_")[1][:3])
        if max_rounds is not None and r >= max_rounds:
            continue
        out += json.load(open(f))
    return out


def post_of(d):
    return (d.get("collective_note") or d.get("public_post") or "").strip()


def private_of(d):
    g = d.get("guidance") or {}
    return " ".join(x for x in [str(d.get("digest", "")), str(d.get("lessons", "")), str(d.get("analysis", ""))]
                    + [f"{k}: {g[k]}" for k in ROLES if g.get(k)] if x)


# ----------------------------------------------------------------------------------------------- metric 1: judge
JUDGE_SYSTEM = (
    "You annotate the editing STRATEGY that a team coordinator instructs its five agents to follow when they turn an authentic "
    "news article into a subtly false one. For each agent pick exactly ONE category from its list, based only on what the "
    "instruction tells the agent to do (not on what the coordinator hopes). Use 'unspecified' only if the instruction gives no "
    "editing direction at all. Then rate the overall editing intensity the instructions ask for.\n\nCATEGORIES:\n"
    + "\n".join(f"{r}: " + "; ".join(f"{k} = {v}" for k, v in TAXONOMY[r].items()) for r in ROLES)
    + "\nINTENSITY: none = no change at all; minimal = tiny, surface-level edits; moderate = noticeable edits to content; "
      "aggressive = substantial rewriting or added content.")


def judge_schema():
    props = {r: {"type": "string", "enum": list(TAXONOMY[r])} for r in ROLES}
    props["intensity"] = {"type": "string", "enum": INTENSITY}
    return {"type": "object", "properties": props, "required": ROLES + ["intensity"], "additionalProperties": False}


def call_ollama(model, system, user, schema, base):
    payload = {"model": model, "stream": False, "think": False, "format": schema,
               "options": {"temperature": 0, "num_ctx": 8192},
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    req = urllib.request.Request(base + "/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(json.loads(resp.read())["message"]["content"])


def cmd_judge(args):
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, "strategy_labels.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path):
            x = json.loads(l); done.add((x["version"], x["scenario"], x["unit"], x["round"]))
    items = []
    for v, s in selected(args):
        for e in events(v, s, args.max_rounds):
            key = (v, s, e["unit"], e["round"])
            if key in done:
                continue
            d = e.get("coordinator") or {}
            g = d.get("guidance") or {}
            user = ("INSTRUCTIONS TO THE AGENTS:\n" + "\n".join(f"{r}: {g.get(r, '')}" for r in ROLES)
                    + f"\n\nCOORDINATOR LESSONS (context): {str(d.get('lessons', ''))[:1500]}")
            items.append((key, user))
    print(f"[judge] {len(items)} team-round da etichettare ({len(done)} gia' fatti)", flush=True)
    base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    schema, lock = judge_schema(), __import__("threading").Lock()
    t0 = time.time()

    def work(it):
        key, user = it
        for attempt in range(3):
            try:
                lab = call_ollama(args.model, JUDGE_SYSTEM, user, schema, base)
                break
            except Exception as ex:
                lab = {"error": str(ex)[:200]}; time.sleep(2)
        rec = dict(zip(["version", "scenario", "unit", "round"], key)); rec.update(lab)
        with lock:
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, _ in enumerate(ex.map(work, items)):
            if i % 100 == 0:
                print(f"[judge] {i}/{len(items)} in {time.time() - t0:.0f}s", flush=True)
    print(f"[judge] fatto in {time.time() - t0:.0f}s -> {path}", flush=True)


# ----------------------------------------------------------------------------------------------- stats helpers
def kripp_alpha_nominal(units):
    """units: list of lists of category labels (one list per unit, values from different coders). Krippendorff's alpha, nominal."""
    coinc = collections.Counter(); n_tot = 0
    for vals in units:
        m = len(vals)
        if m < 2:
            continue
        c = collections.Counter(vals)
        for a in c:
            for b in c:
                coinc[(a, b)] += c[a] * (c[b] - (a == b)) / (m - 1)
        n_tot += m
    if n_tot == 0:
        return float("nan")
    nc = collections.Counter()
    for (a, b), x in coinc.items():
        nc[a] += x
    Do = sum(x for (a, b), x in coinc.items() if a != b)
    De = sum(nc[a] * nc[b] for a in nc for b in nc if a != b) / max(1, (n_tot - 1))
    return 1 - Do / De if De > 0 else 1.0


def entropy(counter):
    n = sum(counter.values())
    return -sum(c / n * math.log2(c / n) for c in counter.values() if c) if n else float("nan")


def jsd(p, q):
    keys = set(p) | set(q); sp, sq = sum(p.values()), sum(q.values())
    P = {k: p.get(k, 0) / sp for k in keys}; Q = {k: q.get(k, 0) / sq for k in keys}
    M = {k: (P[k] + Q[k]) / 2 for k in keys}
    kl = lambda A: sum(A[k] * math.log2(A[k] / M[k]) for k in keys if A[k] > 0)
    return (kl(P) + kl(Q)) / 2


def mean(x):
    x = [v for v in x if v == v]
    return sum(x) / len(x) if x else float("nan")


def slope(xs, ys):
    pts = [(a, b) for a, b in zip(xs, ys) if b == b]
    if len(pts) < 3:
        return float("nan")
    mx = mean([a for a, _ in pts]); my = mean([b for _, b in pts])
    den = sum((a - mx) ** 2 for a, _ in pts)
    return sum((a - mx) * (b - my) for a, b in pts) / den if den else float("nan")


def fmt(x, d=3):
    return "n/a" if x != x else f"{x:.{d}f}"


# ----------------------------------------------------------------------------------------------- compute
def cmd_compute(args):
    import numpy as np, pandas as pd, torch
    out = args.outdir
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "summary.md")):
        sys.exit(f"RIFIUTO: {out}/summary.md esiste gia' (non sovrascrivo)")
    sel = selected(args)
    md = ["# Metriche di accordo (non lessicali)\n", f"Scenari: {', '.join(v + '/' + s for v, s in sel)}\n"]

    # ---------------- metric 1
    lp = os.path.join(out, "strategy_labels.jsonl")
    if os.path.exists(lp):
        labs = [json.loads(l) for l in open(lp)]
        labs = [x for x in labs if "error" not in x and (x["version"], x["scenario"]) in set(sel)]
        rows, dist = [], {}
        for v, s in sel:
            L = [x for x in labs if x["version"] == v and x["scenario"] == s]
            if not L:
                continue
            by_round = collections.defaultdict(list)
            for x in L:
                by_round[x["round"]].append(x)
            rounds = sorted(by_round); half = rounds[len(rounds) // 2] if rounds else 0
            row = {"version": v, "scenario": s, "team_rounds": len(L)}
            for r_ in ROLES + ["intensity"]:
                units = [[x[r_] for x in by_round[r]] for r in rounds]
                row[f"alpha_{r_}"] = kripp_alpha_nominal(units)
                row[f"modal_{r_}"] = mean([collections.Counter(u).most_common(1)[0][1] / len(u) for u in units if len(u) >= 2])
                ent = [entropy(collections.Counter(u)) for u in units]
                row[f"entropy_early_{r_}"] = mean([e for r, e in zip(rounds, ent) if r < half])
                row[f"entropy_late_{r_}"] = mean([e for r, e in zip(rounds, ent) if r >= half])
                dist[(v, s, r_)] = collections.Counter(x[r_] for x in L)
            rows.append(row)
        df1 = pd.DataFrame(rows); df1.to_csv(os.path.join(out, "m1_strategy_agreement.csv"), index=False)
        # distribution and JSD
        dd = [{"version": v, "scenario": s, "role": r_, "category": c, "share": n / sum(cnt.values())}
              for (v, s, r_), cnt in dist.items() for c, n in cnt.items()]
        pd.DataFrame(dd).to_csv(os.path.join(out, "m1_strategy_distribution.csv"), index=False)
        jrows = []
        for v in VERSIONS:
            ss = [s for vv, s in sel if vv == v and (v, s, "NaM") in dist]
            for a in ss:
                for b in ss:
                    jrows.append({"version": v, "a": a, "b": b,
                                  "jsd_mean_roles": mean([jsd(dist[(v, a, r_)], dist[(v, b, r_)]) for r_ in ROLES + ["intensity"]])})
        pd.DataFrame(jrows).to_csv(os.path.join(out, "m1_strategy_jsd.csv"), index=False)
        md.append("\n## 1. Strategie (giudice LLM)\n")
        md.append("alpha = accordo di Krippendorff tra squadre nello stesso round (1 = tutte uguali, 0 = come il caso); "
                  "modale = quota media di squadre sulla strategia piu' comune del round.\n")
        md.append("| versione | scenario | alpha SA | alpha SSE | alpha NM | alpha NaM | alpha TE | alpha intensita' | modale medio | entropia intensita' inizio->fine |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, r in df1.iterrows():
            md.append(f"| {r.version} | {r.scenario} | " + " | ".join(fmt(r[f'alpha_{x}'], 2) for x in ROLES + ['intensity'])
                      + f" | {fmt(mean([r[f'modal_{x}'] for x in ROLES]), 2)} | {fmt(r['entropy_early_intensity'], 2)} -> {fmt(r['entropy_late_intensity'], 2)} |")
        md.append("\nStrategia piu' frequente per agente (quota):\n")
        for v, s in sel:
            if (v, s, "NaM") not in dist:
                continue
            md.append(f"- {v}/{s}: " + "; ".join(f"{r_} {dist[(v, s, r_)].most_common(1)[0][0]} ({dist[(v, s, r_)].most_common(1)[0][1] / sum(dist[(v, s, r_)].values()):.0%})" for r_ in ROLES + ["intensity"]))

    # ---------------- metric 2: behaviour
    feats_all = []
    for v, s in sel:
        for rd in sorted(glob.glob(f"{v}/{s}/rounds/round_*")):
            r = int(rd.split("round_")[1][:3])
            if args.max_rounds is not None and r >= args.max_rounds:
                continue
            try:
                m = pd.read_csv(f"{rd}/gen/agent_metrics.csv", dtype={"ID": str}); res = pd.read_csv(f"{rd}/results.csv", dtype={"article_id": str})
            except Exception:
                continue
            m = m.merge(res[["unit", "article_id"]], left_on="ID", right_on="article_id")
            m["version"], m["scenario"], m["round"] = v, s, r
            feats_all.append(m)
    if feats_all:
        F = pd.concat(feats_all, ignore_index=True)
        cols = [c for c in F.columns if c not in ("ID", "unit", "article_id", "version", "scenario", "round")]
        X = F[cols].astype(float); X = X.fillna(X.mean()); X = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
        F[cols] = X
        rows = []
        for (v, s), g in F.groupby(["version", "scenario"]):
            disp, cos_ = {}, {}
            for r, gr in g.groupby("round"):
                A = gr[cols].values
                if len(A) < 2:
                    continue
                disp[r] = float(np.mean(np.linalg.norm(A - A.mean(0), axis=1)))
                An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
                S = An @ An.T; iu = np.triu_indices(len(A), 1); cos_[r] = float(S[iu].mean())
            rr = sorted(disp); half = rr[len(rr) // 2] if rr else 0
            rows.append({"version": v, "scenario": s, "dispersion": mean(list(disp.values())),
                         "dispersion_early": mean([disp[r] for r in rr if r < half]), "dispersion_late": mean([disp[r] for r in rr if r >= half]),
                         "dispersion_slope_per_round": slope(rr, [disp[r] for r in rr]), "pairwise_cosine": mean(list(cos_.values()))})
        df2 = pd.DataFrame(rows); df2.to_csv(os.path.join(out, "m2_behaviour.csv"), index=False)
        md.append("\n## 2. Comportamento (23 metriche delle modifiche per articolo, standardizzate)\n")
        md.append("dispersione = distanza media delle squadre dal centroide del round (piu' bassa = fanno modifiche piu' simili); coseno = somiglianza media tra squadre.\n")
        md.append("| versione | scenario | dispersione | inizio -> fine | pendenza/round | coseno medio |")
        md.append("|---|---|---|---|---|---|")
        for _, r in df2.iterrows():
            md.append(f"| {r.version} | {r.scenario} | {fmt(r.dispersion, 2)} | {fmt(r.dispersion_early, 2)} -> {fmt(r.dispersion_late, 2)} | {fmt(r.dispersion_slope_per_round, 3)} | {fmt(r.pairwise_cosine, 2)} |")

    # ---------------- models for 3, 5, 6
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
    tok = AutoTokenizer.from_pretrained("textattack/roberta-base-MNLI")
    nli = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-MNLI").eval().to(dev)

    def nli_probs(prem, hyp, bs=32):
        outp = []
        for i in range(0, len(prem), bs):
            enc = tok(prem[i:i + bs], hyp[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=512)
            enc = {k: x.to(dev) for k, x in enc.items()}
            with torch.no_grad():
                outp.append(torch.softmax(nli(**enc).logits, -1).cpu())
        return torch.cat(outp).numpy() if outp else np.zeros((0, 3))
    # label indices: contradiction index from a clear contradiction, entailment from an identical pair
    p = nli_probs(["The cat is on the mat.", "The team scored ten points."], ["The cat is not on the mat.", "The team scored ten points."])
    C_IDX, E_IDX = int(p[0].argmax()), int(p[1].argmax())
    assert C_IDX != E_IDX, (C_IDX, E_IDX)
    md.append(f"\n(Modello NLI: indice contraddizione {C_IDX}, implicazione {E_IDX}; embedding: all-MiniLM-L6-v2 su {dev}.)\n")

    def agree(a, b):
        """NLI agreement in [-1, 1]: mean over both directions of P(entail) - P(contradict)."""
        if not a:
            return np.zeros(0)
        p1 = nli_probs(a, b); p2 = nli_probs(b, a)
        return ((p1[:, E_IDX] - p1[:, C_IDX]) + (p2[:, E_IDX] - p2[:, C_IDX])) / 2

    rows3, rows5, rows6, infl_rows = [], [], [], []
    rng = random.Random(0)
    for v, s in sel:
        ev = events(v, s, args.max_rounds)
        posts = {(e["unit"], e["round"]): post_of(e.get("coordinator") or {}) for e in ev}
        posts = {k: t for k, t in posts.items() if t}
        priv = {(e["unit"], e["round"]): private_of(e.get("coordinator") or {}) for e in ev}
        if not posts:
            continue
        bad = None
        st = json.load(open(f"{v}/{s}/state.json")); bad = st.get("bad_apple")
        # ---- metric 3: NLI agreement between different teams' posts, same round and next round
        pa, pb, kind, rr = [], [], [], []
        for (u, r), t in posts.items():
            for (u2, r2), t2 in posts.items():
                if u2 == u:
                    continue
                if r2 == r and u < u2:
                    pa.append(t); pb.append(t2); kind.append("stesso_round"); rr.append(r)
                elif r2 == r + 1:
                    pa.append(t); pb.append(t2); kind.append("round_successivo"); rr.append(r)
        sc = agree(pa, pb)
        if len(sc):
            half = max(rr) // 2
            k = np.array(kind); R = np.array(rr)
            rows3.append({"version": v, "scenario": s, "pairs": len(sc), "agreement": float(sc.mean()),
                          "agreement_same_round": float(sc[k == "stesso_round"].mean()) if (k == "stesso_round").any() else float("nan"),
                          "agreement_next_round": float(sc[k == "round_successivo"].mean()) if (k == "round_successivo").any() else float("nan"),
                          "share_contradictory_pairs": float((sc < -0.5).mean()),
                          "agreement_early": float(sc[R < half].mean()), "agreement_late": float(sc[R >= half].mean())})
        # ---- embeddings
        keys_p = list(posts); keys_v = [k for k in priv if priv[k]]
        Ep = dict(zip(keys_p, emb.encode([posts[k] for k in keys_p], normalize_embeddings=True, batch_size=64)))
        Ev = dict(zip(keys_v, emb.encode([priv[k] for k in keys_v], normalize_embeddings=True, batch_size=64)))
        units = sorted({u for u, _ in priv})

        # ---- metric 5: semantic influence (lift = after - before), with permutation null
        def lifts(post_round_of):
            per_pair = collections.defaultdict(list)
            for (a, r) in keys_p:
                rp = post_round_of[(a, r)]
                for b in units:
                    if b == a:
                        continue
                    after = [float(Ep[(a, r)] @ Ev[(b, rp + k)]) for k in (1, 2, 3) if (b, rp + k) in Ev]
                    before = [float(Ep[(a, r)] @ Ev[(b, rp - k)]) for k in (0, 1, 2) if (b, rp - k) in Ev]
                    if after and before:
                        per_pair[(a, b)].append(mean(after) - mean(before))
            return per_pair
        ident = {k: k[1] for k in keys_p}
        pp = lifts(ident)
        obs = mean([x for l in pp.values() for x in l])
        rounds_avail = sorted({r for _, r in priv})
        null = []
        for _ in range(200):
            perm = {k: rng.choice(rounds_avail) for k in keys_p}
            pl = lifts(perm); null.append(mean([x for l in pl.values() for x in l]))
        pval = (1 + sum(1 for x in null if x >= obs)) / (1 + len(null))
        out_infl = {a: mean([x for (aa, b), l in pp.items() if aa == a for x in l]) for a in units}
        for (a, b), l in pp.items():
            infl_rows.append({"version": v, "scenario": s, "from": a, "to": b, "lift": mean(l), "n": len(l)})
        top = sorted(out_infl.items(), key=lambda x: -x[1] if x[1] == x[1] else 0)
        rows5.append({"version": v, "scenario": s, "mean_lift": obs, "null_mean": mean(null), "p_perm": pval,
                      "top_influencer": top[0][0] if top else "", "top_lift": top[0][1] if top else float("nan"),
                      "bad_apple": bad or "", "bad_apple_lift": out_infl.get(bad, float("nan")) if bad else float("nan"),
                      "others_mean_lift": mean([x for u, x in out_infl.items() if u != bad]) if bad else float("nan"),
                      **{f"lift_{u}": out_infl[u] for u in units}})

        # ---- metric 6: private-public consistency
        own, oth, nli_own, who = [], [], [], []
        pa, pb = [], []
        for (a, r) in keys_p:
            if (a, r) not in Ev:
                continue
            own.append(float(Ep[(a, r)] @ Ev[(a, r)]))
            o = [float(Ep[(a, r)] @ Ev[(b, r)]) for b in units if b != a and (b, r) in Ev]
            oth.append(mean(o)); who.append(a)
            pa.append(posts[(a, r)]); pb.append(priv[(a, r)][:3000])
        ag = agree(pa, pb) if pa else np.zeros(0)
        gap = [x - y for x, y in zip(own, oth)]
        rec = {"version": v, "scenario": s, "posts": len(own), "cos_own_private": mean(own), "cos_other_private": mean(oth),
               "gap_own_minus_other": mean(gap), "nli_agreement_own": float(ag.mean()) if len(ag) else float("nan"),
               "share_nli_contradiction_own": float((ag < -0.5).mean()) if len(ag) else float("nan")}
        if bad:
            ib = [i for i, u in enumerate(who) if u == bad]; io = [i for i, u in enumerate(who) if u != bad]
            rec.update({"bad_apple": bad, "gap_bad_apple": mean([gap[i] for i in ib]), "gap_others": mean([gap[i] for i in io]),
                        "nli_bad_apple": float(ag[ib].mean()) if ib else float("nan"), "nli_others": float(ag[io].mean()) if io else float("nan")})
        rows6.append(rec)
        print(f"[compute] {v}/{s}: fatto", flush=True)

    df3 = pd.DataFrame(rows3); df3.to_csv(os.path.join(out, "m3_nli_agreement.csv"), index=False)
    df5 = pd.DataFrame(rows5); df5.to_csv(os.path.join(out, "m5_influence.csv"), index=False)
    pd.DataFrame(infl_rows).to_csv(os.path.join(out, "m5_influence_pairs.csv"), index=False)
    df6 = pd.DataFrame(rows6); df6.to_csv(os.path.join(out, "m6_private_public.csv"), index=False)
    md.append("\n## 3. Accordo NLI tra i post di squadre diverse\n")
    md.append("accordo = P(implicazione) - P(contraddizione), media nelle due direzioni, in [-1, 1]; coppie contraddittorie = accordo < -0.5.\n")
    md.append("| versione | scenario | coppie | accordo | stesso round | round successivo | coppie contraddittorie | inizio -> fine |")
    md.append("|---|---|---|---|---|---|---|---|")
    for _, r in df3.iterrows():
        md.append(f"| {r.version} | {r.scenario} | {r.pairs} | {fmt(r.agreement, 2)} | {fmt(r.agreement_same_round, 2)} | {fmt(r.agreement_next_round, 2)} | {r.share_contradictory_pairs:.0%} | {fmt(r.agreement_early, 2)} -> {fmt(r.agreement_late, 2)} |")
    md.append("\n## 5. Influenza semantica (post di A -> memoria privata di B nei 3 round successivi, meno i 3 precedenti)\n")
    md.append("lift > 0 = dopo il post la memoria delle altre squadre gli si avvicina; p = test di permutazione (round del post mescolati, 200 volte).\n")
    md.append("| versione | scenario | lift medio | nullo | p | squadra piu' influente (lift) | bad apple: lift vs altre |")
    md.append("|---|---|---|---|---|---|---|")
    for _, r in df5.iterrows():
        ba = f"{r.bad_apple}: {fmt(r.bad_apple_lift, 4)} vs {fmt(r.others_mean_lift, 4)}" if r.bad_apple else "-"
        md.append(f"| {r.version} | {r.scenario} | {fmt(r.mean_lift, 4)} | {fmt(r.null_mean, 4)} | {fmt(r.p_perm, 3)} | {r.top_influencer} ({fmt(r.top_lift, 4)}) | {ba} |")
    md.append("\n## 6. Coerenza privato-pubblico\n")
    md.append("gap = somiglianza (embedding) del post con il PROPRIO privato meno quella con il privato delle altre squadre nello stesso round "
              "(> 0 = il post rispecchia il proprio privato); NLI = accordo post vs proprio privato.\n")
    md.append("| versione | scenario | post | gap | NLI accordo | quota NLI contraddittoria | bad apple: gap / NLI vs altre |")
    md.append("|---|---|---|---|---|---|---|")
    for _, r in df6.iterrows():
        ba = (f"{r.bad_apple}: {fmt(r.gap_bad_apple, 3)} / {fmt(r.nli_bad_apple, 2)} vs {fmt(r.gap_others, 3)} / {fmt(r.nli_others, 2)}"
              if isinstance(r.get("bad_apple", None), str) and r.get("bad_apple") else "-")
        md.append(f"| {r.version} | {r.scenario} | {r.posts} | {fmt(r.gap_own_minus_other, 3)} | {fmt(r.nli_agreement_own, 2)} | {fmt(r.share_nli_contradiction_own, 2)} | {ba} |")
    with open(os.path.join(out, "summary.md"), "x") as f:
        f.write("\n".join(md) + "\n")
    print(f"[compute] scritto {out}/summary.md", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["judge", "compute"])
    ap.add_argument("outdir")
    ap.add_argument("--versions", default=None, help="comma-separated run folders under the current dir (default: v2,v3)")
    ap.add_argument("--only", default=None)
    ap.add_argument("--max-rounds", type=int, default=None)
    ap.add_argument("--model", default="qwen3:30b-a3b-instruct-2507-q4_K_M")
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    if a.versions:
        VERSIONS[:] = a.versions.split(",")
    {"judge": cmd_judge, "compute": cmd_compute}[a.cmd](a)
