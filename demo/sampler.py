"""sampler.py : random examples for the demo, drawn from the TEST split of the seed (the system never trained on it).

Four kinds, each drawn first by dataset (so a small dataset is not drowned by a big one) and then by item:
  direct_attack  : deepset / Gandalf attacks (+ held-out attack families)          channel direct,   label attack
  direct_benign  : deepset benign, NotInject (benign text with trigger words), our own trigger-word messages
  doc_poisoned   : TaskTracker poisoned paragraphs, BIPIA contexts with an attack inserted, mock-KB attack chunks
  doc_clean      : TaskTracker clean paragraphs, BIPIA contexts, mock-KB documents, security articles quoting attacks
For documents, `task` is a matching user request (when the dataset has one), so the page can send it first.
"""
from __future__ import annotations

import random
import re

_SENT = re.compile(r"(?<=[.!?])\s+")
KINDS = ["direct_attack", "direct_benign", "doc_poisoned", "doc_clean"]


def _txt(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("text", "prompt", "context", "question"):
            if x.get(k):
                return str(x[k])
    return ""


def _src(x, default):
    return str(x.get("source", default)) if isinstance(x, dict) else default


def build_pools(sp) -> dict:
    """kind -> {dataset name: (items, builder(item, rng) -> dict)}"""
    def g(d, k="test"):
        try:
            return list(d.get(k, []))
        except Exception:                                    # noqa: BLE001
            return []

    P = {k: {} for k in KINDS}
    att = g(sp.att)
    for name in sorted({_src(a, "deepset") for a in att}):
        P["direct_attack"][name] = ([a for a in att if _src(a, "deepset") == name],
                                    lambda it, r, n=name: dict(text=_txt(it), channel="direct", label="attack", dataset=n))
    ben = g(sp.ben)
    if ben:
        P["direct_benign"]["deepset (benign)"] = (ben, lambda it, r: dict(text=_txt(it), channel="direct", label="benign", dataset="deepset (benign)"))
    ni = g(sp.notinject)
    if ni:
        P["direct_benign"]["NotInject"] = (ni, lambda it, r: dict(text=_txt(it), channel="direct", label="benign", dataset="NotInject (benign, trigger words)"))
    trg = g(sp.triggers)
    if trg:
        P["direct_benign"]["own trigger-word messages"] = (trg, lambda it, r: dict(text=_txt(it), channel="direct", label="benign", dataset="own benign messages with trigger words"))

    tt = g(sp.tt)
    for lab, kind, name in ((1, "doc_poisoned", "TaskTracker (poisoned)"), (0, "doc_clean", "TaskTracker (clean)")):
        items = [r for r in tt if int(r.get("label", 0)) == lab]
        if items:
            P[kind][name] = (items, lambda it, r, l=lab: dict(text=_txt(it), channel="indirect", label="attack" if l else "benign",
                                                             dataset="TaskTracker", task=it.get("user_prompt") or None))
    try:
        bt = sp.bipia["test"]
        ctxs = list(bt["contexts"])
        atks = [a for v in bt["attacks"].values() for a in v]
    except Exception:                                        # noqa: BLE001
        ctxs, atks = [], []
    if ctxs:
        P["doc_clean"]["BIPIA"] = (ctxs, lambda it, r: dict(text=_txt(it), channel="indirect", label="benign", dataset="BIPIA",
                                                            task=it.get("question") if isinstance(it, dict) else None))
        if atks:
            def poisoned(it, r):
                sents = _SENT.split(_txt(it))
                k = r.randint(0, len(sents))
                body = " ".join(sents[:k] + [r.choice(atks)] + sents[k:])
                return dict(text=body, channel="indirect", label="attack", dataset="BIPIA (attack inserted)",
                            task=it.get("question") if isinstance(it, dict) else None)
            P["doc_poisoned"]["BIPIA"] = (ctxs, poisoned)
    kb = g(sp.kb)
    if kb:
        P["doc_poisoned"]["mock KB"] = (kb, lambda it, r: dict(text=r.choice(it["attacks"]), channel="indirect", label="attack",
                                                               dataset="mock knowledge base (attack chunk)", task=r.choice(it["instructions"])))
        P["doc_clean"]["mock KB"] = (kb, lambda it, r: dict(text=r.choice(it["docs"] + it.get("bridge", [])), channel="indirect",
                                                            label="benign", dataset="mock knowledge base", task=r.choice(it["instructions"])))
    art = g(sp.articles)
    if art:
        P["doc_clean"]["security articles"] = (art, lambda it, r: dict(text=_txt(it), channel="indirect", label="benign",
                                                                       dataset="security article quoting attacks (benign)"))
    return {k: v for k, v in P.items() if v}


def sample(pools: dict, kind: str = "any", rng: random.Random | None = None) -> dict:
    r = rng or random
    kinds = [k for k in KINDS if k in pools]
    if not kinds:
        raise RuntimeError("no data available to sample from")
    if kind == "any" or kind not in pools:
        kind = r.choice(kinds)
    name = r.choice(sorted(pools[kind]))
    items, build = pools[kind][name]
    out = build(r.choice(items), r)
    out["kind"] = kind
    out["split"] = "test (held out)"
    return out


# ------------------------------------------------------------------ whole random sessions (same builders as the experiments)
SCENARIO_TITLES = {
    "benign": "Benign session", "direct": "Direct attack", "doc": "Injection inside a document",
    "split": "Injection split across chunks", "drift": "Topic drift (boiling frog)", "topic-change": "Legitimate topic change"}


def make_scenario(sp, kind: str, rng=None) -> dict:
    """A random session of the requested kind, built with the experiment's own session builders from the TEST split
    (the system never trained on it).  Returns {title, steps:[{channel, text, label, source, role}]}."""
    from pidetect import sessions as S
    r = rng or random.Random()
    sp_t = "test"
    kb = list(sp.kb[sp_t])
    bip = sp.bipia[sp_t]
    bip_atk = [a for v in bip["attacks"].values() for a in v]
    d_att = [d["text"] for d in sp.att[sp_t]]
    d_ben = [d["text"] for d in sp.ben[sp_t]] + list(sp.triggers[sp_t])
    tt = list(sp.tt.get(sp_t, []))
    pos = [x for x in tt if int(x["label"]) == 1]
    neg = [x for x in tt if int(x["label"]) == 0]
    sid = "demo"
    opts = []
    if kind == "benign":
        if kb: opts.append(lambda: S.kb_session(r.choice(kb), r, sid, n_turns=r.randint(1, 3)))
        if bip["contexts"]: opts.append(lambda: S.bipia_session(r.choice(bip["contexts"]), None, r, sid))
        if neg: opts.append(lambda: S.tasktracker_session(r.choice(neg), sid))
        if d_ben: opts.append(lambda: S.benign_direct_session(r.sample(d_ben, min(len(d_ben), r.randint(2, 4))), sid))
        if sp.sec_instr[sp_t] and sp.articles[sp_t]:
            opts.append(lambda: S.security_article_session(r.choice(sp.sec_instr[sp_t]), r.sample(sp.articles[sp_t], min(2, len(sp.articles[sp_t]))), sid))
    elif kind == "direct":
        if d_att: opts.append(lambda: S.direct_multiturn_session(r.sample(d_ben, min(2, len(d_ben))), r.choice(d_att), r, sid, kind="single"))
    elif kind == "doc":
        if bip["contexts"] and bip_atk: opts.append(lambda: S.bipia_session(r.choice(bip["contexts"]), r.choice(bip_atk), r, sid))
        if pos: opts.append(lambda: S.tasktracker_session(r.choice(pos), sid))
        if kb: opts.append(lambda: S.kb_session(r.choice(kb), r, sid, n_turns=r.randint(1, 3), attack=r.choice(r.choice(kb)["attacks"]), attack_kind="single_chunk"))
    elif kind == "split":
        if kb: opts.append(lambda: S.kb_session(r.choice(kb), r, sid, n_turns=r.randint(1, 2), attack=r.choice(r.choice(kb)["attacks"]), attack_kind="split_chunks"))
    elif kind == "drift":
        if kb:
            def drift():
                c = r.choice(kb)
                return S.boiling_frog_session(c, r.choice([3, 5]), r, sid, r.choice(c["attacks"]))
            opts.append(drift)
    elif kind == "topic-change":
        if len(kb) >= 2:
            def tc():
                a, b = r.sample(kb, 2)
                return S.topic_change_session(a, b, r, sid)
            opts.append(tc)
    if not opts:
        raise RuntimeError(f"no data to build a '{kind}' scenario")
    sess = r.choice(opts)()
    steps = [dict(channel=e.channel, text=e.text, label="attack" if int(e.label) == 1 else "benign",
                  source=e.source or "", role=str(e.meta.get("role", ""))) for e in sess.events]
    return dict(title=SCENARIO_TITLES.get(kind, kind), steps=steps)
