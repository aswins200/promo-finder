#!/usr/bin/env python3
"""
promofinder.py - find promo clips (20-60s, hook + cliffhanger) in microdrama screenplays.

Two interchangeable "judges" feed the SAME validation + ranking code:
  --judge heuristic : free, offline, rule-based (baseline; used for the raw outputs in this repo)
  --judge llm       : one LLM call per episode + one global ranking call (model swappable by env vars)

Usage:
  python promofinder.py --input "path/to/show_folder" --out raw_outputs --judge heuristic
  LLM_PROVIDER=anthropic LLM_MODEL=<model> LLM_API_KEY=... python promofinder.py --input ... --judge llm
  LLM_PROVIDER=openai LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai LLM_MODEL=... (Gemini/Groq/OpenRouter etc.)
"""
import argparse, csv, glob, json, os, re, sys, urllib.request

MIN_LEN, MAX_LEN = 20.0, 60.0
TS = r"\((\d+):(\d+\.\d+)\)"

def sec(m, s): return int(m) * 60 + float(s)
def fmt(t): return f"{int(t // 60):02d}:{t % 60:05.2f}"

# ---------------------------------------------------------------- parsing
def parse_episode(path):
    txt = open(path, encoding="utf-8").read()
    ep = int(re.search(r"episode-(\d+)", os.path.basename(path)).group(1))
    logline = (re.search(r"EPISODE LOGLINE:\s*(.+)", txt) or [None, ""])[1].strip()
    cont = (re.search(r"CONTINUITY NOTES:\s*(.+)", txt) or [None, ""])[1].strip()
    body = txt.split("--- SCRIPT START ---")[-1].split("--- SCRIPT END ---")[0]
    events = []
    # blocks look like: **SPEAKER (tone):**\nStart: (mm:ss.ss) | End: (mm:ss.ss)\ntext
    pat = re.compile(r"\*\*(.+?):\*\*\s*\nStart: " + TS + r" \| End: " + TS + r"\s*\n(.*?)(?=\n\*\*|\nSCENE|\Z)", re.S)
    for m in pat.finditer(body):
        head, text = m.group(1).strip(), m.group(6).strip()
        s, e = sec(m.group(2), m.group(3)), sec(m.group(4), m.group(5))
        name = re.sub(r"\s*\(.*\)$", "", head).strip()
        if name == "CAMERA": kind = "camera"
        elif name == "ACTION": kind = "action"
        elif name == "SOUND FX": kind = "sound"
        else: kind = "dialogue"
        events.append(dict(kind=kind, speaker=name if kind == "dialogue" else "", start=s, end=e,
                           text=text.strip('"').strip()))
    events.sort(key=lambda x: (x["start"], x["end"]))
    dur = max([x["end"] for x in events] or [0])
    return dict(ep=ep, logline=logline, continuity=cont, events=events, duration=dur)

def load_show(folder):
    files = sorted(glob.glob(os.path.join(folder, "episode-*.txt")))
    if not files: sys.exit(f"No episode-*.txt files in {folder}")
    return [parse_episode(f) for f in files]

def transcript(ep, with_camera=False):
    """Compact text for an LLM: dialogue + actions + sound, with timestamps."""
    lines = []
    for e in ep["events"]:
        if e["kind"] == "camera" and not with_camera: continue
        tag = e["speaker"] + ":" if e["kind"] == "dialogue" else f"[{e['kind'].upper()}]"
        lines.append(f"[{fmt(e['start'])}-{fmt(e['end'])}] {tag} {e['text']}")
    return "\n".join(lines)

# ---------------------------------------------------------------- heuristic judge
HOOK_WORDS = r"slap|betray|divorce|dead|die|kill|murder|poison|arrest|guilty|liar|lie|secret|truth|pregnan|ring|marry|marriage|duke|prince|king|queen|heir|traitor|forgive|never|how dare|get out|leave|nobody|who are you|you.?re (not|fired|mine)|bastard|disgrace|shame|beg|kneel|revenge|crown|throne|blood|scar|reveal"
CLIFF_WORDS = r"\?|\.\.\.|—$|-$|wait|stop|what|who|you.?ve|how could|impossible|guards|arrest|open|reveal|truth|duke|dead|alive|poison|ring|watch"
WEAK_OPEN = r"^(and|but|so|then|well|yes|no|okay|ok|he|she|it|they|that|this|him|her)\b"
SOUND_CLIFF = r"sting|ominous|cliffhanger|silence|cut to black|stinger|hold"
RESOLVED = r"triumphant|hand-in-hand|happily|embrace|laugh|smiles warmly|kiss|walk(s)? (away|out)|sunlight|fades? out|relie"

def intensity(text):
    t = text.lower()
    return len(re.findall(HOOK_WORDS, t)) + 0.7 * t.count("!") + 0.4 * t.count("?")

def score_window(ep, s, e):
    ev = [x for x in ep["events"] if x["start"] >= s - 0.01 and x["end"] <= e + 0.01]
    dia = [x for x in ev if x["kind"] == "dialogue"]
    if len(dia) < 3: return None
    f = {}
    # HOOK: what happens in the first 3 seconds, and is it self-contained?
    first = [x for x in ev if x["kind"] in ("dialogue", "action") and x["start"] < s + 3]
    ftxt = " ".join(x["text"] for x in first)
    hook = min(intensity(ftxt) / 3.0, 1.0)
    if not first or first[0]["start"] - s > 1.5: hook *= 0.5          # dead air at the start
    if first and re.match(WEAK_OPEN, first[0]["text"].lower()): hook *= 0.6  # needs prior context
    if first and first[0]["kind"] == "dialogue" and len(first[0]["text"].split()) <= 2: hook *= 0.7
    if first and first[0]["text"].startswith(("...", "\u2026")): hook *= 0.5                         # starts mid-sentence
    if first and first[0]["kind"] == "dialogue" and len(first[0]["text"].split()) > 22: hook *= 0.6  # wall of text, no punch
    f["hook"] = hook
    # CLIFFHANGER: does it stop on an open question / threat / reveal, or at the episode's own cut?
    last = [x for x in ev if x["end"] >= e - 3]
    ltxt = " ".join(x["text"] for x in last)
    cliff = 0.0
    if e >= ep["duration"] - 6: cliff += 0.5                          # episodes are written to end on a hook
    if re.search(CLIFF_WORDS, ltxt.lower()): cliff += 0.3
    if any(x["kind"] == "sound" and re.search(SOUND_CLIFF, x["text"].lower()) for x in last): cliff += 0.2
    if last and last[-1]["kind"] == "dialogue" and last[-1]["text"].rstrip().endswith(("?", "...", "—")): cliff += 0.2
    if last and last[-1]["kind"] == "dialogue" and len(last[-1]["text"].split()) <= 1: cliff *= 0.5   # "No." / "Impossible."
    if re.search(RESOLVED, " ".join(x["text"] for x in ev[-4:]).lower()): cliff *= 0.4              # scene resolves, not hangs
    f["cliff"] = min(cliff, 1.0)
    # INTENSITY: emotional density across the window
    f["intensity"] = min(sum(intensity(x["text"]) for x in ev) / ((e - s) / 8.0) / 3.0, 1.0)
    # STANDALONE: enough speakers + not too many different named people to introduce
    spk = {x["speaker"] for x in dia}
    f["standalone"] = (0.5 if 2 <= len(spk) <= 4 else 0.2) + (0.5 if s < 5 or len(dia) >= 5 else 0.2)
    L = e - s
    f["length"] = 1.0 if 30 <= L <= 50 else 0.7
    total = (0.30 * f["hook"] + 0.35 * f["cliff"] + 0.20 * f["intensity"] +
             0.10 * f["standalone"] + 0.05 * f["length"])
    return total, f, ev

def heuristic_candidates(ep, per_episode=3):
    starts = sorted({round(x["start"], 2) for x in ep["events"] if x["kind"] in ("dialogue", "action")})
    ends = sorted({round(x["end"], 2) for x in ep["events"] if x["kind"] in ("dialogue", "action")})
    ends.append(round(ep["duration"], 2))
    cands = []
    for s in starts:
        for e in ends:
            if MIN_LEN <= e - s <= MAX_LEN:
                r = score_window(ep, s, e)
                if r: cands.append((r[0], s, e, r[1], r[2]))
    cands.sort(reverse=True)
    picked = []
    for sc, s, e, f, ev in cands:                                    # non-overlapping best-first
        if all(e <= p["segments"][0][0] or s >= p["segments"][0][1] for p in picked):
            dia = [x for x in ev if x["kind"] == "dialogue"]
            hook_ev = [x for x in ev if x["kind"] in ("dialogue", "action") and x["start"] < s + 3]
            picked.append(dict(
                ep=ep["ep"], segments=[(s, e)], score=round(sc * 10, 2), features={k: round(v, 2) for k, v in f.items()},
                hook=(hook_ev[0]["text"] if hook_ev else dia[0]["text"])[:220],
                cliffhanger=dia[-1]["text"][:220],
                why=f"hook {f['hook']:.1f} / cliffhanger {f['cliff']:.1f} / intensity {f['intensity']:.1f} (rule-based; no semantic check)",
                judge="heuristic"))
        if len(picked) >= per_episode: break
    return picked

# ---------------------------------------------------------------- LLM judge (swappable)
def call_llm(system, user):
    prov = os.environ.get("LLM_PROVIDER", "anthropic"); model = os.environ["LLM_MODEL"]; key = os.environ["LLM_API_KEY"]
    if prov == "anthropic":
        url = os.environ.get("LLM_BASE_URL", "https://api.anthropic.com") + "/v1/messages"
        body = dict(model=model, max_tokens=3000, system=system, messages=[{"role": "user", "content": user}])
        hdr = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        pick = lambda r: r["content"][0]["text"]
    else:  # any OpenAI-compatible endpoint (OpenAI, Gemini, Groq, OpenRouter, local)
        url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1") + "/chat/completions"
        body = dict(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        hdr = {"Authorization": f"Bearer {key}", "content-type": "application/json"}
        pick = lambda r: r["choices"][0]["message"]["content"]
    req = urllib.request.Request(url, json.dumps(body).encode(), hdr)
    return pick(json.load(urllib.request.urlopen(req, timeout=180)))

def extract_json(s):
    m = re.search(r"\[.*\]|\{.*\}", s, re.S)
    return json.loads(m.group(0)) if m else []

EP_SYSTEM = f"""You are a short-drama promo editor for TikTok/Reels/YouTube Shorts (ReelShort/DramaBox style).
From ONE episode screenplay, pick up to 3 promo clips. Each clip is {MIN_LEN:.0f}-{MAX_LEN:.0f}s TOTAL, made of 1-3 segments
(segments must be in order, each starting/ending on a timestamp that exists in the script).
A good promo: (1) HOOK in the first 3 seconds (slap, reveal, shocking line) that works for someone who has never seen the show;
(2) ends on an UNRESOLVED moment (cliffhanger) - cut before the answer; (3) understandable alone.
Never start on a line that depends on unseen context ("Why?", "He did it"). Never end after the tension has resolved.
Return ONLY JSON: [{{"segments":[["mm:ss.ss","mm:ss.ss"],...],"hook":"...","cliffhanger":"...","why":"one line","score":1-10}}]"""

def to_sec(x):
    m = re.match(r"(\d+):(\d+(?:\.\d+)?)", str(x)); return sec(m.group(1), m.group(2)) if m else float(x)

def snap_and_validate(ep, raw):
    """Never trust model timestamps: snap to real event boundaries, enforce length, drop invalid."""
    starts = [x["start"] for x in ep["events"]]; ends = [x["end"] for x in ep["events"]] + [ep["duration"]]
    near = lambda t, arr: min(arr, key=lambda a: abs(a - t))
    out = []
    for c in raw:
        try:
            segs = [(near(to_sec(a), starts), near(to_sec(b), ends)) for a, b in c["segments"]][:3]
            if any(b <= a for a, b in segs) or any(segs[i][1] > segs[i + 1][0] for i in range(len(segs) - 1)): continue
            total = sum(b - a for a, b in segs)
            if not (MIN_LEN <= total <= MAX_LEN): continue
            out.append(dict(ep=ep["ep"], segments=segs, score=float(c.get("score", 0)), features={},
                            hook=c.get("hook", ""), cliffhanger=c.get("cliffhanger", ""), why=c.get("why", ""),
                            judge="llm:" + os.environ.get("LLM_MODEL", "?")))
        except Exception:
            continue
    return out

def llm_candidates(ep, prev=None):
    ctx = f"Logline: {ep['logline']}\nContinuity: {ep['continuity']}\n\nSCRIPT:\n{transcript(ep)}"
    return snap_and_validate(ep, extract_json(call_llm(EP_SYSTEM, ctx)))

RANK_SYSTEM = """You are the head of growth for a microdrama app. Given candidate promos (id, hook, cliffhanger, score) from one show,
rank the best 10 to POST FIRST for cold audiences. Prefer strongest hook, strongest unresolved ending, variety (don't post 5 near-identical
betrayal slaps), and clarity for a viewer who has never seen the show. Return ONLY JSON: [{"id":N,"reason":"one line"}]"""

def llm_rank(promos):
    brief = "\n".join(f"{i}: ep{p['ep']} score={p['score']} HOOK={p['hook'][:120]} | CLIFF={p['cliffhanger'][:120]}" for i, p in enumerate(promos))
    return extract_json(call_llm(RANK_SYSTEM, brief))

# ---------------------------------------------------------------- output
def write_outputs(promos, out, top_reasons=None):
    os.makedirs(out, exist_ok=True)
    for i, p in enumerate(promos, 1): p["rank"] = i
    json.dump(promos, open(f"{out}/promos.json", "w"), indent=1)
    with open(f"{out}/promos.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["rank", "episode", "segments", "duration_s", "score", "hook", "cliffhanger", "why", "judge"])
        for p in promos:
            w.writerow([p["rank"], p["ep"], " + ".join(f"{fmt(a)}-{fmt(b)}" for a, b in p["segments"]),
                        round(sum(b - a for a, b in p["segments"]), 1), p["score"], p["hook"], p["cliffhanger"], p["why"], p["judge"]])
    with open(f"{out}/ranked_promos.md", "w") as fh:
        fh.write("# Ranked promos\n\n")
        for p in promos:
            d = sum(b - a for a, b in p["segments"])
            fh.write(f"**#{p['rank']} - Episode {p['ep']}** | {' + '.join(f'{fmt(a)}-{fmt(b)}' for a, b in p['segments'])} ({d:.0f}s) | score {p['score']}\n"
                     f"- Hook: {p['hook']}\n- Cliffhanger: {p['cliffhanger']}\n- Why: {p['why']}\n\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--out", default="raw_outputs")
    ap.add_argument("--judge", choices=["heuristic", "llm"], default="heuristic")
    ap.add_argument("--per-episode", type=int, default=3)
    a = ap.parse_args()
    show = load_show(a.input)
    print(f"Loaded {len(show)} episodes, {sum(len(e['events']) for e in show)} events")
    allp = []
    for ep in show:
        try:
            c = heuristic_candidates(ep, a.per_episode) if a.judge == "heuristic" else llm_candidates(ep)
        except Exception as ex:
            print(f"ep {ep['ep']}: FAILED ({ex}) - skipped"); continue
        allp += c
    allp.sort(key=lambda p: -p["score"])
    if a.judge == "llm" and allp:
        try:
            order = llm_rank(allp[:60]); ids = [o["id"] for o in order if isinstance(o.get("id"), int) and o["id"] < len(allp)]
            for o in order:
                if isinstance(o.get("id"), int) and o["id"] < len(allp): allp[o["id"]]["why"] += " | rank: " + o.get("reason", "")
            allp = [allp[i] for i in ids] + [p for i, p in enumerate(allp) if i not in ids]
        except Exception as ex:
            print("ranking call failed, falling back to score order:", ex)
    write_outputs(allp, a.out)
    print(f"Wrote {len(allp)} promos to {a.out}/ (ranked_promos.md, promos.csv, promos.json)")

if __name__ == "__main__":
    main()
