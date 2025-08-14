# ============================================================
# Barthes Lexia Explication — Two Inputs + Side-by-Side Compare
# Input A: your lexia (e.g., lexia_render_lines.txt or inline)
# Input B: Barthes' original lexia (lexia_barthes_lines.txt)
# Output: per-set JSONL/CSV/HTML + a comparative CSV/HTML
# ============================================================

# ---------- CONFIG ----------
PRIMARY_INPUT_CANDIDATES = [
    "/content/lexia_render_lines.txt",
    "/content/lexia_render_inline.txt",
    "/mnt/data/lexia_render_lines.txt",
    "/mnt/data/lexia_render_inline.txt",
    "/content/lexia_input.txt",
    "/mnt/data/lexia_input.txt",
]

SECONDARY_INPUT_CANDIDATES = [
    "/content/lexia_barthes_lines.txt",
    "/mnt/data/lexia_barthes_lines.txt",
]

OUT_DIR = "/content"
LABEL_A = "your"
LABEL_B = "barthes"

CLAUDE_MODEL = "claude-3-5-sonnet-20240620"
MAX_TOKENS   = 1800
TEMPERATURE  = 0.2
NEIGHBOR_WINDOW = 1
DRY_RUN = False  # set True to test without API calls

# ---------- INSTALLS ----------
import sys, subprocess, pkgutil, os, re, json, unicodedata, html, textwrap, csv
def pip_install(pkg): subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
for pkg in ["anthropic", "spacy", "rapidfuzz", "sentence-transformers", "numpy", "pandas"]:
    if pkgutil.find_loader(pkg) is None:
        pip_install(pkg)

import numpy as np, pandas as pd
import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util as sbert_util
sbert = SentenceTransformer("all-mpnet-base-v2")

# ---------- API KEY ----------
try:
    from google.colab import userdata  # Colab only
    if not os.getenv("ANTHROPIC_API_KEY"):
        val = userdata.get("ANTHROPIC_API_KEY")
        if val: os.environ["ANTHROPIC_API_KEY"] = val
except Exception:
    pass

if not os.environ.get("ANTHROPIC_API_KEY"):
    try:
        import getpass
        os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter Anthropic API key (hidden): ")
    except Exception:
        pass

from typing import List, Dict, Tuple
from dataclasses import dataclass

# ---------- UTILS ----------
def norm(s: str) -> str:
    s = unicodedata.normalize('NFKC', s)
    s = s.replace('“','"').replace('”','"').replace('’',"'").replace('—','-').replace('–','-')
    s = re.sub(r'\s+',' ', s).strip()
    return s

def find_existing(paths: List[str]) -> str:
    for p in paths:
        if os.path.exists(p):
            return p
    return ""  # don't raise; we'll handle gracefully

LEXIA_LINE_RE   = re.compile(r'^\s*\[(\d+)\]\s*(.+?)\s*$', re.S)
LEXIA_INLINE_RE = re.compile(r'\[(\d+)\]\s*')

def parse_lexia_lines(text: str) -> List[Tuple[int,str]]:
    out=[]
    for line in text.splitlines():
        line=line.strip()
        if not line: continue
        m = LEXIA_LINE_RE.match(line)
        if m:
            out.append((int(m.group(1)), norm(m.group(2))))
    if out:
        return out
    # Inline fallback
    parts = LEXIA_INLINE_RE.split(text)
    if len(parts) >= 3:
        it = iter(parts)
        _ = next(it, "")
        while True:
            try:
                nid = int(next(it))
            except StopIteration:
                break
            chunk = next(it, "")
            out.append((nid, norm(chunk)))
    return sorted(out, key=lambda x: x[0])

# ---------- HINTS ----------
DEICTIC_PIVOTS = ["then,", "thus,", "on my right", "on my left", "here", "there", "on the borderline"]
OPPOSITION_CANDIDATES = [
    ("inside","outside"), ("life","death"), ("cold","heat"), ("dark","light"),
    ("right","left"), ("silence","noise"), ("nature","culture")
]
SENSORY_LEXICONS = {
    "visual":   ["light","dark","glitter","sparkle","moon","ghost","shadow","color","shine","chandelier","gold","silver"],
    "auditory": ["murmur","music","voice","voices","outbursts","clink","sound","sounds","dice","whisper","rustle"],
    "olfactory":["perfume","scent","odor","fragrance","smell"],
    "tactile":  ["cold","heat","warm","chill","freeze","humid","fevered","silk","draft","brocade","gauze"]
}
CATALOGUE_MIN_COMMAS = 4

def detect_hints(lexia_text: str) -> Dict:
    low = lexia_text.lower()
    deictics = [p for p in DEICTIC_PIVOTS if p in low]
    opps=[]
    for a,b in OPPOSITION_CANDIDATES:
        if a in low and b in low:
            opps.append((a,b))
    commas = lexia_text.count(',')
    doc = nlp(lexia_text)
    adj = sum(1 for t in doc if t.pos_=="ADJ")
    ger = sum(1 for t in doc if t.tag_=="VBG")
    tokens = sum(1 for t in doc if not t.is_space)
    catalogue_ratio = (adj+ger)/max(1,tokens)
    is_catalogue = (commas >= CATALOGUE_MIN_COMMAS and catalogue_ratio >= 0.12)
    ents = [{"text":e.text, "label":e.label_} for e in doc.ents]
    senses = {k:[] for k in SENSORY_LEXICONS}
    for k,words in SENSORY_LEXICONS.items():
        for w in words:
            if re.search(rf'\b{re.escape(w)}\b', low):
                senses[k].append(w)
    return {
        "deictics": deictics,
        "oppositions": opps,
        "is_catalogue": is_catalogue,
        "commas": commas,
        "adj+ger_ratio": round(catalogue_ratio,3),
        "entities": ents,
        "sensory": senses,
    }

# ---------- PROMPT ----------
SYSTEM_PROMPT = (
    "You are assisting in a Roland Barthes–style explication of a numbered lexia from Balzac’s Sarrasine.\n"
    "Follow Barthes’s method in S/Z: treat the text as a weave of five codes—HER (Hermeneutic), ACT (Proairetic), "
    "SEM (Semic), SYM (Symbolic), REF (Referential)—and explain how meaning is produced.\n"
    "A lexia is the best possible space to observe meanings and should carry at most three or four enumerated meanings. "
    "Do not summarize the whole story; stay with the given lexia and its adjacent context only. "
    "Do not invent plot details beyond the passage. Quote short substrings from the lexia to justify claims.\n"
    "\n"
    "HER: name any enigma/withholding/suspense staged or deferred.\n"
    "ACT: name actions or action cues (gestures, shifts, temporal steps).\n"
    "SEM: list motifs/traits (e.g., objects, settings, adjectives) that construct character or scene texture.\n"
    "SYM: expose oppositions (inside/outside, life/death, right/left, cold/heat, dark/light) and explain their function.\n"
    "REF: cite domains of knowledge/culture evoked (e.g., Parisian salon, fashion, gambling) without outside spoilers.\n"
    "Return STRICT JSON; do not include commentary outside JSON."
)

def make_user_payload(lex_id:int, lex_text:str, prev_text:str, next_text:str, hints:Dict) -> str:
    payload = {
        "lexia_id": lex_id,
        "lexia_text": lex_text,
        "context": {"prev_lexia": prev_text, "next_lexia": next_text},
        "preanalysis_hints": hints,
        "task": (
            "Unpack this lexia with the Five Codes and brief wider thinking. "
            "Constrain enumerated meanings to ≤4. Quote words from the lexia for evidence. "
            "Avoid spoilers and external plot knowledge."
        ),
        "expected_schema": {
            "lexia_id": "int",
            "codes": {
                "HER": {"present": "bool", "notes": "string", "evidence": ["strings"]},
                "ACT": {"present": "bool", "notes": "string", "evidence": ["strings"]},
                "SEM": {"present": "bool", "motifs": ["strings"], "evidence": ["strings"]},
                "SYM": {"present": "bool", "oppositions": ["A vs B"], "notes": "string", "evidence": ["strings"]},
                "REF": {"present": "bool", "domains": ["strings"], "notes": "string", "evidence": ["strings"]}
            },
            "enumerated_meanings": ["≤4 concise items"],
            "sensory_fields": {"visual": ["strings"], "auditory": ["strings"], "olfactory": ["strings"], "tactile": ["strings"]},
            "wider_commentary": "2–3 sentences connecting this lexia to the local weave (no outside plot).",
            "code_strengths": {"HER": "0–3", "ACT": "0–3", "SEM": "0–3", "SYM": "0–3", "REF": "0–3"}
        }
    }
    return json.dumps(payload, ensure_ascii=False)

# ---------- ANTHROPIC ----------
from anthropic import Anthropic
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def call_claude(json_payload:str) -> Dict:
    if DRY_RUN:
        temp = json.loads(json_payload)
        return {
            "lexia_id": temp["lexia_id"],
            "codes": {
                "HER": {"present": False, "notes":"", "evidence":[]},
                "ACT": {"present": bool(temp["preanalysis_hints"]["deictics"]), "notes":"deictic pivot", "evidence": temp["preanalysis_hints"]["deictics"][:1]},
                "SEM": {"present": True, "motifs": ["window","salon","garden"][:2], "evidence": ["window","salon"]},
                "SYM": {"present": bool(temp["preanalysis_hints"]["oppositions"]), "oppositions":[f"{a} vs {b}" for a,b in temp["preanalysis_hints"]["oppositions"]], "notes":"", "evidence":[]},
                "REF": {"present": temp["preanalysis_hints"]["is_catalogue"], "domains": ["Parisian salon"] if temp["preanalysis_hints"]["is_catalogue"] else [], "notes":"", "evidence":[]}
            },
            "enumerated_meanings": ["stub A","stub B"],
            "sensory_fields": temp["preanalysis_hints"]["sensory"],
            "wider_commentary": "Stub commentary (DRY_RUN).",
            "code_strengths": {"HER":0,"ACT":1,"SEM":2,"SYM":1,"REF":2}
        }
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=SYSTEM_PROMPT,
        messages=[{"role":"user","content": json_payload}]
    )
    txt = msg.content[0].text.strip()
    try:
        return json.loads(txt)
    except Exception:
        m = re.search(r'\{.*\}', txt, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise ValueError("Claude did not return valid JSON.")

# ---------- DRIVER ----------
@dataclass
class LexiaRecord:
    lexia_id: int
    lexia_text: str
    prev_text: str
    next_text: str
    hints: Dict
    result: Dict

def window(items: List[Tuple[int,str]], i:int, k:int=1) -> Tuple[str,str]:
    prev_text = items[i-1][1] if i-1 >= 0 else ""
    next_text = items[i+1][1] if i+1 < len(items) else ""
    if k > 1:
        if i-k >= 0:
            prev_text = " ".join(x[1] for x in items[max(0,i-k):i])
        if i+k < len(items):
            next_text = " ".join(x[1] for x in items[i+1:min(len(items), i+1+k)])
    return prev_text, next_text

def analyze_list(items: List[Tuple[int,str]]) -> List[LexiaRecord]:
    records=[]
    for i,(lex_id, text) in enumerate(items):
        prev_text, next_text = window(items, i, NEIGHBOR_WINDOW)
        hints = detect_hints(text)
        payload = make_user_payload(lex_id, text, prev_text, next_text, hints)
        try:
            result = call_claude(payload)
        except Exception as e:
            result = {
                "lexia_id": lex_id,
                "error": str(e),
                "codes": {"HER":{"present":False},"ACT":{"present":False},"SEM":{"present":True,"motifs":[]},"SYM":{"present":False},"REF":{"present":False}},
                "enumerated_meanings": [],
                "sensory_fields": hints.get("sensory", {}),
                "wider_commentary": "",
                "code_strengths": {"HER":0,"ACT":0,"SEM":1,"SYM":0,"REF":0}
            }
        records.append(LexiaRecord(lex_id, text, prev_text, next_text, hints, result))
        print(f"Analyzed [{lex_id}] ({len(text)} chars).")
    return records

def load_input(paths: List[str]) -> List[Tuple[int,str]]:
    p = find_existing(paths)
    if not p:
        return []
    with open(p, encoding="utf-8") as f:
        raw = f.read()
    items = parse_lexia_lines(raw)
    if not items:
        raise ValueError(f"Could not parse any lexia from: {p}")
    print(f"Loaded {len(items)} lexia from {p}")
    return items

# ---------- SAVE HELPERS ----------
def save_jsonl(records: List[LexiaRecord], path:str):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "lexia_id": r.lexia_id,
                "lexia_text": r.lexia_text,
                "hints": r.hints,
                "result": r.result
            }, ensure_ascii=False) + "\n")
    print(f"Saved: {path}")

def flatten_row(r: LexiaRecord) -> Dict:
    res = r.result or {}
    codes = res.get("codes", {})
    def safe(x, d=None): 
        return codes.get(x, d or {})
    row = {
        "lexia_id": r.lexia_id,
        "lexia_text": r.lexia_text,
        "HER_present": safe("HER",{}).get("present"),
        "HER_notes":   safe("HER",{}).get("notes",""),
        "HER_evidence":"|".join(safe("HER",{}).get("evidence",[]) or []),
        "ACT_present": safe("ACT",{}).get("present"),
        "ACT_notes":   safe("ACT",{}).get("notes",""),
        "ACT_evidence":"|".join(safe("ACT",{}).get("evidence",[]) or []),
        "SEM_present": safe("SEM",{}).get("present"),
        "SEM_motifs":  "|".join(safe("SEM",{}).get("motifs",[]) or []),
        "SEM_evidence":"|".join(safe("SEM",{}).get("evidence",[]) or []),
        "SYM_present": safe("SYM",{}).get("present"),
        "SYM_pairs":   "|".join(safe("SYM",{}).get("oppositions",[]) or []),
        "SYM_notes":   safe("SYM",{}).get("notes",""),
        "SYM_evidence":"|".join(safe("SYM",{}).get("evidence",[]) or []),
        "REF_present": safe("REF",{}).get("present"),
        "REF_domains": "|".join(safe("REF",{}).get("domains",[]) or []),
        "REF_notes":   safe("REF",{}).get("notes",""),
        "REF_evidence":"|".join(safe("REF",{}).get("evidence",[]) or []),
        "enumerated_meanings": "|".join(res.get("enumerated_meanings",[]) or []),
        "visual":   "|".join((res.get("sensory_fields",{}) or {}).get("visual",[]) or []),
        "auditory": "|".join((res.get("sensory_fields",{}) or {}).get("auditory",[]) or []),
        "olfactory":"|".join((res.get("sensory_fields",{}) or {}).get("olfactory",[]) or []),
        "tactile":  "|".join((res.get("sensory_fields",{}) or {}).get("tactile",[]) or []),
        "wider_commentary": res.get("wider_commentary",""),
        "HER_strength": (res.get("code_strengths",{}) or {}).get("HER",0),
        "ACT_strength": (res.get("code_strengths",{}) or {}).get("ACT",0),
        "SEM_strength": (res.get("code_strengths",{}) or {}).get("SEM",0),
        "SYM_strength": (res.get("code_strengths",{}) or {}).get("SYM",0),
        "REF_strength": (res.get("code_strengths",{}) or {}).get("REF",0),
    }
    return row

def save_csv(records: List[LexiaRecord], path:str):
    rows = [flatten_row(r) for r in records]
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Saved: {path}")

def esc(s: str) -> str:
    return html.escape(s or "")

def save_html(records: List[LexiaRecord], path:str, title:str):
    blocks=[]
    for r in records:
        res = r.result or {}
        codes = res.get("codes", {})
        def show_code(label, data):
            if not data: return ""
            present = data.get("present")
            badge = "✅" if present else "—"
            ev = ", ".join(data.get("evidence",[]) or [])
            extra = ""
            if label=="SEM":
                extra = f"<div><b>Motifs:</b> {esc(', '.join(data.get('motifs',[]) or []))}</div>"
            if label=="SYM":
                extra = f"<div><b>Oppositions:</b> {esc(', '.join(data.get('oppositions',[]) or []))}</div>"
            if label=="REF":
                extra = f"<div><b>Domains:</b> {esc(', '.join(data.get('domains',[]) or []))}</div>"
            notes = data.get("notes","")
            return f"<div><b>{label}</b> {badge} &nbsp; <i>{esc(notes)}</i><br/><small>{esc(ev)}</small>{extra}</div>"
        senses = res.get("sensory_fields",{}) or {}
        em = res.get("enumerated_meanings",[]) or []
        comm = res.get("wider_commentary","")
        strengths = res.get("code_strengths",{}) or {}
        blocks.append(f"""
        <section style="padding:12px 16px; margin:12px 0; border:1px solid #ddd; border-radius:12px;">
          <div style="font-weight:700; font-size:1.05rem; margin-bottom:6px;">Lexia [{r.lexia_id}]</div>
          <div style="margin:8px 0; line-height:1.4">{esc(r.lexia_text)}</div>
          <div style="display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:8px; margin:8px 0;">
            {show_code('HER', codes.get('HER'))}
            {show_code('ACT', codes.get('ACT'))}
            {show_code('SEM', codes.get('SEM'))}
            {show_code('SYM', codes.get('SYM'))}
            {show_code('REF', codes.get('REF'))}
          </div>
          <div><b>Enumerated meanings (≤4):</b> {esc('; '.join(em))}</div>
          <div style="margin-top:6px;"><b>Sensory fields:</b> visual={esc(', '.join(senses.get('visual',[]) or []))}; auditory={esc(', '.join(senses.get('auditory',[]) or []))}; olfactory={esc(', '.join(senses.get('olfactory',[]) or []))}; tactile={esc(', '.join(senses.get('tactile',[]) or []))}</div>
          <div style="margin-top:6px;"><b>Wider commentary:</b> {esc(comm)}</div>
          <div style="margin-top:6px; font-size:0.9rem; color:#444;">
            <b>Code strengths:</b> HER={esc(str(strengths.get('HER',0)))}, ACT={esc(str(strengths.get('ACT',0)))}, SEM={esc(str(strengths.get('SEM',0)))}, SYM={esc(str(strengths.get('SYM',0)))}, REF={esc(str(strengths.get('REF',0)))}
          </div>
        </section>
        """)
    html_doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>{esc(title)}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    </head><body style="max-width:880px; margin:24px auto; font-family:system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">
    <h1 style="margin:0 0 8px;">{esc(title)}</h1>
    <div style="color:#444; margin-bottom:16px;">Automated explication of each lexia with Barthes’s five codes and brief commentary.</div>
    {"".join(blocks)}
    </body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"Saved: {path}")

# ---------- COMPARISON ----------
def records_to_df(records: List[LexiaRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        row = flatten_row(r)
        rows.append(row)
    return pd.DataFrame(rows)

def truncate(s, n=120):
    s = s or ""
    return (s[:n-1] + "…") if len(s) > n else s

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # strengths → float
    strength_cols = [c for c in df.columns if c.endswith("_strength")]
    for c in strength_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # present flags → bool
    present_cols = [c for c in df.columns if c.endswith("_present")]
    def _to_bool(v):
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        s = str(v).strip().lower()
        return s in ("true", "1", "yes", "y")
    for c in present_cols:
        df[c] = df[c].apply(_to_bool)

    # ids → int (tidy)
    if "lexia_id" in df.columns:
        df["lexia_id"] = pd.to_numeric(df["lexia_id"], errors="coerce").fillna(0).astype(int)
    return df


def align_sets(dfA: pd.DataFrame, dfB: pd.DataFrame) -> pd.DataFrame:
    # Ensure numeric/bool types so the math below is safe
    dfA = coerce_types(dfA)
    dfB = coerce_types(dfB)

    # SBERT semantic alignment A -> best B
    A_texts = dfA["lexia_text"].tolist()
    B_texts = dfB["lexia_text"].tolist()
    if not A_texts or not B_texts:
        return pd.DataFrame()
    EA = sbert.encode(A_texts, normalize_embeddings=True)
    EB = sbert.encode(B_texts, normalize_embeddings=True)
    sims = sbert_util.cos_sim(EA, EB).cpu().numpy()  # shape [lenA, lenB]
    rows = []
    for i, arow in dfA.reset_index(drop=True).iterrows():
        j = int(np.argmax(sims[i]))
        sim = float(sims[i, j])
        brow = dfB.reset_index(drop=True).iloc[j]
        rows.append({
            "A_id": int(arow["lexia_id"]),
            "A_text": truncate(arow["lexia_text"]),
            "B_id": int(brow["lexia_id"]),
            "B_text": truncate(brow["lexia_text"]),
            "similarity": round(sim, 3),
            "HER_A": arow["HER_present"], "HER_B": brow["HER_present"],
            "dHER": (int(bool(arow["HER_present"])) - int(bool(brow["HER_present"]))),
            "ACT_A": arow["ACT_present"], "ACT_B": brow["ACT_present"],
            "dACT": (int(bool(arow["ACT_present"])) - int(bool(brow["ACT_present"]))),
            "SEM_A": arow["SEM_present"], "SEM_B": brow["SEM_present"],
            "dSEM": (int(bool(arow["SEM_present"])) - int(bool(brow["SEM_present"]))),
            "SYM_A": arow["SYM_present"], "SYM_B": brow["SYM_present"],
            "dSYM": (int(bool(arow["SYM_present"])) - int(bool(brow["SYM_present"]))),
            "REF_A": arow["REF_present"], "REF_B": brow["REF_present"],
            "dREF": (int(bool(arow["REF_present"])) - int(bool(brow["REF_present"]))),
            "HERs_A": arow["HER_strength"], "HERs_B": brow["HER_strength"],
            "dHERs": (arow["HER_strength"] - brow["HER_strength"]),
            "ACTs_A": arow["ACT_strength"], "ACTs_B": brow["ACT_strength"],
            "dACTs": (arow["ACT_strength"] - brow["ACT_strength"]),
            "SEMs_A": arow["SEM_strength"], "SEMs_B": brow["SEM_strength"],
            "dSEMs": (arow["SEM_strength"] - brow["SEM_strength"]),
            "SYMs_A": arow["SYM_strength"], "SYMs_B": brow["SYM_strength"],
            "dSYMs": (arow["SYM_strength"] - brow["SYM_strength"]),
            "REFs_A": arow["REF_strength"], "REFs_B": brow["REF_strength"],
            "dREFs": (arow["REF_strength"] - brow["REF_strength"]),
        })
    return pd.DataFrame(rows)


def save_compare_html(df_align: pd.DataFrame, path: str, title: str, statsA: Dict, statsB: Dict):
    if df_align.empty:
        print("No alignment rows to render; skipping compare HTML.")
        return

    rows = []
    for _, r in df_align.sort_values("A_id").iterrows():
        rows.append(f"""
        <tr>
          <td>[{int(r.A_id)}]</td>
          <td>{esc(r.A_text)}</td>
          <td>[{int(r.B_id)}]</td>
          <td>{esc(r.B_text)}</td>
          <td style="text-align:right">{r.similarity:.3f}</td>
          <td style="text-align:center">{'✅' if r.HER_A else '—'}</td>
          <td style="text-align:center">{'✅' if r.HER_B else '—'}</td>
          <td style="text-align:center">{'✅' if r.ACT_A else '—'}</td>
          <td style="text-align:center">{'✅' if r.ACT_B else '—'}</td>
          <td style="text-align:center">{'✅' if r.SEM_A else '—'}</td>
          <td style="text-align:center">{'✅' if r.SEM_B else '—'}</td>
          <td style="text-align:center">{'✅' if r.SYM_A else '—'}</td>
          <td style="text-align:center">{'✅' if r.SYM_B else '—'}</td>
          <td style="text-align:center">{'✅' if r.REF_A else '—'}</td>
          <td style="text-align:center">{'✅' if r.REF_B else '—'}</td>
        </tr>
        """)
    html_doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>{esc(title)}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
      table{{border-collapse:collapse;width:100%;}} th,td{{border:1px solid #ddd;padding:6px 8px;}} th{{background:#fafafa;}}
      td:nth-child(5){{font-variant-numeric:tabular-nums;}}
    </style></head>
    <body style="max-width:1024px;margin:24px auto;font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
      <h1 style="margin:0 0 8px;">{esc(title)}</h1>
      <div style="margin-bottom:10px;color:#444">Alignment Your↔︎Barthes with code presence deltas and similarity scores.</div>
      <div style="display:flex;gap:12px;margin:12px 0;">
        <div style="flex:1;border:1px solid #ddd;border-radius:10px;padding:8px;">
          <b>Your set</b><br/>
          Lexia: {statsA['count']}<br/>
          HER/ACT/SEM/SYM/REF present %: {statsA['HER']:.0f}/{statsA['ACT']:.0f}/{statsA['SEM']:.0f}/{statsA['SYM']:.0f}/{statsA['REF']:.0f}<br/>
          Avg strengths (0–3): {statsA['HERs']:.2f}/{statsA['ACTs']:.2f}/{statsA['SEMs']:.2f}/{statsA['SYMs']:.2f}/{statsA['REFs']:.2f}
        </div>
        <div style="flex:1;border:1px solid #ddd;border-radius:10px;padding:8px;">
          <b>Barthes set</b><br/>
          Lexia: {statsB['count']}<br/>
          HER/ACT/SEM/SYM/REF present %: {statsB['HER']:.0f}/{statsB['ACT']:.0f}/{statsB['SEM']:.0f}/{statsB['SYM']:.0f}/{statsB['REF']:.0f}<br/>
          Avg strengths (0–3): {statsB['HERs']:.2f}/{statsB['ACTs']:.2f}/{statsB['SEMs']:.2f}/{statsB['SYMs']:.2f}/{statsB['REFs']:.2f}
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th colspan="2">Your lexia</th>
            <th colspan="2">Barthes lexia</th>
            <th>sim</th>
            <th>HER A</th><th>HER B</th>
            <th>ACT A</th><th>ACT B</th>
            <th>SEM A</th><th>SEM B</th>
            <th>SYM A</th><th>SYM B</th>
            <th>REF A</th><th>REF B</th>
          </tr>
        </thead>
        <tbody>
          {"".join(rows)}
        </tbody>
      </table>
    </body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"Saved: {path}")


def set_stats(df: pd.DataFrame) -> Dict:
    n = max(1, len(df))
    def pct(col):
        return 100.0 * (pd.Series(df[col]).fillna(False).astype(bool).sum()) / n
    def avg(col):
        return float(pd.Series(df[col]).fillna(0).astype(float).mean())
    return {
        "count": len(df),
        "HER": pct("HER_present"), "ACT": pct("ACT_present"), "SEM": pct("SEM_present"),
        "SYM": pct("SYM_present"), "REF": pct("REF_present"),
        "HERs": avg("HER_strength"), "ACTs": avg("ACT_strength"), "SEMs": avg("SEM_strength"),
        "SYMs": avg("SYM_strength"), "REFs": avg("REF_strength"),
    }


# ---------- RUN ----------
def main():
    # Load both inputs
    itemsA = load_input(PRIMARY_INPUT_CANDIDATES)
    itemsB = load_input(SECONDARY_INPUT_CANDIDATES)
    if not itemsA:
        raise FileNotFoundError("No primary input found. Put your lexia file in one of PRIMARY_INPUT_CANDIDATES.")
    if not itemsB:
        print("WARNING: No secondary (Barthes) input found; will analyze primary only.")

    # Analyze A
    print(f"\nAnalyzing primary ({LABEL_A})… DRY_RUN={DRY_RUN}")
    recsA = analyze_list(itemsA)
    jsonA = os.path.join(OUT_DIR, f"lexia_analysis_{LABEL_A}.jsonl")
    csvA  = os.path.join(OUT_DIR, f"lexia_analysis_{LABEL_A}.csv")
    htmlA = os.path.join(OUT_DIR, f"lexia_analysis_{LABEL_A}.html")
    save_jsonl(recsA, jsonA); save_csv(recsA, csvA); save_html(recsA, htmlA, f"Barthes Codes — {LABEL_A}")

    # Analyze B (if present)
    if itemsB:
        print(f"\nAnalyzing secondary ({LABEL_B})… DRY_RUN={DRY_RUN}")
        recsB = analyze_list(itemsB)
        jsonB = os.path.join(OUT_DIR, f"lexia_analysis_{LABEL_B}.jsonl")
        csvB  = os.path.join(OUT_DIR, f"lexia_analysis_{LABEL_B}.csv")
        htmlB = os.path.join(OUT_DIR, f"lexia_analysis_{LABEL_B}.html")
        save_jsonl(recsB, jsonB); save_csv(recsB, csvB); save_html(recsB, htmlB, f"Barthes Codes — {LABEL_B}")

        # Comparison / alignment
        dfA = records_to_df(recsA)
        dfB = records_to_df(recsB)
        dfA = coerce_types(dfA)
        dfB = coerce_types(dfB)

        align = align_sets(dfA, dfB)
        cmp_csv  = os.path.join(OUT_DIR, f"lexia_compare_{LABEL_A}_vs_{LABEL_B}.csv")
        align.to_csv(cmp_csv, index=False, encoding="utf-8")
        statsA = set_stats(dfA); statsB = set_stats(dfB)
        cmp_html = os.path.join(OUT_DIR, f"lexia_compare_{LABEL_A}_vs_{LABEL_B}.html")
        save_compare_html(align, cmp_html, f"Your vs Barthes — Alignment & Codes", statsA, statsB)

    print("\nDone. Files in /content:")
    print(" -", jsonA)
    print(" -", csvA)
    print(" -", htmlA)
    if itemsB:
        print(" -", jsonB)
        print(" -", csvB)
        print(" -", htmlB)
        print(" -", cmp_csv)
        print(" -", cmp_html)

main()

