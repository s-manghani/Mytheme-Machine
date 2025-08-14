# ============================================
# Barthes Lexia — Five Codes + Focused LLM Ops (Barthes-mode enhanced)
# Single Colab cell (with Colab secrets + final rendering)
# ============================================

# ----- CONFIG: edit paths if needed -----
BALZAC_CANDIDATES = [
    "/content/balzac.txt",
    "/mnt/data/balzac.txt",
]
GOLD_CANDIDATES = [
    "/content/barthes_lexia.csv",
    "/mnt/data/barthes_lexia.csv",
]

# Core run knobs
RUN_OPS_ROUNDS       = 2            # focused LLM refinement rounds
TOP_K_FEEDBACK       = 3            # worst mismatches per round for focus
LOCAL_WINDOW_RADIUS  = 2            # neighborhood size around mismatches
BOUNDARY_TOL         = 5            # boundary tolerance (chars)
USE_COORD_COMMAS     = True         # may be overridden by BARTHES_MODE
MAX_LEXIA_CHARS      = 320          # may be overridden by BARTHES_MODE
TARGET_COUNT_ALPHA   = 0.15         # acceptance ring around gold (used loosely now)
CLAUDE_MODEL         = "claude-3-5-sonnet-20240620"  # update if you have newer Sonnet
BARTHES_MODE         = True         # <--- turn on the “macro-lexia” behavior

# Linguistic anchors / locks
LOCK_PHRASES = [
    "i was deep in one of those daydreams",
    "seated in a window recess",
    "then, turning",
    "thus, on my right",
    "on my left",
    "hidden behind the sinuous folds of a silk curtain",
]

ANCHORS = {
    "deictic": ["Then,", "Thus,", "On the borderline", "On my right", "on my left", "Here", "There"],
    "inside_outside": ["window recess", "window", "garden", "room", "salon"],
    "gestures_voice": ["outbursts of the gamblers", "clink of gold", "movements of the head", "murmur of conversation"],
    "oppositions": ["half pleasant, half funereal", "life", "death", "cold", "heat", "dark", "light", "right", "left"]
}

# ----- installs (safe to re-run) -----
import sys, subprocess, pkgutil
def pip_install(pkg): subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
for pkg in ["spacy", "sentence-transformers", "anthropic", "rapidfuzz"]:
    if pkgutil.find_loader(pkg) is None:
        pip_install(pkg)

import spacy
try:
    spacy.load("en_core_web_sm")
except Exception:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)

# ----- imports -----
import os, re, json, unicodedata, numpy as np, pandas as pd
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz
from anthropic import Anthropic

# ----- API key: Colab secrets -> env -> prompt -----
try:
    from google.colab import userdata  # works only on Colab
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

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ----- utils -----
def find_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("None of these paths exist:\n" + "\n".join(paths))

def norm(s: str) -> str:
    s = unicodedata.normalize('NFKC', s)
    s = s.replace('“','"').replace('”','"').replace('’',"'").replace('—','-').replace('–','-')
    s = re.sub(r'\s+',' ', s).strip()
    return s

BALZAC = find_existing(BALZAC_CANDIDATES)
GOLD    = find_existing(GOLD_CANDIDATES)

with open(BALZAC, encoding='utf-8') as f:
    src_raw = f.read()
src = norm(src_raw)

def load_gold(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8-sig')
    df.columns = [c.strip().lower() for c in df.columns]
    # choose a likely text column
    priority = ['gold_text','text','lexia_text','lexia','segment','unit']
    text_col = next((c for c in priority if c in df.columns), None)
    if text_col is None:
        obj = [c for c in df.columns if df[c].dtype=='object']
        if not obj: raise ValueError("No text-like column in gold CSV.")
        text_col = max(obj, key=lambda c: df[c].astype(str).str.len().mean())
    id_candidates = ['gold_id','id','lexia_id','index','#','no']
    id_col = next((c for c in id_candidates if c in df.columns), None)
    if id_col is None:
        df = df.reset_index().rename(columns={'index':'gold_id'})
    else:
        df = df.rename(columns={id_col:'gold_id'})
    df = df.rename(columns={text_col:'gold_text'})
    df = df[['gold_id','gold_text']].copy()
    df['gold_text'] = df['gold_text'].astype(str).map(norm)
    if not pd.api.types.is_integer_dtype(df['gold_id']):
        df['gold_id'] = pd.factorize(df['gold_id'])[0] + 1
    def find_span(snippet):
        i = src.find(snippet)
        return (i, i+len(snippet)) if i!=-1 else (-1, -1)
    spans = [find_span(t) for t in df['gold_text']]
    df['gold_start'] = [a for a,b in spans]
    df['gold_end']   = [b for a,b in spans]
    return df.sort_values('gold_id').reset_index(drop=True)

gold = load_gold(GOLD)
nlp  = spacy.load('en_core_web_sm')
sbert = SentenceTransformer('all-mpnet-base-v2')

# ----- segmentation: clauses + strong punct + optional coord-commas -----
CLAUSE_DEPS = {'ccomp','xcomp','advcl','relcl','conj','parataxis'}

def split_clausewise(src_text: str) -> pd.DataFrame:
    doc = nlp(src_text)
    spans=[]
    for sent in doc.sents:
        start = sent.start
        for tok in sent:
            if tok.dep_ in CLAUSE_DEPS and tok.i > start:
                chunk = doc[start:tok.i].text.strip()
                if chunk: spans.append(chunk); start = tok.i
        tail = doc[start:sent.end].text.strip()
        if tail: spans.append(tail)
    chunks=[]; off=0
    for sp in spans:
        i = src_text.find(sp, off)
        if i!=-1:
            chunks.append((i, i+len(sp), sp)); off = i+len(sp)
    df = pd.DataFrame(chunks, columns=['start_char','end_char','text'])
    df.insert(0,'cand_id', range(1, len(df)+1))
    df['method']='clauses'; df['notes']=''
    return df

# NOTE: we intentionally drop '-' from strong punctuation to avoid hyphen splits (e.g., Elysée-Bourbon)
def split_on_strong_punct(df: pd.DataFrame, src_text: str, punct=r'[;:]') -> pd.DataFrame:
    out=[]
    for _,row in df.iterrows():
        s0, t = int(row.start_char), row.text
        parts = re.split(f'({punct})', t)
        merged=[]; it=iter(parts)
        for p in it:
            if re.fullmatch(punct, p or ''):
                if merged: merged[-1] = (merged[-1] + p).strip()
            else:
                if p.strip(): merged.append(p.strip())
        off=s0
        for m in merged:
            i = src_text.find(m, off)
            if i!=-1:
                out.append((i, i+len(m), m)); off=i+len(m)
    df2 = pd.DataFrame(out, columns=['start_char','end_char','text'])
    df2.insert(0,'cand_id', range(1, len(df2)+1))
    df2['method']='clause+punct'; df2['notes']=''
    return df2

# --- optional coord-comma split with catalogue detection ---
def is_catalogue_like(t: str) -> bool:
    doc = nlp(t)
    commas = t.count(',')
    adj = sum(1 for tok in doc if tok.pos_ == 'ADJ')
    ger = sum(1 for tok in doc if tok.tag_ == 'VBG')
    tokens = sum(1 for tok in doc if not tok.is_space)
    ratio = (adj + ger) / max(1, tokens)
    return commas >= 4 and ratio >= 0.12  # tweakable

def split_on_coord_commas(df: pd.DataFrame, src_text: str) -> pd.DataFrame:
    out=[]
    for _,row in df.iterrows():
        s0, t = int(row.start_char), row.text
        if is_catalogue_like(t):  # hold tableau-like catalogues intact
            out.append((s0, s0+len(t), t))
            continue
        parts = re.split(r'(, (?:and|but|or) )', t, flags=re.I)
        merged=[]; it=iter(parts)
        for p in it:
            if re.fullmatch(r', (?:and|but|or) ', p or '', flags=re.I):
                if merged: merged[-1] = (merged[-1] + p.strip()).strip()
            else:
                if p.strip(): merged.append(p.strip())
        off=s0
        for m in merged:
            i = src_text.find(m, off)
            if i!=-1:
                out.append((i, i+len(m), m)); off=i+len(m)
    df2 = pd.DataFrame(out, columns=['start_char','end_char','text'])
    df2.insert(0,'cand_id', range(1, len(df2)+1))
    df2['method']='clause+punct+comma'; df2['notes']=''
    return df2

# --- NER-aware boundary protection: do not split across named entities
def merge_crossing_entities(df: pd.DataFrame, src_text: str) -> pd.DataFrame:
    doc = nlp(src_text)
    ents = [(e.start_char, e.end_char) for e in doc.ents]
    rows = df.sort_values('start_char').to_dict('records')
    out=[]; i=0
    def inside_entity(pos):
        return any(s0 < pos < e0 for (s0,e0) in ents)
    while i < len(rows):
        s, e, t = rows[i]['start_char'], rows[i]['end_char'], rows[i]['text']
        j=i
        while j+1 < len(rows):
            boundary = rows[j]['end_char']
            if boundary == rows[j+1]['start_char'] and inside_entity(boundary):
                e = rows[j+1]['end_char']
                t = (t + " " + rows[j+1]['text']).strip()
                j += 1
            else:
                break
        out.append((s, e, t))
        i = j+1
    df2 = pd.DataFrame(out, columns=['start_char','end_char','text'])
    df2.insert(0,'cand_id', range(1, len(df2)+1))
    df2['method']='ner_merge'; df2['notes']='merged across entity boundaries'
    return df2

# ----- alignment & metrics -----
def span_overlap(a0,a1,b0,b1)->float:
    inter = max(0, min(a1,b1)-max(a0,b0))
    union = max(a1,b1)-min(a0,b0)
    return inter/union if union>0 else 0.0

def align_scores(gold_df: pd.DataFrame, cand_df: pd.DataFrame):
    E_gold = sbert.encode(gold_df['gold_text'].tolist(), normalize_embeddings=True)
    E_cand = sbert.encode(cand_df['text'].tolist(), normalize_embeddings=True)
    rows=[]
    for i,g in gold_df.iterrows():
        best=(-1,-1,-1,-1)
        for j,c in cand_df.iterrows():
            ov = span_overlap(g['gold_start'], g['gold_end'], c['start_char'], c['end_char']) if g['gold_start']>=0 else 0.0
            sim = float(util.cos_sim(E_gold[i], E_cand[j]))
            score = 0.6*ov + 0.4*sim
            if score>best[0]: best=(score, ov, sim, j)
        match=('exact' if best[1] >= 0.90 else
               'partial' if best[1] >= 0.30 else
               'semantic' if best[2] >= 0.85 else
               'miss')
        rows.append({'gold_id': int(g['gold_id']),
                     'cand_id': int(cand_df.iloc[best[3]]['cand_id']),
                     'overlap_score': best[1],
                     'sim_score': best[2],
                     'match_type': match})
    return pd.DataFrame(rows)

def boundaries_from_spans(df: pd.DataFrame, start_col='start_char'):
    return sorted(int(x) for x in df[start_col].tolist()[1:])

def boundary_f1(gold_df: pd.DataFrame, cand_df: pd.DataFrame, tol:int=5):
    G = boundaries_from_spans(gold_df.rename(columns={'gold_start':'start_char'}), 'start_char')
    C = boundaries_from_spans(cand_df, 'start_char')
    matched=set(); tp=0
    for c in C:
        gm=None
        for g in G:
            if abs(c-g)<=tol and g not in matched:
                gm=g; break
        if gm is not None:
            matched.add(gm); tp+=1
    precision = tp/max(1,len(C)); recall = tp/max(1,len(G))
    f1 = 2*precision*recall / max(precision+recall, 1e-9)
    return precision, recall, f1

def report(name: str, cand_df: pd.DataFrame, save_prefix: str):
    align = align_scores(gold, cand_df)
    counts = align['match_type'].value_counts(dropna=False).to_dict()
    avg_ov = float(align['overlap_score'].mean())
    avg_si = float(align['sim_score'].mean())
    p,r,f1 = boundary_f1(gold, cand_df, tol=BOUNDARY_TOL)
    print(f"\n=== {name} ===")
    print(f"Counts: {counts}")
    print(f"Avg overlap: {avg_ov:.3f} | Avg sim: {avg_si:.3f} | Boundary P/R/F1: {p:.2f}/{r:.2f}/{f1:.2f} | Lexia count: {len(cand_df)}")
    align.to_csv(f"/content/{save_prefix}_alignment.csv", index=False)
    cand_df.to_csv(f"/content/{save_prefix}_candidate_lexia.csv", index=False)
    return align, (p,r,f1)

def worst_mismatches(cand_df: pd.DataFrame, k:int=5):
    E_gold = sbert.encode(gold['gold_text'].tolist(), normalize_embeddings=True)
    E_cand = sbert.encode(cand_df['text'].tolist(), normalize_embeddings=True)
    triples=[]
    for i,g in gold.iterrows():
        sims = util.cos_sim(E_gold[i], E_cand)[0].cpu().numpy()
        j = int(np.argmax(sims))
        ov = span_overlap(g['gold_start'], g['gold_end'], cand_df.iloc[j]['start_char'], cand_df.iloc[j]['end_char'])
        triples.append((ov, float(sims[j]), i, j))
    triples.sort(key=lambda x: (x[0], x[1]))
    out=[]
    print(f"\n--- {k} worst matches ---")
    for ov, sim, gi, cj in triples[:k]:
        gtxt = gold.iloc[gi]['gold_text']; ctxt = cand_df.iloc[cj]['text']
        cid  = int(cand_df.iloc[cj]['cand_id'])
        print(f"\nGOLD {int(gold.iloc[gi]['gold_id'])}  ov={ov:.2f}  sim={sim:.2f}  (cand_id={cid})")
        print('--- GOLD ---'); print(gtxt)
        print('--- CAND ---'); print(ctxt)
        out.append((int(gold.iloc[gi]['gold_id']), gtxt, cid, ctxt))
    return out

## --- helper to (re)index and add method/notes safely ---
def _finalize(df, method:str, notes:str):
    # keep only core cols and (re)create cand_id
    core = df[['start_char','end_char','text']].copy()
    if 'cand_id' in core.columns:
        core = core.drop(columns=['cand_id'])
    core.insert(0, 'cand_id', range(1, len(core)+1))
    core['method'] = method
    core['notes']  = notes
    return core

# --- UPDATED MACRO FUNCTIONS ---

PIVOT_BUNDLE   = re.compile(r'^(Then|Thus),\s*$', re.I)
PARTICIPLE_HEAD= re.compile(r'^(turning|seated|hidden|on)\b', re.I)
TABLEAU_START  = re.compile(r'(Dance of the Living!|a splendid salon)', re.I)
TABLEAU_END    = re.compile(r'fevered mind\.\s*$', re.I)

def bundle_pivot_with_gesture(df):
    df = df[['start_char','end_char','text']].copy()  # sanitize
    rows = df.sort_values('start_char').to_dict('records')
    merged=[]; i=0
    while i < len(rows):
        t = rows[i]['text'].strip()
        if PIVOT_BUNDLE.match(t) and i+1 < len(rows) and PARTICIPLE_HEAD.match(rows[i+1]['text'].strip()):
            merged.append({
                'start_char': rows[i]['start_char'],
                'end_char':   rows[i+1]['end_char'],
                'text':       (rows[i]['text'] + " " + rows[i+1]['text']).strip()
            })
            i += 2
        else:
            merged.append({
                'start_char': rows[i]['start_char'],
                'end_char':   rows[i]['end_char'],
                'text':       rows[i]['text']
            })
            i += 1
    df2 = pd.DataFrame(merged, columns=['start_char','end_char','text'])
    return _finalize(df2, 'pivot_bundle', 'Then/Thus bundled with following gesture')

def merge_antithesis_block(df, src_text):
    df = df[['start_char','end_char','text']].copy()  # sanitize
    rows = df.sort_values('start_char').to_dict('records')
    i = next((k for k,r in enumerate(rows) if re.match(r'^\s*Thus,', r['text'], re.I)), None)
    if i is None:
        return _finalize(df, 'antithesis_merge', 'no-op (no Thus, found)')
    j = i
    seen_right = seen_left = False
    while j < len(rows):
        txt = rows[j]['text'].lower()
        seen_right |= 'on my right' in txt
        seen_left  |= 'on my left'  in txt
        if txt.rstrip().endswith('.') and seen_right and seen_left:
            break
        j += 1
    if i < len(rows) and j < len(rows) and j >= i:
        merged_txt = " ".join(r['text'] for r in rows[i:j+1]).strip()
        block = rows[:i] + [{
            'start_char': rows[i]['start_char'],
            'end_char':   rows[j]['end_char'],
            'text':       merged_txt
        }] + rows[j+1:]
        df2 = pd.DataFrame(block, columns=['start_char','end_char','text'])
        return _finalize(df2, 'antithesis_merge', 'Thus/right/left/here/there tableau')
    else:
        df2 = pd.DataFrame(rows, columns=['start_char','end_char','text'])
        return _finalize(df2, 'antithesis_merge', 'no-op (incomplete block)')

def merge_salon_tableau(df, src_text):
    df = df[['start_char','end_char','text']].copy()  # sanitize
    m0 = TABLEAU_START.search(src_text); m1 = TABLEAU_END.search(src_text)
    if not (m0 and m1 and m1.start() > m0.start()):
        return _finalize(df, 'salon_tableau', 'no-op (window not found)')
    start_abs, end_abs = m0.start(), m1.end()
    rows = df.sort_values('start_char').to_dict('records')
    keep=[]; i=0; merged_once=False
    while i < len(rows):
        s,e,t = rows[i]['start_char'], rows[i]['end_char'], rows[i]['text']
        if not merged_once and s >= start_abs and e <= end_abs:
            j=i
            while j+1 < len(rows) and rows[j+1]['end_char'] <= end_abs:
                j += 1
            merged_txt = " ".join(r['text'] for r in rows[i:j+1]).strip()
            keep.append({'start_char': start_abs, 'end_char': end_abs, 'text': merged_txt})
            i = j+1; merged_once=True
        else:
            keep.append({'start_char': s, 'end_char': e, 'text': t}); i += 1
    df2 = pd.DataFrame(keep, columns=['start_char','end_char','text'])
    return _finalize(df2, 'salon_tableau', 'catalogue held as one tableau')

def merge_borderline(df):
    df = df[['start_char','end_char','text']].copy()  # sanitize
    rows = df.sort_values('start_char').to_dict('records')
    i = next((k for k,r in enumerate(rows) if r['text'].lower().startswith('on the borderline')), None)
    if i is None:
        return _finalize(df, 'borderline_tableau', 'no-op (no borderline found)')
    j=i
    while j < len(rows) and not rows[j]['text'].strip().endswith('.'):
        j += 1
    if j >= len(rows):  # paragraph end fallback
        j = len(rows) - 1
    merged_txt = " ".join(r['text'] for r in rows[i:j+1]).strip()
    block = rows[:i] + [{
        'start_char': rows[i]['start_char'],
        'end_char':   rows[j]['end_char'],
        'text':       merged_txt
    }] + rows[j+1:]
    df2 = pd.DataFrame(block, columns=['start_char','end_char','text'])
    return _finalize(df2, 'borderline_tableau', 'moral macédoine held together')

def polish_microfragments(df):
    df = df[['start_char','end_char','text']].copy()  # sanitize
    rows = df.sort_values('start_char').to_dict('records')
    out=[]; i=0
    while i < len(rows):
        t = rows[i]['text'].strip()
        if len(t) < 15 and i+1 < len(rows):
            if re.search(r'[,-]\s*$', t) or t.lower() in {"then,", "thus,"} or t.endswith('-'):
                out.append((
                    rows[i]['start_char'],
                    rows[i+1]['end_char'],
                    (rows[i]['text'] + " " + rows[i+1]['text']).strip()
                ))
                i += 2; continue
        out.append((rows[i]['start_char'], rows[i]['end_char'], rows[i]['text'])); i += 1
    df2 = pd.DataFrame(out, columns=['start_char','end_char','text'])
    return _finalize(df2, 'polish', 'microfragments merged')

# ----- five-codes aware, focused LLM ops -----
SYSTEM_PROMPT = (
"You assist with Roland Barthes–style segmentation into lexia, guided by Barthes’s five codes.\n"
"A lexia is the best span to observe meanings (≤ 3–4 meanings), often short; sometimes a longer tableau when imagery/culture dominates.\n"
"Codes:\n"
"• HER (Hermeneutic): suspense/withholding/questions.\n"
"• ACT (Proairetic): action, motion, temporal steps.\n"
"• SEM (Semic): motifs/traits (e.g., “daydream”, “window recess”).\n"
"• SYM (Symbolic): binary oppositions (inside/outside; life/death; cold/heat; right/left; dark/light).\n"
"• REF (Referential): encyclopedic knowledge (Parisian salon, fashion, gambling).\n"
"Operators (boundary cues): deictic pivots (“Then,” “Thus,” “On my right/left,” “Here/There”), inside/outside frames (window/recess/garden/salon/room), catalogues/ekphrasis (colon/semicolon series), gestures/voices, explicit pairings/oppositions.\n"
"Rules:\n"
"1) You may only MERGE contiguous chunk IDs, or SPLIT a single chunk at an EXACT substring (split_before or split_after).\n"
"2) Never paraphrase or reorder text; preserve tokens and order exactly.\n"
"3) Respect locks: any chunk in lock_ids must remain a standalone lexia; do NOT merge into or across it.\n"
"4) Prefer short lexia encapsulating a single operator; allow a few long tableaux (REF/SYM-heavy catalogues).\n"
"5) Do not merge across a clear code shift, especially SYM oppositions and deictic pivots.\n"
"6) Keep total lexia count near target_count; avoid over-long lexia (≤ max_lexia_chars) unless allow_tableau=true and REF/SYM dominate.\n"
"7) Bundle deictic pivots (‘Then,’ ‘Thus,’ ‘On the borderline’) with the immediately following participial/summarizing phrase if no code shift intervenes.\n"
"8) Treat dense catalogues (≥4 commas, high ADJ/VBG ratio) as a single REF/SYM tableau; do not split catalogues at coordinating commas.\n"
"9) Do not split hyphenated proper names or within named entities.\n"
"Output STRICT JSON: {\"operations\":[...], \"notes\":[{\"id\":<chunk_id>,\"codes\":[\"SYM\",\"REF\"],\"operator\":\"inside/outside pivot\"}]} .\n"
)

def make_user_prompt(chunks_df, focus_ids, lock_ids, tableau_ids, target_count, max_chars, anchors, feedback=None):
    payload = [{"id":int(r['cand_id']), "text": r['text']} for _,r in chunks_df.iterrows()]
    prompt = {
        "task": "Propose merges/splits ONLY for focus_ids to better approximate Barthes' lexia boundaries.",
        "constraints": {
            "focus_ids": sorted(list(set(int(i) for i in focus_ids))),
            "lock_ids":  sorted(list(set(int(i) for i in lock_ids))),
            "target_count": int(target_count),
            "max_lexia_chars": int(max_chars),
            "allow_tableau": True,
            "tableau_hint_ids": sorted(list(tableau_ids)),
            "avoid_code_shift_merges": True
        },
        "anchors": anchors,
        "chunks": payload
    }
    if feedback:
        prompt["feedback_examples"] = feedback
    return json.dumps(prompt, ensure_ascii=False)

def call_claude_ops_focused(chunks_df, focus_ids, lock_ids, tableau_ids, anchors, feedback=None):
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1600,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role":"user","content": make_user_prompt(
            chunks_df=chunks_df,
            focus_ids=focus_ids,
            lock_ids=lock_ids,
            tableau_ids=tableau_ids,
            target_count=len(gold),
            max_chars=MAX_LEXIA_CHARS,
            anchors=anchors,
            feedback=feedback
        )}]
    )
    text = msg.content[0].text.strip()
    try:
        data = json.loads(text)
    except Exception:
        m = re.search(r'\{.*\}', text, flags=re.S)
        data = json.loads(m.group(0)) if m else {"operations":[]}
    ops = data.get("operations", [])
    fset, lset = set(int(x) for x in focus_ids), set(int(x) for x in lock_ids)
    clean=[]
    for op in ops:
        if op.get("op")=="merge" and isinstance(op.get("ids"), list) and len(op["ids"])>=2:
            ids = [int(x) for x in op["ids"]]
            if ids == list(range(min(ids), max(ids)+1)) and all(x in fset for x in ids) and not any(x in lset for x in ids if len(ids)>1):
                clean.append({"op":"merge","ids":ids})
        elif op.get("op")=="split" and isinstance(op.get("id"), (int,str)):
            sid = int(op["id"])
            if sid in fset and sid not in lset and (op.get("split_before") or op.get("split_after")):
                clean.append({"op":"split","id":sid,"split_before":op.get("split_before"),"split_after":op.get("split_after")})
    return clean

# ----- apply ops with hard guards -----
PIVOTS = {'then,','thus,','on the borderline','on my right','on my left','here','there'}

def has_pivot(s: str) -> bool:
    low = s.lower()
    return any(p in low for p in PIVOTS)

def apply_ops_guarded(chunks_df: pd.DataFrame, ops, lock_ids, tableau_ids):
    items=[]
    for _,r in chunks_df.sort_values('start_char').iterrows():
        items.append({"ids":[int(r['cand_id'])],"start":int(r['start_char']),"end":int(r['end_char']),"text":r['text']})
    LOCK_SET=set(int(x) for x in lock_ids)
    TABL_SET=set(int(x) for x in tableau_ids)

    def merge_run(ids):
        nonlocal items
        rmin, rmax = min(ids), max(ids)
        new=[]; i=0
        while i<len(items):
            start_id = items[i]["ids"][0]
            if start_id==rmin:
                block=[items[i]]; j=i+1; ok=True
                while j<len(items) and items[j]["ids"][0] <= rmax:
                    if any(x in LOCK_SET for x in items[j]["ids"]): ok=False; break
                    if has_pivot(items[j]["text"]): ok=False; break
                    block.append(items[j]); j+=1
                if ok:
                    merged_txt = " ".join(x["text"] for x in block).strip()
                    block_ids  = sum((x["ids"] for x in block), [])
                    if len(merged_txt) <= MAX_LEXIA_CHARS or any(x in TABL_SET for x in block_ids):
                        new.append({"ids": block_ids,"start": block[0]["start"],"end": block[-1]["end"],"text": merged_txt})
                        i=j; continue
            new.append(items[i]); i+=1
        items=new

    def fuzzy_index(hay, needle, min_score=85):
        if not needle: return -1
        idx = hay.find(needle)
        if idx!=-1: return idx
        n=len(needle); best=(-1,-1); step=max(1, n//6)
        for s in range(0, max(1,len(hay)-n+1), step):
            win = hay[s:s+n+20]
            score = fuzz.partial_ratio(needle, win)
            if score>best[0]:
                best=(score, s + max(0, win.find(needle[:max(1,n//2)])))
        return best[1] if best[0]>=min_score and best[1]>=0 else -1

    def split_run(cid, split_before=None, split_after=None):
        nonlocal items
        for k, it in enumerate(items):
            if cid in it["ids"]:
                if any(x in LOCK_SET for x in it["ids"]): return
                hay = it["text"]
                idx = fuzzy_index(hay, split_before) if split_before else -1
                if idx==-1 and split_after:
                    idx2 = fuzzy_index(hay, split_after)
                    idx = idx2 + len(split_after) if idx2!=-1 else -1
                if idx<=0 or idx>=len(hay)-1: return
                left, right = hay[:idx].strip(), hay[idx:].strip()
                if not left or not right: return
                abs_left  = src.find(left, it["start"]);  abs_left  = abs_left  if abs_left!=-1 else it["start"]
                abs_right = src.find(right, abs_left+len(left)); abs_right = abs_right if abs_right!=-1 else it["start"]+len(left)
                items = items[:k] + [
                    {"ids": it["ids"], "start": abs_left,  "end": abs_left+len(left),  "text": left},
                    {"ids": it["ids"], "start": abs_right, "end": abs_right+len(right), "text": right},
                ] + items[k+1:]
                return

    for op in ops:
        if op["op"]=="merge":
            if any(x in set(lock_ids) for x in op["ids"]) and len(op["ids"])>1:
                continue
            merge_run(op["ids"])
    for op in ops:
        if op["op"]=="split":
            split_run(int(op["id"]), op.get("split_before"), op.get("split_after"))

    out=[]
    for it in sorted(items, key=lambda x: x["start"]):
        out.append((it["start"], it["end"], it["text"]))
    df = pd.DataFrame(out, columns=['start_char','end_char','text'])
    df.insert(0,'cand_id', range(1, len(df)+1))
    df['method']='llm_ops_focused'; df['notes']=''
    return df

# ----- BASELINE (with optional Barthes-mode macro moves) -----
cand_base = split_clausewise(src)
cand_base = split_on_strong_punct(cand_base, src)
if BARTHES_MODE:
    USE_COORD_COMMAS   = False
    MAX_LEXIA_CHARS    = 800
    TARGET_COUNT_ALPHA = 0.75  # start loose; we will tighten per round
if USE_COORD_COMMAS:
    cand_base = split_on_coord_commas(cand_base, src)

cand_base = merge_crossing_entities(cand_base, src)
if BARTHES_MODE:
    cand_base = bundle_pivot_with_gesture(cand_base)
    cand_base = merge_salon_tableau(cand_base, src)
    cand_base = merge_antithesis_block(cand_base, src)
    cand_base = merge_borderline(cand_base)
    cand_base = polish_microfragments(cand_base)

cand_base.to_csv('/content/candidate_lexia_baseline.csv', index=False)

# ----- locks & tableaux hints -----
def find_lock_ids(cand_df):
    lock=[]
    for _,r in cand_df.iterrows():
        t = r['text'].lower()
        if any(p in t for p in LOCK_PHRASES):
            lock.append(int(r['cand_id']))
    return sorted(set(lock))

def tableau_hint_ids_from_gold(cand_df, top_k=2):
    gold_sorted = gold.assign(length=gold['gold_text'].str.len()).sort_values('length', ascending=False).head(top_k)
    E_cand = sbert.encode(cand_df['text'].tolist(), normalize_embeddings=True)
    hints=set()
    for _,g in gold_sorted.iterrows():
        eg = sbert.encode([g['gold_text']], normalize_embeddings=True)
        sims = util.cos_sim(eg, E_cand)[0].cpu().numpy()
        j = int(np.argmax(sims))
        hints.add(int(cand_df.iloc[j]['cand_id']))
    return sorted(hints)

# ----- run baseline -----
base_align, base_scores = report(
    f"BASELINE (clause+punct{' +comma' if USE_COORD_COMMAS else ''}{' + Barthes-mode' if BARTHES_MODE else ''})",
    cand_base, "fivecodes_base"
)

# ----- focused LLM rounds (smarter acceptance) -----
def composite(align_df, f1_tuple):
    ov = float(align_df['overlap_score'].mean())
    si = float(align_df['sim_score'].mean())
    f1 = f1_tuple[2]
    return 0.6*f1 + 0.25*ov + 0.15*si  # weights are tunable

cand_cur = cand_base.copy()
cur_align, cur_scores = base_align, base_scores

# Extended anchors for prompt
anchors_extended = dict(ANCHORS)
anchors_extended["tableau_windows"] = ["Dance of the Living!", "a splendid salon ... fevered mind."]
anchors_extended["macro_rules"] = [
    "bundle_pivot_with_gesture",
    "antithesis_block_hold",
    "salon_tableau_hold",
    "borderline_hold"
]

for round_i in range(1, RUN_OPS_ROUNDS+1):
    # Optionally tighten the target count band each round
    if BARTHES_MODE:
        TARGET_COUNT_ALPHA = max(0.20, TARGET_COUNT_ALPHA * 0.5)

    worst = worst_mismatches(cand_cur, k=TOP_K_FEEDBACK)
    focus_ids=set()
    for (_, _, cand_id, _) in worst:
        for d in range(-LOCAL_WINDOW_RADIUS, LOCAL_WINDOW_RADIUS+1):
            nid = cand_id + d
            if 1 <= nid <= len(cand_cur): focus_ids.add(nid)
    lock_ids   = find_lock_ids(cand_cur)
    tableau_ids= tableau_hint_ids_from_gold(cand_cur, top_k=4 if BARTHES_MODE else 2)
    feedback = [{"gold_id": gid, "gold_text": gtxt, "cand_id": cid, "cand_text": ctxt}
                for (gid, gtxt, cid, ctxt) in worst]

    print(f"\n[Claude] Focused ops (round {round_i}) on ids: {sorted(focus_ids)} | locks: {lock_ids} | tableaux: {tableau_ids}")
    ops = call_claude_ops_focused(cand_cur, focus_ids, lock_ids, tableau_ids, anchors_extended, feedback=feedback)
    print("Proposed ops:", json.dumps(ops, ensure_ascii=False))

    cand_next = apply_ops_guarded(cand_cur, ops, lock_ids, tableau_ids)
    # (Optional) light post-ops polish
    cand_next = merge_crossing_entities(cand_next, src)
    cand_next = polish_microfragments(cand_next)

    # Evaluate and accept if better by F1, or fewer lexia, or better composite
    next_align, next_scores = report(f"FOCUSED LLM-OPS Round {round_i}", cand_next, f"fivecodes_ops_r{round_i}")
    if (next_scores[2] >= cur_scores[2] - 1e-9) or (len(cand_next) < len(cand_cur)) or (composite(next_align, next_scores) > composite(cur_align, cur_scores)):
        print("(Accept) Next is better or smaller; updating current.")
        cand_cur, cur_align, cur_scores = cand_next, next_align, next_scores
    else:
        print("(Guard) No improvement; keeping previous.")

# ----- FINAL RENDERING: bracketed lexia numbering -----
def render_numbered(cand_df):
    # one-per-line
    lines = []
    for _, r in cand_df.sort_values('start_char').iterrows():
        lines.append(f"[{int(r['cand_id'])}] {r['text']}")
    block = "\n\n".join(lines)
    with open("/content/lexia_render_lines.txt", "w", encoding="utf-8") as f:
        f.write(block)
    # inline paragraph (full text with markers)
    inline = " ".join(lines)
    with open("/content/lexia_render_inline.txt", "w", encoding="utf-8") as f:
        f.write(inline)
    print("\nRendered lexia saved:")
    print(" - /content/lexia_render_lines.txt (one lexia per line)")
    print(" - /content/lexia_render_inline.txt (single paragraph with [n] markers)")
    # also print a short preview
    print("\nPreview (first 8 lexia as lines):\n")
    print("\n\n".join(lines[:8]))

render_numbered(cand_cur)

print("\nDone. Files saved (latest round):")
print(" - /content/fivecodes_base_candidate_lexia.csv and _alignment.csv")
if RUN_OPS_ROUNDS>=1:
    print(f" - /content/fivecodes_ops_r{RUN_OPS_ROUNDS}_candidate_lexia.csv and _alignment.csv")
print(" - /content/lexia_render_lines.txt")
print(" - /content/lexia_render_inline.txt")
