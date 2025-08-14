# Print the whole paragraph with [n] markers inserted at each lexia start.
# Works with the variables from the previous cell; falls back to the saved CSV if needed.

import os, re, unicodedata, pandas as pd

def norm(s: str) -> str:
    s = unicodedata.normalize('NFKC', s)
    s = s.replace('“','"').replace('”','"').replace('’',"'").replace('—','-').replace('–','-')
    s = re.sub(r'\s+',' ', s).strip()
    return s

# 1) Get the candidate lexia DF (prefer the latest in-memory; else load from disk)
try:
    df = cand_cur.copy()  # from the last script
except NameError:
    # adjust filename if your latest round differs
    df = pd.read_csv('/content/fivecodes_ops_r2_candidate_lexia.csv')

# 2) Get the normalized source text used for offsets
try:
    src  # already defined
except NameError:
    # adjust path if needed
    BALZAC = '/content/balzac.txt'
    src = norm(open(BALZAC, encoding='utf-8').read())

# 3) Build the flowing paragraph with [n] markers inserted at the exact offsets
df = df.sort_values('start_char').reset_index(drop=True)
parts = []
cursor = 0
for _, r in df.iterrows():
    s, e = int(r['start_char']), int(r['end_char'])
    parts.append(src[cursor:s])                         # text before this lexia
    parts.append(f"[{int(r['cand_id'])}] ")             # the [n] marker + a space
    parts.append(src[s:e])                              # the lexia itself
    cursor = e
parts.append(src[cursor:])                               # tail after last lexia
inline = "".join(parts)

# 4) Print it and save it
print(inline)
with open('/content/lexia_render_full_inline.txt', 'w', encoding='utf-8') as f:
    f.write(inline)

print("\nSaved: /content/lexia_render_full_inline.txt")
