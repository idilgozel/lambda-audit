#!/usr/bin/env bash
# Report the page count of the MAIN TEXT only (sections 1-6), which is what the
# 9-page ICLR limit governs. References, AI-use statement and appendices are
# excluded because they do not count toward it.
set -uo pipefail
cd "$(dirname "$0")"
python3 - <<'PY'
import pathlib, re
t = pathlib.Path("main.tex").read_text()
t = re.sub(r'\\subsubsection\*\{Use of AI\}.*?\\end\{document\}', r'\\end{document}', t, flags=re.S)
for tag, inp in [("intro","01-introduction"),("testbed","02-testbed"),("theory","03-theory"),
                 ("results","04-results"),("discussion","05-discussion"),("related","06-related")]:
    t = t.replace("\\input{sections/%s}" % inp,
                  "\\input{sections/%s}\n\\label{end:%s}" % (inp, tag))
pathlib.Path("maintext.tex").write_text(t)
PY
latexmk -pdf -interaction=nonstopmode maintext.tex >/dev/null 2>&1 || true
P=$(pdfinfo maintext.pdf 2>/dev/null | awk '/^Pages/{print $2}')
echo "MAIN TEXT: ${P:-?} pages   (limit 9)"
python3 - <<'PY'
import re, pathlib
try: aux = pathlib.Path("maintext.aux").read_text()
except FileNotFoundError: raise SystemExit
m = dict(re.findall(r'newlabel\{end:([a-z]+)\}\{\{[^}]*\}\{(\d+)\}', aux))
prev = 1
for name, key in [("Title+abstract+Intro","intro"),("Testbed","testbed"),("Theory","theory"),
                  ("Results","results"),("Discussion","discussion"),("Related","related")]:
    if key not in m: continue
    end = int(m[key]); print(f"  {name:24} ends p.{end:<3} ({end-prev} pp)"); prev = end
PY
latexmk -c maintext.tex >/dev/null 2>&1 || true
