#!/usr/bin/env bash
# Summarise preserved logs; this is not live scheduler accounting.
# Optional argument: another repository/evidence snapshot root.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${1:-$(cd "$HERE/.." && pwd)}"
python3 - "$ROOT" <<'PY'
from pathlib import Path
import collections
import json
import re
import sys

root = Path(sys.argv[1])
logs = root / 'results' / 'logs'
if not logs.is_dir():
    raise SystemExit(f'No saved log directory: {logs}')
governing = {
    'lam_cap': ['prereg_lambda_pilot.py'],
    'lam_pilot': ['prereg_lambda_pilot.py'],
    'lam_e6': ['prereg_phase2.py'],
    'lam_e9': ['prereg_phase2.py'],
    'lam_e7': ['prereg_phase2.py', 'prereg_e7_bias.py'],
    'lam_e10': ['prereg_e10.py'],
    'lam_e11': ['prereg_e11.py'],
    'lam_e12p': ['prereg_e12.py'],
    'lam_e12': ['prereg_e12.py'],
    'E13-local': ['prereg_e13.py'],
}

def declarations(name):
    files = ['prereg_e14.py'] if name.startswith('lam_e14') else governing.get(name, [])
    if not files:
        return 'UNMAPPED'
    found = []
    for f in files:
        p = root / f
        text = p.read_text() if p.exists() else ''
        m = re.search(r'FROZEN (\d{4}-\d{2}-\d{2})', text)
        if not m:
            m = re.search(r'FROZEN_ON\s*=\s*[\"\x27](\d{4}-\d{2}-\d{2})', text)
        found.append(f"{f} ({m[1] if m else 'date not found'})")
    return '; '.join(found)

groups = collections.defaultdict(list)
for p in sorted(logs.iterdir()):
    fn = re.fullmatch(r'(.+)\.o(\d+)\.(\d+)', p.name)
    e13 = re.fullmatch(r'e13\.(\d+)\.log', p.name)
    if not fn and not e13:
        continue
    text = p.read_text()
    host = re.search(r'^host\s*:\s*(.+)$', text, re.M)
    host = host[1] if host else ''
    cls = re.fullmatch(r'node-([a-z0-9]+)-\d+\.myriad\.ucl\.ac\.uk', host)
    source = 'Myriad' if cls else 'local' if e13 or '(local,' in host else 'unknown'
    name, run_id, task = (fn[1], fn[2], int(fn[3])) if fn else ('E13-local', '-', int(e13[1]))
    start = re.search(r'^start(?:ing\s*|ed)\s*:\s*(\S+)', text, re.M)
    finish = re.search(r'^finished\s*:\s*(\S+)', text, re.M)
    rows = 0
    for line in text.splitlines():
        if line.startswith('{"run":'):
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f'Invalid result JSON in {p}: {e}')
            rows += 1
    groups[(source, name, run_id)].append({
        'task': task, 'start': start[1] if start else None,
        'finish': finish[1] if finish else None, 'rows': rows,
        'class': cls[1] if cls else host,
    })

records = []
for (source, name, run_id), tasks in groups.items():
    starts = sorted(t['start'] for t in tasks if t['start'])
    finishes = sorted(t['finish'] for t in tasks if t['finish'])
    records.append((starts[0] if starts else '?', source, name, run_id, tasks, finishes))

print('Saved-log chronology; classification follows recorded host lines.')
print('Dates below are declarations in current files, not independently verified freeze times.')
print('Local run labels are not SGE job IDs. Last finish covers only tasks with finish markers.\n')
print('source\tjob\tjob/run id\tlogs\tresult rows\tfinish markers\tfirst start UTC\tlast recorded finish UTC\tclasses/host\tdeclared freezes')
for first, source, name, run_id, tasks, finishes in sorted(records):
    classes = ','.join(sorted({t['class'] for t in tasks if t['class']})) or '(see local provenance)'
    fields = [source, name, run_id, str(len(tasks)), str(sum(t['rows'] for t in tasks)), str(len(finishes)), first, finishes[-1] if finishes else '?', classes, declarations(name)]
    print('\t'.join(fields))
    missing_rows = [str(t['task']) for t in tasks if not t['rows']]
    missing_start = [str(t['task']) for t in tasks if not t['start']]
    missing_finish = [str(t['task']) for t in tasks if not t['finish']]
    if missing_rows:
        print('  No result row: tasks ' + ', '.join(missing_rows))
    if missing_start:
        print('  No start marker: tasks ' + ', '.join(missing_start))
    if missing_finish:
        note = ' (E13 runner does not write finish markers)' if name == 'E13-local' else ''
        print('  No finish marker: tasks ' + ', '.join(missing_finish) + note)

print('\nSaved stdout supports run chronology and row matching; it does not authenticate historical freeze-before-data claims.')
print('Job IDs alone do not establish preregistration time. See PROVENANCE.md for Git evidence and its limits.')
PY
