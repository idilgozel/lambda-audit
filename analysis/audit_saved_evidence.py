"""Cross-check saved lambda-audit result rows against their exact log lines.

Read-only by default. Does not train, score predictions, alter evidence, or
infer freeze-before-data authentication. Optional --json-out saves an audit.
"""
from pathlib import Path
import argparse
import datetime as dt
import collections
import csv
import hashlib
import json
import re
import shlex

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
ap.add_argument("--json-out", type=Path)
a = ap.parse_args()
ROOT = a.root.resolve()
if not (ROOT / "results/logs").is_dir():
    ap.error(f"No saved logs under {ROOT}")


def key(row):
    return (row.get("env"), row.get("arch"), row.get("lam"), row.get("seed"))


def options(s):
    tokens = shlex.split(s)
    return {a: b for a, b in zip(tokens, tokens[1:]) if a.startswith("--")}


def parse_log(path):
    txt = path.read_text()
    fn = re.fullmatch(r"(.+)\.o(\d+)\.(\d+)", path.name)
    host = re.search(r"^host\s*:\s*(.+)$", txt, re.M)
    start = re.search(r"^start(?:ing\s*|ed)\s*:\s*(\S+)", txt, re.M)
    finish = re.search(r"^finished\s*:\s*(\S+)", txt, re.M)
    args = re.search(r"^(?:args\s*|cmd)\s*:\s*(.+)$", txt, re.M)
    task = re.search(r"^task\s*:\s*(\d+) -> config line (\d+), seed (\d+)", txt, re.M)
    versions = re.search(r"^torch\s*:\s*(.+)$", txt, re.M)
    rows = []
    for n, line in enumerate(txt.splitlines(), 1):
        if line.startswith('{"run":'):
            rows.append({"line": n, "raw": line, "row": json.loads(line)})
    hostname = host[1] if host else None
    source = "myriad" if hostname and re.fullmatch(r"node-[a-z0-9]+-\d+\.myriad\.ucl\.ac\.uk", hostname) else "local" if (hostname and "(local," in hostname) or path.name.startswith("e13.") else "unknown"
    opts = options(args[1]) if args else {}
    mismatches = []
    if fn and task and int(fn[3]) != int(task[1]):
        mismatches.append({"field": "task", "filename": fn[3], "header": task[1]})
    if start and finish and finish[1] < start[1]:
        mismatches.append({"field": "time", "start": start[1], "finish": finish[1]})
    for rr in rows:
        row = rr['row']
        for opt, field, cast in (("--env", "env", str), ("--arch", "arch", str), ("--gae-lambda", "lam", float), ("--total-timesteps", "total_timesteps", int), ("--gamma", "gamma", float), ("--fix-sigma", "fix_sigma", float)):
            if opt in opts and row.get(field) != cast(opts[opt]):
                mismatches.append({"line": rr['line'], "field": field, "args": opts[opt], "result": row.get(field)})
        if task and row.get('seed') != int(task[3]):
            mismatches.append({"field": "seed", "header": task[3], "result": row.get('seed')})
        if '--seed' in opts and row.get('seed') != int(opts['--seed']):
            mismatches.append({"field": "seed", "command": opts['--seed'], "result": row.get('seed')})
    return {"path": str(path.relative_to(ROOT)), "source": source, "job_name": fn[1] if fn else "E13-local", "job_id": fn[2] if fn else None, "task": int(fn[3]) if fn else int(re.search(r"e13\.(\d+)", path.name)[1]), "host": hostname, "start": start[1] if start else None, "finish": finish[1] if finish else None, "versions": versions[1] if versions else None, "rows": rows, "header_mismatches": mismatches}


logs = [parse_log(p) for p in sorted((ROOT/'results/logs').iterdir()) if re.fullmatch(r".+\.o\d+\.\d+|e13\.\d+\.log", p.name)]
index = collections.defaultdict(list)
by_key = collections.defaultdict(list)
for log in logs:
    for rr in log['rows']:
        record = {"path": log['path'], "line": rr['line'], "source": log['source']}
        index[rr['raw']].append(record)
        by_key[key(rr['row'])].append({**record, 'row': rr['row']})

files = []
saved_raw = set()
for p in sorted((ROOT/'results/results').glob('results*.jsonl')):
    lines = [(n, line) for n, line in enumerate(p.read_text().splitlines(), 1) if line.strip()]
    rows = [json.loads(line) for _, line in lines]
    matches = []
    for n, line in lines:
        saved_raw.add(line)
        row = json.loads(line)
        matches.append({"line": n, "key": key(row), "exact_log_matches": index[line], "same_key_log_count": len(by_key[key(row)])})
    duplicates = [{"key": k, "count": n} for k,n in collections.Counter(map(key, rows)).items() if n > 1]
    files.append({"path": str(p.relative_to(ROOT)), "rows": len(rows), "exact_matched_rows": sum(bool(m['exact_log_matches']) for m in matches), "duplicate_keys": duplicates, "matches": matches})

jobs = []
groups = collections.defaultdict(list)
for log in logs:
    groups[(log['source'], log['job_name'], log['job_id'])].append(log)
for (source, name, jid), ls in groups.items():
    starts = sorted(l['start'] for l in ls if l['start'])
    finishes = sorted(l['finish'] for l in ls if l['finish'])
    classes = sorted({re.search(r'node-([a-z0-9]+)-',l['host'])[1] for l in ls if l['host'] and re.search(r'node-([a-z0-9]+)-',l['host'])})
    jobs.append({"source": source, "job_name": name, "job_id": jid, "logs": len(ls), "result_rows": sum(len(l['rows']) for l in ls), "with_finish": sum(bool(l['finish']) for l in ls), "first_start": starts[0] if starts else None, "last_recorded_finish": finishes[-1] if finishes else None, "classes": classes, "versions": sorted({l['versions'] for l in ls if l['versions']}), "tasks_without_result": sorted(l['task'] for l in ls if not l['rows']), "tasks_without_finish": sorted(l['task'] for l in ls if not l['finish'])})
jobs.sort(key=lambda j:j['first_start'] or '')
orphans = [{"path":l['path'],"line":r['line'],"key":key(r['row'])} for l in logs for r in l['rows'] if r['raw'] not in saved_raw]
cluster = [l for l in logs if l['source']=='myriad']
audit = {"snapshot": json.loads((ROOT/'manifest.json').read_text())['snapshot_utc'] if (ROOT/'manifest.json').exists() else dt.datetime.now(dt.timezone.utc).isoformat(), "summary":{"myriad_logs":len(cluster),"myriad_result_rows_in_logs":sum(len(l['rows']) for l in cluster),"myriad_logs_missing_start":sum(not l['start'] for l in cluster),"myriad_logs_missing_finish":sum(not l['finish'] for l in cluster),"header_mismatches":sum(len(l['header_mismatches']) for l in logs),"saved_rows":sum(f['rows'] for f in files),"saved_rows_without_exact_log_match":sum(f['rows']-f['exact_matched_rows'] for f in files),"orphan_log_rows":len(orphans)},"jobs":jobs,"result_files":files,"orphan_log_rows":orphans,"logs":logs}
if a.json_out:
    a.json_out.write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit['summary'],indent=2))
for j in jobs: print(json.dumps(j))
for f in files: print(f['path'], f['exact_matched_rows'], '/', f['rows'], 'duplicate keys:', f['duplicate_keys'])
print('Orphan log rows:',orphans)

failures = (audit['summary']['header_mismatches']
            + audit['summary']['saved_rows_without_exact_log_match']
            + audit['summary']['orphan_log_rows']
            + sum(len(f['duplicate_keys']) for f in files))
raise SystemExit(1 if failures else 0)
