"""Generate the paper's ledger counts from the scorers, so prose cannot drift.

Written 2026-08-21. paper/01-intro-arc.md asserted "of the eighteen predictions
we registered across five experiments, six failed" and, forty lines later, "six
registered failures reported alongside nine passes". 6 + 9 = 15, not 18, and
README.md carried a third figure. The draft's own revision note asked for the
counts to be generated rather than typed; this is that script.

    python3 analysis/recount_ledger.py              # counts + markdown table
    python3 analysis/recount_ledger.py --check      # exit 1 if the README
                                                    # headline disagrees

WHAT IT COUNTS, AND THE DISTINCTION THAT CAUSED THE ERROR
---------------------------------------------------------
The arithmetic broke because capability GATES were counted as predictions when
they passed (the pilot's G1a-c) and omitted when they failed (E11's G11a/G11b).
A gate is not a finding: it establishes that an experiment is capable of
testing what it claims to test, and a failed gate means the predictions behind
it are NOT SCORED rather than refuted. Gates and predictions are therefore
tallied separately here, and gates are reported in full in both directions.

G2 and G3 are counted as PREDICTIONS, not gates, despite their names: the pilot
froze G1a-c as preconditions and G2/G3 as substantive claims about what a
lambda sweep would show. Their failure is a finding and is counted as one.

Verdicts are parsed from each scorer's own output. Nothing is hard-coded except
the classification above and the list of scorers, so a re-scored experiment
changes these numbers automatically.
"""
import argparse
import ast
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "results")

# (scorer, argv tail, results file, label). A scorer that cannot run is
# reported as such rather than silently contributing nothing.
SCORERS = [
    ("prereg_lambda_pilot.py", [], "results_lambda_pilot.jsonl", "pilot"),
    ("prereg_phase2.py",       [], "results_phase2.jsonl",       "E6/E7/E9"),
    ("prereg_e7_bias.py", ["--score"], "results_phase2.jsonl",   "E7 mechanism"),
    ("prereg_e10.py",          [], "results_e10.jsonl",          "E10"),
    ("prereg_e8.py",     ["--e11"], "results_e11.jsonl",         "E8 diagnostic"),
    ("prereg_e11.py",          [], "results_e11.jsonl",          "E11"),
    ("prereg_e12.py",          [], "results_e12.jsonl",          "E12"),
]

# Capability gates. Everything else parsed out of a VERDICTS line is a
# registered prediction.
GATES = {"G1a", "G1b", "G1c", "G11a", "G11b", "G12a", "G12b", "G12c"}

VERDICT_RE = re.compile(r"^VERDICTS:\s*(\{.*\})\s*$", re.M)
# prereg_e8.py reports P13.1 on its own line rather than in a VERDICTS dict.
LOOSE_RE = re.compile(r"^(P\d+\.\d+):\s.*->\s*(PASS|FAIL)\s*$", re.M)


def normalise(v):
    if isinstance(v, bool):
        return "PASS" if v else "FAIL"
    return str(v).upper()


def run(scorer, tail, results):
    path = os.path.join(RES, results)
    if not os.path.isfile(path):
        return None, f"{results} absent"
    p = subprocess.run([sys.executable, os.path.join(ROOT, scorer)] + tail + [path],
                       capture_output=True, text=True)
    out = p.stdout + p.stderr
    verdicts = {}
    for m in VERDICT_RE.finditer(out):
        try:
            for k, v in ast.literal_eval(m.group(1)).items():
                verdicts[k] = normalise(v)
        except (ValueError, SyntaxError):
            pass
    for m in LOOSE_RE.finditer(out):
        verdicts.setdefault(m.group(1), m.group(2))
    if not verdicts:
        return None, ("scorer produced no VERDICTS line "
                      f"(exit {p.returncode})")
    return verdicts, None


def collect():
    preds, gates, problems = {}, {}, []
    for scorer, tail, results, label in SCORERS:
        v, err = run(scorer, tail, results)
        if err:
            problems.append(f"{label:14} ({scorer}): {err}")
            continue
        for k, verdict in v.items():
            (gates if k in GATES else preds)[k] = (verdict, label)
    return preds, gates, problems


def bucket(d):
    out = {"PASS": [], "FAIL": [], "PARTIAL": [], "OTHER": []}
    for k, (verdict, _) in sorted(d.items()):
        out.get(verdict, out["OTHER"]).append(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if README.md's headline disagrees")
    a = ap.parse_args()

    preds, gates, problems = collect()
    pb, gb = bucket(preds), bucket(gates)
    npass, nfail, npart = len(pb["PASS"]), len(pb["FAIL"]), len(pb["PARTIAL"])
    total = npass + nfail + npart + len(pb["OTHER"])

    print("=" * 74)
    print(" REGISTERED PREDICTIONS")
    print("=" * 74)
    for b in ("PASS", "FAIL", "PARTIAL", "OTHER"):
        if pb[b]:
            print(f"  {b:8} {len(pb[b]):2}   {', '.join(pb[b])}")
    print(f"\n  HEADLINE: {npass} pass / {nfail} fail"
          + (f" / {npart} partial" if npart else "") + f" of {total}")

    print("\n" + "=" * 74)
    print(" CAPABILITY GATES (reported separately, in both directions)")
    print("=" * 74)
    for b in ("PASS", "FAIL", "OTHER"):
        if gb[b]:
            print(f"  {b:8} {len(gb[b]):2}   {', '.join(gb[b])}")
    if gb["FAIL"]:
        print("\n  A failed gate means the predictions behind it are NOT SCORED,")
        print("  not that they were refuted. Those must be named in the paper.")

    if problems:
        print("\n" + "=" * 74)
        print(" NOT COUNTED")
        print("=" * 74)
        for p in problems:
            print(f"  {p}")
        print("\n  Every line above is a prediction absent from the tally.")

    print("\n" + "=" * 74)
    print(" LEDGER ROWS (paste into the paper)")
    print("=" * 74)
    print("| id | verdict | source |")
    print("|---|---|---|")
    for k, (verdict, label) in sorted(preds.items()):
        mark = f"**{verdict}**" if verdict != "PASS" else verdict
        print(f"| {k} | {mark} | {label} |")

    if a.check:
        readme = os.path.join(ROOT, "README.md")
        txt = open(readme).read() if os.path.isfile(readme) else ""
        m = re.search(r"(\d+)\s+predictions passed,\s*(\d+)\s+failed", txt)
        if not m:
            print("\n--check: no parsable headline in README.md")
            return 1
        if (int(m.group(1)), int(m.group(2))) != (npass, nfail):
            print(f"\n--check: README says {m.group(1)}/{m.group(2)}, "
                  f"scorers say {npass}/{nfail}. MISMATCH")
            return 1
        print(f"\n--check: README headline {npass}/{nfail} agrees with scorers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
