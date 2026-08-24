#!/usr/bin/env python3
"""Compare two DEBUG_SHADER_DUMP sections produced by the Phase-D demo."""
import argparse, csv, math, re, sys

def sections(path):
    current = None
    for line in open(path, encoding="utf8"):
        if line.startswith("# dump "):
            if current: yield current
            current = {"header": [line], "rows": []}
        elif current and line.startswith("#"):
            current["header"].append(line)
        elif current and line.startswith("shader,"):
            current["columns"] = next(csv.reader([line.strip()]))
        elif current and line.startswith("material_detail,"):
            current["rows"].append(next(csv.reader([line.strip()])))
    if current: yield current

def detail(section):
    cols = section["columns"]
    out = {}
    for row in section["rows"]:
        if row[0] == "material_detail":
            out[(row[1], row[2])] = [float(x) for x in row[3:]]
    return out

def phase(section):
    values = {r[10] for r in detail(section).values()} # f10
    return values.pop() if len(values) == 1 else None

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("dump"); args = parser.parse_args()
    ss = list(sections(args.dump))
    if len(ss) < 2: sys.exit("need two dump sections")
    # Automatic dumps can append an extra identical final frame.  Find the
    # latest D section and its nearest preceding C section with same camera.
    d = next((s for s in reversed(ss) if phase(s) == 1), None)
    if not d: sys.exit("no uniform Phase D section")
    camera = next((h for h in d["header"] if h.startswith("# camera")), "")
    c = next((s for s in reversed(ss[:ss.index(d)]) if phase(s) == 0 and camera in s["header"]), None)
    if not c: sys.exit("no matched uniform Phase C section")
    cd, dd = detail(c), detail(d)
    if phase(c) != 0 or phase(d) != 1: sys.exit("Mixed-mode capture: rejected")
    keys = cd.keys() & dd.keys()
    if not keys: sys.exit("no matched material fragments")
    deltas = [dd[k][1] - cd[k][1] for k in keys]
    raised = [x for x in deltas if x > .02]
    never_down = all(x >= -1e-5 for x in deltas)
    finite = all(math.isfinite(x) and 0.045 <= x <= 1 for k in keys for x in (cd[k][1], dd[k][1]))
    lengths = [dd[k][4] for k in keys]
    print(f"Matched fragments: {len(keys)}")
    print(f"Filtered normal length: min={min(lengths):.5f} mean={sum(lengths)/len(lengths):.5f}")
    print(f"Raised fragments: {len(raised)/len(keys)*100:.1f}%")
    print(f"Average/max roughness increase: {sum(deltas)/len(deltas):.6f}/{max(deltas):.6f}")
    print(f"Phase D active: {'yes' if any(x > 1e-5 for x in deltas) else 'no'}")
    print(f"Acceptance threshold met (>0.02): {'yes' if raised else 'no'}")
    print(f"Effective roughness never decreased: {'pass' if never_down else 'FAIL'}")
    print(f"Finite/in-range: {'pass' if finite else 'FAIL'}")
    print("Matched camera: pass (verify section headers before comparing)")
    print("Mixed-mode capture: no")
    return 0 if never_down and finite else 1
if __name__ == "__main__": raise SystemExit(main())
