"""Rebuild results/hunger_<menu>.json from hunger.py's printed log, for the drinks it finished.

Used when the run is stopped early (the beer was dropped from the menu while it was being measured).
Only drinks whose "-> drinks it from hunger" line (or "starving" with no bisection) is complete and that
are still on the menu are kept; the ordering rule is the same as hunger.py's.

usage: python -m runs.hunger_from_log <log file> [--menu cocktails]
"""
import argparse
import json
import re
from pathlib import Path

from brain.taste import MENUS

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--menu", default="cocktails")
    args = ap.parse_args()

    menu = set(MENUS[args.menu])
    rows, cur = [], None
    for line in Path(args.log).read_text(encoding="utf-8").splitlines():
        if m := re.match(r"== (.+?)  bitter [\d.]+  starving \(h=1\): ([\d.]+) Hz", line):
            cur = {"drink": m.group(1), "tried": {"1.0000": float(m.group(2))}}
        elif (m := re.match(r"\s+h=([\d.]+)\s+([\d.]+) Hz", line)) and cur:
            cur["tried"][f"{float(m.group(1)):.4f}"] = float(m.group(2))
        elif (m := re.match(r"\s+-> drinks it from hunger ([\d.]+)", line)) and cur:
            h = float(m.group(1))
            cur.update(hunger=h, hz_at_threshold=cur["tried"][f"{h:.4f}"])
            rows.append(cur)
            cur = None
    rows = [r for r in rows if r["drink"] in menu]
    missing = sorted(menu - {r["drink"] for r in rows})
    if missing:
        raise SystemExit(f"not finished in the log: {missing}")
    rows.sort(key=lambda r: (r["hunger"], -r["hz_at_threshold"]))
    out = ROOT / "results" / f"hunger_{args.menu}.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("order:", " > ".join(f"{r['drink']} ({r['hunger']:.3f}, {r['hz_at_threshold']:.1f} Hz)" for r in rows))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
