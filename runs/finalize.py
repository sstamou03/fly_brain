"""Turn the hunger results into the app's final data, as soon as hunger.py has finished.

1. wait for results/hunger_<menu>.json (hunger.py writes it once, at the very end)
2. record the brain replays of every drink at the winner's hunger level (export_ui.py --hunger),
   so the chosen drink visibly makes her want to drink and the others do not
3. rebuild app/public/data (export_web.py)

usage: python -m runs.finalize [--menu cocktails] [--trials 6]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    print("\n$", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--menu", default="cocktails")
    ap.add_argument("--trials", type=int, default=6)
    args = ap.parse_args()

    results = ROOT / "results" / f"hunger_{args.menu}.json"
    waited = 0
    while not results.exists():
        time.sleep(30)
        waited += 30
        if waited % 600 == 0:
            print(f"still waiting for {results.name} ({waited // 60} min)", flush=True)

    rows = json.loads(results.read_text(encoding="utf-8"))
    chosen = next((r for r in rows if r["hunger"] is not None), None)
    if chosen is None:
        sys.exit("no drink is accepted even when starving; nothing to choose")
    print(f"choice: {chosen['drink']} from hunger {chosen['hunger']:.3f}", flush=True)

    run("export_ui.py", "--menu", args.menu, "--trials", str(args.trials), "--hunger", f"{chosen['hunger']:.4f}")
    run("export_web.py", "--hunger", str(results.relative_to(ROOT)))
    print("\nfinal app data ready in app/public/data — reload the page (or rebuild the Docker image)", flush=True)


if __name__ == "__main__":
    main()
