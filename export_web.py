"""Assemble the Fly Bar web app data (app/public/data) from the simulation exports.

Inputs
  ui/data.json                 brain replays per drink (export_ui.py)
  results/hunger_<menu>.json   how hungry the fly must be to drink each one (hunger.py), optional
  ui/rates/<slug>.npy          whole-brain mean firing per drink (export_ui.py), optional
  web_data/brain_*             neuron positions (export_brain_cloud.py)

The app speaks in words, not measurements: every drink gets `likes` (0-100), a mood and two
plain traces over the tasting ("wants to drink" = feeding motor neuron activity, "tastes bitter" =
the relays that receive most bitter input), plus which neurons light up in the 3D brain.

usage: python export_web.py [--data ui/data.json] [--hunger results/hunger_cocktails.json]
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "app" / "public" / "data"
FIRE_HZ = 60.0       # a trace reads "full" at this firing rate
CLOUD_HZ = 20.0      # brain points are fully lit at this mean rate
LIT_HZ = 2.0         # a neuron "lights up" above this mean rate


def region(t, c, s):
    """Plain-language brain region of a neuron from its type, class and superclass."""
    t, c, s = (x if isinstance(x, str) else "" for x in (t, c, s))  # missing annotations come as NaN
    if c == "gustatory":
        return "γεύση"
    if c in ("olfactory", "ALPN", "ALLN", "ALIN", "ALON") or t.startswith(("lLN", "vLN", "l2LN")):
        return "όσφρηση"
    if c in ("Kenyon_Cell", "MBON", "DAN"):
        return "μνήμη και ανταμοιβή"
    if c == "CX":
        return "πλοήγηση"
    if s.startswith(("ol_", "visual")) or c == "visual":
        return "όραση"
    if "motor" in s or "efferent" in s:
        return "μύες"
    if s == "descending_neuron":
        return "εντολές προς το σώμα"
    if s.startswith("vnc") or s == "ascending_neuron":
        return "νευρικό κορδόνι (πόδια, φτερά)"
    if t.startswith(("GNG", "PRW", "SEZ")):
        return "κέντρο γεύσης και σίτισης"
    if "sensory" in s:
        return "αισθήσεις"
    return "κεντρικός εγκέφαλος"

#                    glass        liquid     garnish
STYLE = {
    "Negroni":          ("rocks",     "#A3201A", "orange"),
    "Mojito":           ("highball",  "#D8EDBE", "mint"),
    "Paloma":           ("highball",  "#F2A99A", "grapefruit"),
    "Old Fashioned":    ("rocks",     "#A85A1C", "orange"),
    "Pornstar Martini": ("martini",   "#F0BC45", "passion"),
    "Aperol Spritz":    ("wine",      "#F0651C", "orange"),
    "Cosmopolitan":     ("martini",   "#D93E72", "lime"),
    "Piña Colada":      ("hurricane", "#F4E4B4", "pineapple"),
    "Gin & Tonic":      ("highball",  "#E4F1EE", "lime"),
    "Lager Beer":       ("pint",      "#E09A2C", "foam"),
}
RECIPE = [("sugar_gL", "Ζάχαρη", "g/L"), ("abv_pct", "Αλκοόλ", "%"), ("caffeine_mgL", "Καφεΐνη", "mg/L"),
          ("ibu", "Πικράδα", ""), ("salt_gL", "Αλάτι", "g/L"), ("co2_gL", "Ανθρακικό", "g/L"), ("ph", "pH", "")]


def slugify(name):
    return re.sub(r"[^0-9A-Za-zΑ-Ωα-ωά-ώ]+", "_", name).strip("_")


def smooth(x, k=3):
    return np.convolve(np.pad(x, k // 2, mode="edge"), np.ones(k) / k, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ui/data.json")
    ap.add_argument("--hunger", default="results/hunger_cocktails.json")
    args = ap.parse_args()

    d = json.loads((ROOT / args.data).read_text(encoding="utf-8"))
    (OUT / "cloud").mkdir(parents=True, exist_ok=True)
    for f in ["brain_xyz.bin", "brain_group.bin", "brain_cloud.json"]:
        shutil.copy(ROOT / "web_data" / f, OUT / f)
    point_idx = np.fromfile(ROOT / "web_data" / "brain_idx.bin", dtype=np.int32)
    meta = pd.read_parquet(ROOT / "brain_meta.parquet", columns=["idx", "type", "class", "superclass"]).sort_values("idx")
    regions = np.array([region(t, c, s) for t, c, s in zip(meta["type"], meta["class"], meta["superclass"])])

    hunger_path = ROOT / args.hunger
    hunger = None
    if hunger_path.exists():
        hunger = {r["drink"].split(" ", 1)[1]: r for r in json.loads(hunger_path.read_text(encoding="utf-8"))}

    motor = next(n["id"] for n in d["nodes"] if n["kind"] == "motor")
    from_bitter = sorted((e for e in d["edges"] if e["source"] == "ch:bitter"), key=lambda e: -e["synapses"])
    bitter_relays = [e["target"] for e in from_bitter if e["target"] != motor][:8]
    pop_max = max(max(x["pop_hz"]) for x in d["drinks"]) or 1.0

    drinks = []
    for x in d["drinks"]:
        name = x["name"]
        glass, liquid, garnish = STYLE.get(name, ("rocks", "#C98B3A", "none"))
        slug = x.get("slug") or slugify(name)
        act = x["activity"]
        tv = x["taste_vector"]
        cloud = None
        rates_file = ROOT / "ui" / "rates" / f"{slug}.npy"
        if rates_file.exists():
            rates = np.load(rates_file).astype(np.float32)
            q = np.clip(rates[point_idx] / CLOUD_HZ * 255, 0, 255).astype(np.uint8)
            cloud_name = f"cloud/{slugify(name).lower()}.bin"
            q.tofile(OUT / cloud_name)
            lit = rates > LIT_HZ
            names, counts = np.unique(regions[lit], return_counts=True)
            top = sorted(zip(names.tolist(), counts.tolist()), key=lambda p: -p[1])[:6]
            cloud = {"file": cloud_name, "lit": int(lit.sum()),
                     "regions": [{"name": nm, "lit": int(k)} for nm, k in top]}
        hrow = hunger.get(name, {}) if hunger else {}
        need = hrow.get("hunger")
        drinks.append({
            "id": slugify(name).lower(), "name": name, "emoji": x["emoji"],
            "glass": glass, "liquid": liquid, "garnish": garnish,
            "recipe": [{"label": lab, "value": f"{x['drink_vector'][k]:g}{(' ' + unit) if unit else ''}"}
                       for k, lab, unit in RECIPE if x["drink_vector"][k] or k in ("sugar_gL", "abv_pct")],
            "taste": {"sweet": tv["sweet"], "bitter": x.get("bitter_fed", tv["bitter"]), "fizz": tv["fizz"],
                      "salt": max(tv["ir94e"], tv["low_salt"], tv["high_salt"]), "water": tv["water"]},
            "hungerNeeded": need,
            "desire": smooth(np.clip(np.array(act[motor]) / FIRE_HZ, 0, 1)).round(3).tolist(),
            "aversion": smooth(np.clip(np.mean([act[r] for r in bitter_relays], axis=0) / FIRE_HZ, 0, 1)).round(3).tolist(),
            "pulse": smooth(np.array(x["pop_hz"]) / pop_max).round(3).tolist(),
            "cloud": cloud,
            "_hz": hrow.get("hz_at_threshold", x["mn9_hz"]), "_v": x["mn9_v_mean"],
        })

    provisional = hunger is None
    if provisional:
        drinks.sort(key=lambda r: (-r["_hz"], -r["_v"]))
    else:
        drinks.sort(key=lambda r: (r["hungerNeeded"] is None, r["hungerNeeded"] or 1.0, -r["_hz"]))
    n = len(drinks)
    # relative scale: every cocktail needs a lot of hunger, so compare drinks with each other
    needs = [r["hungerNeeded"] for r in drinks if r["hungerNeeded"] is not None]
    lo, hi = (min(needs), max(needs)) if needs else (0.0, 1.0)
    for rank, r in enumerate(drinks, 1):
        r["rank"] = rank
        if provisional:
            r["likes"] = round(90 - 80 * (rank - 1) / max(n - 1, 1))
        elif r["hungerNeeded"] is None:
            r["likes"] = 0
        else:
            spread = (hi - r["hungerNeeded"]) / (hi - lo) if hi > lo else 1.0
            r["likes"] = round(12 + 88 * spread)
    # moods follow the acceptance tiers (drinks accepted at the same level share a word), not the bar
    # length: the winner can be far ahead and would otherwise push every other drink into "yuck"
    tiers = sorted({r["hungerNeeded"] for r in drinks if r["hungerNeeded"] is not None})
    tier_mood = ["like", "meh", "yuck"]
    for rank, r in enumerate(drinks, 1):
        if rank == 1:
            r["mood"] = "love"
        elif provisional:
            r["mood"] = "like" if r["likes"] >= 60 else "meh" if r["likes"] >= 35 else "yuck"
        elif r["hungerNeeded"] is None:
            r["mood"] = "yuck"
        else:
            later = [h for h in tiers if h > tiers[0]]  # tiers after the winner's
            k = later.index(r["hungerNeeded"]) if r["hungerNeeded"] in later else 0
            r["mood"] = tier_mood[min(k * len(tier_mood) // max(len(later), 1), len(tier_mood) - 1)]
        del r["_hz"], r["_v"]

    order = {name: i for i, name in enumerate(x["name"] for x in d["drinks"])}
    drinks.sort(key=lambda r: order[r["name"]])  # counter order = menu order
    cloud_info = json.loads((ROOT / "web_data" / "brain_cloud.json").read_text(encoding="utf-8"))
    out = {
        "binMs": d["bin_ms"], "tasteMs": d["t_run_ms"], "hunger": d.get("hunger", 0.0), "provisional": provisional,
        "winner": next(r["id"] for r in drinks if r["rank"] == 1),
        "neurons": cloud_info["neurons_total"], "points": cloud_info["count"], "drinks": drinks,
    }
    (OUT / "drinks.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT / 'drinks.json'}  provisional={provisional}  winner={out['winner']}  "
          f"clouds={sum(r['cloud'] is not None for r in drinks)}/{n}")


if __name__ == "__main__":
    main()
