"""Turn Slay the Spire .run files into one row per (card reward screen, candidate).

Run from the project root:  python3 src/parse_runs.py
Output: data/processed/card_choices.csv
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed/card_choices.csv")

START_DECK = {
    "IRONCLAD": ["Strike_R"] * 5 + ["Defend_R"] * 4 + ["Bash"],
    "THE_SILENT": ["Strike_G"] * 5 + ["Defend_G"] * 5 + ["Survivor", "Neutralize"],
    "DEFECT": ["Strike_B"] * 4 + ["Defend_B"] * 4 + ["Zap", "Dualcast"],
    "WATCHER": ["Strike_P"] * 4 + ["Defend_P"] * 4 + ["Eruption", "Vigilance"],
}
SKIP_PICKS = {"SKIP", "Skip", "Singing Bowl"}


def base(name):
    return re.sub(r"\+\d+$", "", name)


def act_of(floor):
    return 1 if floor <= 16 else 2 if floor <= 33 else 3 if floor <= 50 else 4


def at_floor(series, floor):
    if not series:
        return None
    return series[min(max(floor - 1, 0), len(series) - 1)]


def parse_run(path):
    r = json.loads(path.read_text())
    char = r["character_chosen"]
    deck = Counter(base(c) for c in START_DECK[char])

    known_cards = {base(c) for c in r["master_deck"]}
    for ch in r["card_choices"]:
        known_cards.add(base(ch["picked"]))
        known_cards.update(base(c) for c in ch["not_picked"])

    other = [(f, "purge", base(n)) for n, f in zip(r["items_purged"], r["items_purged_floors"])]
    other += [(f, "buy", base(n)) for n, f in zip(r["items_purchased"], r["item_purchase_floors"])
              if base(n) in known_cards]
    for e in r["event_choices"]:
        ef = int(e["floor"])
        other += [(ef, "purge", base(c)) for c in e.get("cards_removed", [])]
        other += [(ef, "purge", base(c)) for c in e.get("cards_transformed", [])]
        other += [(ef, "buy", base(c)) for c in e.get("cards_obtained", [])]
    other.sort(key=lambda e: e[0])

    def apply(kind, name):
        if kind == "purge" and deck[name] > 0:
            deck[name] -= 1
        elif kind == "buy":
            deck[name] += 1

    rows, i = [], 0
    for n, ch in enumerate(sorted(r["card_choices"], key=lambda c: c["floor"])):
        floor = int(ch["floor"])
        while i < len(other) and other[i][0] < floor:
            apply(other[i][1], other[i][2])
            i += 1

        hp = at_floor(r["current_hp_per_floor"], floor)
        max_hp = at_floor(r["max_hp_per_floor"], floor)
        snapshot = {k: v for k, v in deck.items() if v > 0}
        picked_skip = ch["picked"] in SKIP_PICKS
        candidates = [base(c) for c in ch["not_picked"]]
        if not picked_skip:
            candidates.append(base(ch["picked"]))
        candidates.append("SKIP")
        chosen = "SKIP" if picked_skip else base(ch["picked"])

        for cand in candidates:
            rows.append({
                "screen_id": f"{path.stem}_{n}",
                "run_id": path.stem,
                "character": char,
                "ascension": r["ascension_level"],
                "floor": floor,
                "act": act_of(floor),
                "hp": hp,
                "max_hp": max_hp,
                "hp_ratio": round(hp / max_hp, 3) if hp and max_hp else None,
                "deck_size": sum(snapshot.values()),
                "deck": json.dumps(snapshot, sort_keys=True),
                "candidate": cand,
                "is_skip": int(cand == "SKIP"),
                "picked": int(cand == chosen),
                "victory": int(r["victory"]),
                "floor_reached": r["floor_reached"],
                "floors_after": r["floor_reached"] - floor,
            })
        if not picked_skip:
            deck[base(ch["picked"])] += 1

    # Validation: does the rebuilt final deck match the real final deck?
    while i < len(other):
        apply(other[i][1], other[i][2])
        i += 1
    real = Counter(base(c) for c in r["master_deck"])
    mismatch = sum(((deck - real) + (real - deck)).values())
    return rows, mismatch


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_rows, mismatches, exact = [], [], 0
    for path in sorted(RAW.glob("*/*.run")):
        rows, mm = parse_run(path)
        all_rows += rows
        if rows:
            mismatches.append(mm)
            exact += mm == 0
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    screens = len({r["screen_id"] for r in all_rows})
    print(f"rows={len(all_rows)} screens={screens} -> {OUT}")
    print(f"deck rebuild: {exact}/{len(mismatches)} runs match the real final deck exactly, "
          f"mean card difference {sum(mismatches) / len(mismatches):.2f}")


if __name__ == "__main__":
    main()
