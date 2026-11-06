"""Compare what the value models would pick with what players picked, card by card.

Scores real Ironclad card screens from the human data (A15+) with each model and prints, for
every model, the cards it takes more often than players do and the cards it takes less often.
"Offered" counts every time a card appeared on a screen; a pick rate is the share of those
appearances where the card was chosen.

    python3 src/compare_picks.py
    python3 src/compare_picks.py --models models/sim_value_model.json --top 20

A difference is a question, not an error: the human-trained model learned from player choices
and outcomes, the simulator-trained one learned what works for the simulator's combat bot, and
the screens here are human decks that the simulator-trained model has not seen. Ironclad only.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from xgboost import XGBRegressor

import sim_eval as se

CSV = "data/processed/card_choices_big.csv"
COLS = ["screen_id", "character", "ascension", "floor", "hp", "max_hp", "deck", "candidate",
        "picked", "relics", "upgrades"]


def num(v):
    return 0.0 if pd.isna(v) else float(v)


def load_screens(path, max_screens, seed):
    raw = pd.read_csv(path, usecols=COLS)
    raw = raw[raw["character"] == "IRONCLAD"]
    if raw.empty:
        raise SystemExit(f"no IRONCLAD rows in {path}")
    ids = raw["screen_id"].unique()
    if len(ids) > max_screens:
        keep = np.random.default_rng(seed).choice(ids, max_screens, replace=False)
        raw = raw[raw["screen_id"].isin(keep)]
    screens = []
    for _, g in raw.groupby("screen_id", sort=False):
        picked = np.flatnonzero(g["picked"].to_numpy() == 1)
        names = [None if c == "SKIP" else c for c in g["candidate"]]
        if len(picked) != 1 or None not in names:
            continue
        first = g.iloc[0]
        state = {"deck": Counter(json.loads(first["deck"])),
                 "ups": Counter(json.loads(first["upgrades"])),
                 "relics": json.loads(first["relics"]),
                 "hp": num(first["hp"]), "max_hp": num(first["max_hp"]),
                 "floor": int(first["floor"])}
        screens.append({"state": state, "names": names, "picked": int(picked[0]),
                        "ascension": int(first["ascension"]), "skip": names.index(None)})
    return screens


def top_choices(model_path, screens):
    """Index of the model's top candidate on every screen, and the offered card names the
    model's vocabulary does not contain (it cannot tell those cards apart from nothing)."""
    model_path = Path(model_path)
    meta_path = model_path.with_name(model_path.stem + ".meta.json")
    if not (model_path.exists() and meta_path.exists()):
        raise SystemExit(f"need {model_path} and {meta_path}")
    fz = se.Featurizer(json.loads(meta_path.read_text()), 5)
    model = XGBRegressor()
    model.load_model(str(model_path))
    mats = []
    for s in screens:
        fz.ascension = s["ascension"]
        mats.append(fz.rows(s["state"], s["names"]))
    pred = model.predict(sparse.vstack(mats, format="csr"))
    tops, start = [], 0
    for s in screens:
        n = len(s["names"])
        tops.append(int(np.argmax(pred[start:start + n])))
        start += n
    missing = Counter(n for s in screens for n in s["names"]
                      if n is not None and n not in fz.card_idx)
    return tops, missing


def pct(v):
    return f"{100 * v:.1f}%"


def show(t, label, labels, top):
    t = t.copy()
    t["diff"] = t[label] - t["players"]
    blocks = (("taken more often by the model than by players",
               t.sort_values("diff", ascending=False).head(top)),
              ("taken less often by the model than by players", t.sort_values("diff").head(top)))
    for title, part in blocks:
        print(f"\n{label}: {title}")
        print(f"  {'card':<24}{'offered':>8}{'players':>9}"
              + "".join(f"{l:>{len(l) + 2}}" for l in labels))
        for card, r in part.iterrows():
            print(f"  {card:<24}{int(r['offered']):>8}{pct(r['players']):>9}"
                  + "".join(f"{pct(r[l]):>{len(l) + 2}}" for l in labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--models", nargs="+",
                    default=["models/value_model.json", "models/sim_value_model.json"])
    ap.add_argument("--max-screens", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-offered", type=int, default=150,
                    help="leave out cards offered fewer times than this")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    screens = load_screens(args.csv, args.max_screens, args.seed)
    labels = [Path(m).stem for m in args.models]
    tops, missing = {}, {}
    for label, m in zip(labels, args.models):
        tops[label], missing[label] = top_choices(m, screens)

    picked = np.array([s["picked"] for s in screens])
    skip = np.array([s["skip"] for s in screens])
    print(f"{len(screens)} Ironclad screens from {args.csv}\n")
    print(f"{'':<24}{'agrees with players':<22}{'skip rate'}")
    print(f"{'players':<24}{'-':<22}{(picked == skip).mean():.3f}")
    for label in labels:
        t = np.array(tops[label])
        print(f"{label:<24}{(t == picked).mean():<22.3f}{(t == skip).mean():.3f}")
    if len(labels) == 2:
        same = (np.array(tops[labels[0]]) == np.array(tops[labels[1]])).mean()
        print(f"\nthe two models choose the same option on {same:.3f} of screens")
    for label in labels:
        if missing[label]:
            top = ", ".join(f"{n} x{c}" for n, c in missing[label].most_common(8))
            print(f"offered cards missing from {label}'s vocabulary: {top}")

    cards, players = [], []
    flags = {label: [] for label in labels}
    for k, s in enumerate(screens):
        for j, name in enumerate(s["names"]):
            if name is None:
                continue
            cards.append(name)
            players.append(int(j == s["picked"]))
            for label in labels:
                flags[label].append(int(j == tops[label][k]))
    df = pd.DataFrame({"card": cards, "players": players, **flags})
    table = df.groupby("card").agg(offered=("players", "size"),
                                   **{c: (c, "mean") for c in ["players"] + labels})
    table = table[table["offered"] >= args.min_offered]
    if table.empty:
        raise SystemExit("no card was offered often enough; lower --min-offered")
    for label in labels:
        show(table, label, labels, args.top)


if __name__ == "__main__":
    main()
