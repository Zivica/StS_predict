"""Collect card-reward data from simulator runs, to train a value model on data that has no
player skill in it.

Plays full Ironclad runs in sts_lightspeed. At every card reward the pick is random (each
offered card and skip equally likely), or, with --epsilon below 1, the value model's top
choice the rest of the time. The output has one row per offered card plus a SKIP row per
screen, in the same columns as data/processed/card_choices_big.csv, so train_value.py reads
it unchanged:

    python3 src/sim_collect.py --sim-build /path/to/build --runs 50 --out /tmp/sim_test.csv
    python3 src/train_value.py --csv /tmp/sim_test.csv
    python3 src/sim_collect.py --sim-build /path/to/build --runs 20000
    python3 src/train_value.py --csv data/processed/sim_choices.csv --save models/sim_value_model.json

Notes:
- Seeds start at 100000 by default, away from the seeds sim_eval.py tests on, so the model
  is never trained on a seed it is evaluated on.
- player_experience is 0 in every row: there is no player skill in this data.
- Runs that did not finish are dropped (their floor_reached would be wrong).
- The value model learns what is good for the simulator's combat bot when later picks are
  random (or mostly random), not what is good for a human.
- Card and relic names use the spelling of the model's meta file where they exist there, so
  the data matches the human data. The meta file is the only part of --model used when
  --epsilon is 1.
"""
import argparse
import csv
import json
import os
import random
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import sim_eval as se

COLUMNS = ["screen_id", "run_id", "character", "ascension", "floor", "act", "hp", "max_hp",
           "hp_ratio", "deck_size", "deck", "candidate", "is_skip", "picked", "victory",
           "floor_reached", "floors_after", "relics", "upgrades", "player_experience"]


def make_chooser(seed, epsilon):
    rng = random.Random(seed * 1000 + 11)
    pick_random = se.make_picker("random", rng)
    pick_model = se.make_picker("value model", rng) if epsilon < 1 else None

    def choose(gc, cards):
        if pick_model is None or rng.random() < epsilon:
            return pick_random(gc, cards)
        return pick_model(gc, cards)
    return choose


def collect(seed):
    sts, cfg, fz = se._G["sts"], se._G["cfg"], se._G["fz"]
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, cfg["ascension"])
    agent = sts.Agent()
    agent.simulation_count_base = cfg["sims"]
    agent.print_logs = False
    agent.pause_on_card_reward = True
    choose = make_chooser(seed, cfg["epsilon"])
    screens = []  # (state before the pick, offered card names, index picked or -1 for skip)
    for _ in range(500):
        agent.playout(gc)
        if gc.outcome != sts.GameOutcome.UNDECIDED:
            break
        cards = gc.get_card_reward()
        if not cards:
            break
        st = fz.state(gc)
        names = [fz.card_name(c) for c in cards]
        i = choose(gc, cards)
        screens.append((st, names, i))
        if i < 0:
            gc.skip_reward_cards()
        else:
            gc.pick_reward_card(cards[i])

    finished = gc.outcome != sts.GameOutcome.UNDECIDED
    win = gc.outcome == sts.GameOutcome.PLAYER_VICTORY
    last_floor = gc.floor_num
    rows = []
    for k, (st, names, i) in enumerate(screens if finished else []):
        hp, mhp, floor = st["hp"], st["max_hp"], st["floor"]
        common = {
            "screen_id": f"{seed}_{k}", "run_id": seed, "character": "IRONCLAD",
            "ascension": cfg["ascension"], "floor": floor, "act": se.act_of(floor),
            "hp": hp, "max_hp": mhp, "hp_ratio": round(hp / mhp, 3) if mhp else 0.0,
            "deck_size": sum(st["deck"].values()),
            "deck": json.dumps(dict(st["deck"]), sort_keys=True),
            "victory": int(win), "floor_reached": last_floor,
            "floors_after": last_floor - floor,
            "relics": json.dumps(st["relics"]),
            "upgrades": json.dumps(dict(st["ups"]), sort_keys=True),
            "player_experience": 0}
        picked = len(names) if i < 0 else i
        for j, cand in enumerate(names + ["SKIP"]):
            rows.append({**common, "candidate": cand, "is_skip": int(j == len(names)),
                         "picked": int(j == picked)})
    return {"seed": seed, "finished": finished, "win": win, "floor": last_floor,
            "screens": len(screens), "skips": sum(1 for s in screens if s[2] < 0),
            "rows": rows, "unmapped": dict(fz.unmapped)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-build", required=True,
                    help="folder containing the compiled slaythespire module")
    ap.add_argument("--runs", type=int, default=20000)
    ap.add_argument("--seed-start", type=int, default=100000,
                    help="keep this away from the seeds sim_eval.py tests on")
    ap.add_argument("--sims", type=int, default=1000, help="combat search simulations per turn")
    ap.add_argument("--ascension", type=int, default=5)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--model", default="models/value_model.json",
                    help="its meta file gives the card names; the model itself plays when "
                         "--epsilon is below 1")
    ap.add_argument("--epsilon", type=float, default=1.0,
                    help="chance of a random pick on each screen; otherwise the model's top choice")
    ap.add_argument("--out", default="data/processed/sim_choices.csv")
    ap.add_argument("--overwrite", action="store_true", help="replace --out if it exists")
    args = ap.parse_args()

    if not 0.0 <= args.epsilon <= 1.0:
        raise SystemExit("--epsilon must be between 0 and 1")
    model_path = Path(args.model)
    meta_path = model_path.with_name(model_path.stem + ".meta.json")
    if not (model_path.exists() and meta_path.exists()):
        raise SystemExit(f"need {model_path} and {meta_path}; run train_value.py --save first")
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)

    cfg = {"sim_build": str(Path(args.sim_build).resolve()), "sims": args.sims,
           "ascension": args.ascension, "epsilon": args.epsilon,
           "model_path": str(model_path), "meta_path": str(meta_path)}
    seeds = list(range(args.seed_start, args.seed_start + args.runs))
    print(f"{len(seeds)} runs, Ironclad A{args.ascension}, {args.sims} combat simulations per "
          f"turn, {args.workers} workers, random pick chance {args.epsilon}")

    start = time.time()
    kept = dropped = screens = skips = wins = n_rows = 0
    floors = []
    unmapped = Counter()
    with open(out, "w", newline="") as f, \
            Pool(args.workers, initializer=se.init_worker, initargs=(cfg,)) as pool:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for k, r in enumerate(pool.imap_unordered(collect, seeds, chunksize=4), 1):
            if r["finished"]:
                writer.writerows(r["rows"])
                kept += 1
                screens += r["screens"]
                skips += r["skips"]
                wins += int(r["win"])
                n_rows += len(r["rows"])
                floors.append(r["floor"])
            else:
                dropped += 1
            for n, c in r["unmapped"].items():
                unmapped[n] = max(unmapped[n], c)
            if k % max(1, len(seeds) // 10) == 0:
                print(f"  {k}/{len(seeds)} runs done, {time.time() - start:.0f}s")

    print(f"\n{kept} runs written ({dropped} dropped, not finished): {screens} screens, "
          f"{n_rows} rows, to {out}")
    if kept and screens:
        print(f"mean floor {sum(floors) / kept:.1f}, wins {wins}, skip rate {skips / screens:.2f}")
    if unmapped:
        top = ", ".join(f"{n} x{c}" for n, c in unmapped.most_common(12))
        print(f"\nnames not in the meta file (written as the simulator spells them): {top}")


if __name__ == "__main__":
    main()
