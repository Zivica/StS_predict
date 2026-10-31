"""Evaluate card-reward policies by playing full Ironclad runs in sts_lightspeed.

Every policy plays the same seeds. The simulator's built-in combat search plays all fights and
pauses at each card reward, so a policy can choose a card or skip.

    python3 src/sim_eval.py --sim-build /path/to/build --runs 500
    python3 src/sim_eval.py --sim-build /path/to/build --runs 2000 --sims 1000 --workers 8

The value model and popularity policies need the files written by
    python3 src/train_value.py --save models/value_model.json
Without them only the built-in, always-skip and random policies run.

Caveats: the models were trained on human runs, and the combat bot plays differently from a
human, so a good result here shows the model works as a card picker for this bot.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import sparse

# Must match NUMERIC_COLUMNS in train_value.py (the first columns of every feature row).
NUMERIC = ["hp", "max_hp", "hp_ratio", "floor", "act", "ascension",
           "deck_size_after", "n_relics", "n_upgrades", "log_player_experience"]
STARTER_RELIC = "BURNING_BLOOD"  # in the simulator's relic list, not in the training data
# Simulator names that do not match the run files after dropping case, spaces and underscores.
ALIASES = {"strikered": "Strike_R", "defendred": "Defend_R", "strikegreen": "Strike_G",
           "defendgreen": "Defend_G", "strikeblue": "Strike_B", "defendblue": "Defend_B",
           "strikepurple": "Strike_P", "defendpurple": "Defend_P", "tranquility": "ClearTheMind",
           "apparition": "Ghostly", "recursion": "Redo", "steambarrier": "Steam",
           "overclock": "Steam Power"}
RELIC_ALIASES = {"theboot": "Boot", "paperphrog": "Paper Frog", "wingboots": "WingedGreaves"}
ALL_POLICIES = ["built-in", "always skip", "random", "popularity", "value model"]
EXTRA_POLICIES = ["random no skip"]  # only runs when named in --policies


def norm(name):
    """'SHRUG_IT_OFF' and 'Shrug It Off' both become 'shrugitoff' (digits dropped too)."""
    return "".join(ch for ch in name.lower() if ch.isalpha())


def act_of(floor):
    return 1 if floor <= 16 else 2 if floor <= 33 else 3 if floor <= 50 else 4


class Featurizer:
    """Builds feature rows in exactly the layout train_value.py used for training."""

    def __init__(self, meta, ascension):
        if meta["numeric"] != NUMERIC:
            raise SystemExit("meta file has a different numeric layout than sim_eval.py expects")
        self.ascension = ascension
        self.card_idx = {n: i for i, n in enumerate(meta["deck_vocab"])}
        self.relic_idx = {n: i for i, n in enumerate(meta["relic_vocab"])}
        self.up_idx = {n: i for i, n in enumerate(meta["upgrade_vocab"])}
        self.card_by_norm = {norm(n): n for n in meta["deck_vocab"]}
        self.relic_by_norm = {norm(n): n for n in meta["relic_vocab"]}
        self.deck_off = len(NUMERIC)
        self.relic_off = self.deck_off + len(self.card_idx)
        self.up_off = self.relic_off + len(self.relic_idx)
        self.char_off = self.up_off + len(self.up_idx)
        self.width = self.char_off + len(meta["chars"])
        if "IRONCLAD" not in meta["chars"]:
            raise SystemExit("the model was not trained on Ironclad runs")
        self.char_col = meta["chars"].index("IRONCLAD")
        self.log_exp = math.log1p(meta["experience_hi"])
        self.unmapped = Counter()

    def card_name(self, card):
        n = card.id.name
        key = norm(n)
        if key in ALIASES:
            return ALIASES[key]
        if key in self.card_by_norm:
            return self.card_by_norm[key]
        self.unmapped["card " + n] += 1
        return n

    def relic_name(self, relic):
        n = relic.id.name
        key = norm(n)
        if key in RELIC_ALIASES:
            return RELIC_ALIASES[key]
        if key in self.relic_by_norm:
            return self.relic_by_norm[key]
        self.unmapped["relic " + n] += 1
        return n

    def state(self, gc):
        deck, ups = Counter(), Counter()
        for c in gc.deck:
            name = self.card_name(c)
            deck[name] += 1
            if c.upgraded:
                ups[name] += 1
        relics = sorted({self.relic_name(r) for r in gc.relics if r.id.name != STARTER_RELIC})
        return {"deck": deck, "ups": ups, "relics": relics, "hp": gc.cur_hp,
                "max_hp": gc.max_hp, "floor": gc.floor_num}

    def rows(self, st, names):
        """One feature row per candidate in names (a card name, or None for SKIP): the state
        after taking it. Zeros are left out, like in the training matrix (XGBoost treats a
        missing entry in sparse input differently from a stored zero)."""
        deck_size = sum(st["deck"].values())
        n_up = sum(st["ups"].values())
        hp, mhp, floor = st["hp"], st["max_hp"], st["floor"]
        head = [hp, mhp, round(hp / mhp, 3) if mhp else 0.0, floor, act_of(floor), self.ascension]
        r_i, c_i, vals = [], [], []

        def add(r, c, v):
            if v != 0:
                r_i.append(r)
                c_i.append(c)
                vals.append(float(v))

        for r, cand in enumerate(names):
            tail = [deck_size + (cand is not None), len(st["relics"]), n_up, self.log_exp]
            for j, v in enumerate(head + tail):
                add(r, j, v)
            counts = Counter(st["deck"])
            if cand is not None:
                counts[cand] += 1
            for name, k in counts.items():
                if name in self.card_idx:
                    add(r, self.deck_off + self.card_idx[name], k)
            for name in st["relics"]:
                if name in self.relic_idx:
                    add(r, self.relic_off + self.relic_idx[name], 1)
            for name, k in st["ups"].items():
                if name in self.up_idx:
                    add(r, self.up_off + self.up_idx[name], k)
            add(r, self.char_off + self.char_col, 1)
        return sparse.csr_matrix((np.array(vals, dtype=np.float32), (r_i, c_i)),
                                 shape=(len(names), self.width), dtype=np.float32)


_G = {}


def init_worker(cfg):
    sys.path.insert(0, cfg["sim_build"])
    import slaythespire as sts
    _G["sts"], _G["cfg"] = sts, cfg
    if cfg["model_path"]:
        from xgboost import XGBRegressor
        _G["meta"] = json.loads(Path(cfg["meta_path"]).read_text())
        _G["fz"] = Featurizer(_G["meta"], cfg["ascension"])
        model = XGBRegressor(n_jobs=1)
        model.load_model(cfg["model_path"])
        _G["model"] = model


def make_picker(policy, rng):
    """Returns picker(gc, cards) -> index into cards, or -1 to skip. None = built-in weights."""
    if policy == "built-in":
        return None
    if policy == "always skip":
        return lambda gc, cards: -1
    if policy == "random":
        return lambda gc, cards: rng.choice([-1] + list(range(len(cards))))
    if policy == "random no skip":
        return lambda gc, cards: rng.randrange(len(cards))
    fz, meta = _G["fz"], _G["meta"]
    if policy == "popularity":
        rates, prior = meta["popularity"], meta["prior"]

        def pick_popular(gc, cards):
            scores = [rates.get(fz.card_name(c), prior) for c in cards] + [rates.get("SKIP", prior)]
            best = int(np.argmax(scores))
            return -1 if best == len(cards) else best
        return pick_popular
    if policy == "value model":
        model = _G["model"]

        def pick_value(gc, cards):
            names = [fz.card_name(c) for c in cards] + [None]
            best = int(np.argmax(model.predict(fz.rows(fz.state(gc), names))))
            return -1 if best == len(cards) else best
        return pick_value
    raise ValueError(policy)


def play(seed, policy):
    sts, cfg = _G["sts"], _G["cfg"]
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, cfg["ascension"])
    agent = sts.Agent()
    agent.simulation_count_base = cfg["sims"]
    agent.print_logs = False
    picker = make_picker(policy, random.Random(seed * 1000 + 7))
    agent.pause_on_card_reward = picker is not None
    screens = skips = 0
    for _ in range(500):
        agent.playout(gc)
        if gc.outcome != sts.GameOutcome.UNDECIDED:
            break
        cards = gc.get_card_reward()
        if not cards:
            break
        i = picker(gc, cards)
        screens += 1
        if i < 0:
            gc.skip_reward_cards()
            skips += 1
        else:
            gc.pick_reward_card(cards[i])
    return {"floor": gc.floor_num, "win": gc.outcome == sts.GameOutcome.PLAYER_VICTORY,
            "finished": gc.outcome != sts.GameOutcome.UNDECIDED, "screens": screens,
            "skips": skips, "deck": len(gc.deck)}


def run_seed(seed):
    out = {p: play(seed, p) for p in _G["cfg"]["policies"]}
    unmapped = dict(_G["fz"].unmapped) if "fz" in _G else {}
    return seed, out, unmapped


def paired_gap(a, b, n_boot=2000, seed=0):
    """Mean of a - b over seeds, with a 95% bootstrap interval (seeds are resampled)."""
    d = a - b
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return d.mean(), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-build", required=True,
                    help="folder containing the compiled slaythespire module")
    ap.add_argument("--runs", type=int, default=200, help="seeds per policy")
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--sims", type=int, default=1000, help="combat search simulations per turn")
    ap.add_argument("--ascension", type=int, default=5,
                    help="the bot is weak at A15 and dies in act 1; the models were trained on A15+")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--model", default="models/value_model.json")
    ap.add_argument("--policies", nargs="+", default=ALL_POLICIES, choices=ALL_POLICIES + EXTRA_POLICIES)
    ap.add_argument("--ref", default="built-in", help="policy the others are compared with")
    ap.add_argument("--out", default="", help="optional CSV with one row per run and policy")
    args = ap.parse_args()

    model_path = Path(args.model)
    meta_path = model_path.with_name(model_path.stem + ".meta.json")
    have_model = model_path.exists() and meta_path.exists()
    policies = list(args.policies)
    if not have_model:
        dropped = [p for p in policies if p in ("popularity", "value model")]
        if dropped:
            print(f"no model files at {model_path} / {meta_path}: skipping {dropped}")
        policies = [p for p in policies if p not in dropped]
    if args.ref not in policies:
        args.ref = policies[0]

    cfg = {"sim_build": str(Path(args.sim_build).resolve()), "sims": args.sims,
           "ascension": args.ascension, "policies": policies,
           "model_path": str(model_path) if have_model else "",
           "meta_path": str(meta_path) if have_model else ""}
    seeds = list(range(args.seed_start, args.seed_start + args.runs))
    print(f"{len(seeds)} seeds x {len(policies)} policies, Ironclad A{args.ascension}, "
          f"{args.sims} combat simulations per turn, {args.workers} workers")

    start = time.time()
    results, unmapped = {}, Counter()
    with Pool(args.workers, initializer=init_worker, initargs=(cfg,)) as pool:
        for k, (seed, out, um) in enumerate(pool.imap_unordered(run_seed, seeds), 1):
            results[seed] = out
            for n, c in um.items():
                unmapped[n] = max(unmapped[n], c)
            if k % max(1, len(seeds) // 10) == 0:
                print(f"  {k}/{len(seeds)} seeds done, {time.time() - start:.0f}s")

    order = sorted(results)
    floors = {p: np.array([results[s][p]["floor"] for s in order], dtype=float) for p in policies}
    print(f"\n{'policy':<14}{'mean floor':<18}{'wins':<12}{'skip rate':<11}{'final deck'}")
    for p in policies:
        rs = [results[s][p] for s in order]
        se = floors[p].std(ddof=1) / math.sqrt(len(rs)) if len(rs) > 1 else float("nan")
        wins = sum(r["win"] for r in rs)
        screens = sum(r["screens"] for r in rs)
        skip = f"{sum(r['skips'] for r in rs) / screens:.2f}" if screens else "-"
        print(f"{p:<14}{floors[p].mean():>5.1f} +- {se:<8.1f}{wins:>4} ({100 * wins / len(rs):.1f}%)  "
              f"{skip:<11}{np.mean([r['deck'] for r in rs]):.1f}")
    unfinished = sum(not results[s][p]["finished"] for s in order for p in policies)
    if unfinished:
        print(f"\nwarning: {unfinished} runs did not finish (their floor is where they stopped)")

    print(f"\nmean floor difference vs '{args.ref}' (same seeds, 95% interval over seeds)")
    for p in policies:
        if p != args.ref:
            m, lo, hi = paired_gap(floors[p], floors[args.ref])
            print(f"  {p:<14}{m:+6.2f}  ({lo:+.2f} to {hi:+.2f})")
    if "value model" in policies and "popularity" in policies:
        m, lo, hi = paired_gap(floors["value model"], floors["popularity"])
        print(f"  value model vs popularity: {m:+.2f}  ({lo:+.2f} to {hi:+.2f})")

    if unmapped:
        top = ", ".join(f"{n} x{c}" for n, c in unmapped.most_common(12))
        print(f"\nnames the model does not know (ignored in its features): {top}")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seed", "policy", "floor", "win", "finished", "screens", "skips", "deck"])
            for s in order:
                for p in policies:
                    r = results[s][p]
                    w.writerow([s, p, r["floor"], int(r["win"]), int(r["finished"]),
                                r["screens"], r["skips"], r["deck"]])
        print(f"\nper-run results written to {args.out}")


if __name__ == "__main__":
    main()
