"""Value model: predict how many floors a run survives from its state, then score each
offered card by the state you would be in after taking it (SKIP included).

Run from the project root (after python3 src/parse_big.py):
    python3 src/train_value.py --sample-runs 3000
    python3 src/train_value.py
    python3 src/train_value.py --save models/value_model.json   (also saves a final model)

Training uses the state after the pick that was really made, with the outcome of that run
(floors survived after the pick) as the target. Cross-validation is grouped by run.
Player skill confounds all of this: strong players both pick better and survive longer,
so the player_experience column is a feature to soak up part of that.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

CSV = "data/processed/card_choices_big.csv"
NEEDED = {"relics", "upgrades", "player_experience"}
PARAMS = dict(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.6, min_child_weight=20, reg_lambda=5,
              tree_method="hist", n_jobs=-1, verbosity=0)


def one_hot(values):
    codes, uniques = pd.factorize(values)
    n = len(codes)
    return sparse.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), codes)),
                             shape=(n, len(uniques)))


def build_state_features(raw):
    """One row per (screen, candidate): features of the state AFTER taking that candidate."""
    df = raw.reset_index(drop=True).copy()
    codes, _ = pd.factorize(df["screen_id"])  # screens numbered by first appearance
    per_screen = df.loc[~df["screen_id"].duplicated()]
    decks = per_screen["deck"].map(json.loads).tolist()
    relics = [dict.fromkeys(json.loads(s), 1) for s in per_screen["relics"]]
    upgrades = per_screen["upgrades"].map(json.loads).tolist()

    # Deck counts after adding the candidate (nothing is added for SKIP).
    cands = sorted(set(df["candidate"]) - {"SKIP"})
    dv = DictVectorizer(dtype=np.float32).fit(decks + [{c: 1} for c in cands])
    cols = df["candidate"].map(dv.vocabulary_)
    is_card = cols.notna().to_numpy()
    add = sparse.csr_matrix(
        (np.ones(is_card.sum(), dtype=np.float32),
         (np.flatnonzero(is_card), cols[is_card].astype(int).to_numpy())),
        shape=(len(df), len(dv.vocabulary_)))
    deck_after = dv.transform(decks).tocsr()[codes] + add

    relic_m = DictVectorizer(dtype=np.float32).fit_transform(relics).tocsr()[codes]
    upgrade_m = DictVectorizer(dtype=np.float32).fit_transform(upgrades).tocsr()[codes]
    n_relics = np.array([len(r) for r in relics])[codes]
    n_upgrades = np.array([sum(u.values()) for u in upgrades])[codes]

    num = np.column_stack([
        df["hp"].astype(float).fillna(0), df["max_hp"].astype(float).fillna(0),
        df["hp_ratio"].astype(float).fillna(0), df["floor"].astype(float),
        df["act"].astype(float), df["ascension"].astype(float),
        df["deck_size"].astype(float) + is_card, n_relics, n_upgrades,
        np.log1p(df["player_experience"].astype(float).fillna(0).clip(lower=0)),
    ]).astype(np.float32)
    X = sparse.hstack([sparse.csr_matrix(num), deck_after, relic_m, upgrade_m,
                       one_hot(df["character"])], format="csr")
    return df, X


def screen_table(te, score, base, base_all):
    """One row per held-out screen: how the model's choice compares with the real pick."""
    t = pd.DataFrame({"run_id": te["run_id"].to_numpy(), "screen": te["screen_id"].to_numpy(),
                      "floor": te["floor"].to_numpy(), "picked": te["picked"].to_numpy(),
                      "skip": te["is_skip"].to_numpy(), "score": np.asarray(score, dtype=float),
                      "y": te["floors_after"].to_numpy()})
    t["rank"] = t.groupby("screen")["score"].rank(ascending=False, method="first")
    best = t.loc[t.groupby("screen")["score"].idxmax()].set_index("screen")
    pick = t[t["picked"] == 1].set_index("screen")
    out = pd.DataFrame({"run_id": pick["run_id"], "floor": pick["floor"], "y": pick["y"],
                        "pred_picked": pick["score"], "pick_rank": pick["rank"],
                        "pred_best": best["score"], "agree": best["picked"],
                        "model_skip": best["skip"], "player_skip": pick["skip"]})
    out["resid"] = out["y"] - out["floor"].map(base).fillna(base_all)
    return out


def agree_gap(oof, n_boot=300, seed=0):
    """Mean outcome (vs typical for that floor) when the player took the model's top card,
    minus the same when they did not. Confidence interval resamples whole runs."""
    a = oof["agree"].to_numpy().astype(float)
    g = pd.DataFrame({"run_id": oof["run_id"].to_numpy(), "n": 1.0, "na": a,
                      "sa": oof["resid"].to_numpy() * a,
                      "sd": oof["resid"].to_numpy() * (1 - a)}).groupby("run_id").sum()
    n, na, sa, sd = (g[c].to_numpy() for c in ("n", "na", "sa", "sd"))

    def gap(idx):
        den_a, den_d = na[idx].sum(), (n[idx] - na[idx]).sum()
        if den_a == 0 or den_d == 0:
            return np.nan
        return sa[idx].sum() / den_a - sd[idx].sum() / den_d

    point = gap(np.arange(len(g)))
    rng = np.random.default_rng(seed)
    boots = [gap(rng.integers(0, len(g), len(g))) for _ in range(n_boot)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return point, lo, hi


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-runs", type=int, default=0,
                    help="use a random subset of this many runs (0 = all)")
    ap.add_argument("--save", default="", help="also fit on all data and save the model here")
    args = ap.parse_args()

    raw = pd.read_csv(args.csv)
    missing = NEEDED - set(raw.columns)
    if missing:
        raise SystemExit(f"{args.csv} lacks {sorted(missing)}; rerun python3 src/parse_big.py")
    if args.sample_runs:
        runs = raw["run_id"].unique()
        if len(runs) > args.sample_runs:
            keep = np.random.default_rng(args.seed).choice(runs, args.sample_runs, replace=False)
            raw = raw[raw["run_id"].isin(keep)]
    df, X = build_state_features(raw)
    counts = df.groupby("screen_id").size()
    print(f"{len(counts)} screens from {df['run_id'].nunique()} runs, {X.shape[1]} features, "
          f"{args.folds}-fold CV grouped by run\n")

    tables = []
    for k, (tr_idx, te_idx) in enumerate(
            GroupKFold(n_splits=args.folds).split(X, groups=df["run_id"]), 1):
        start = time.time()
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        m = tr["picked"].to_numpy() == 1
        reg = XGBRegressor(**PARAMS, random_state=args.seed)
        reg.fit(X[tr_idx][m], tr["floors_after"].to_numpy()[m])
        base = tr[m].groupby("floor")["floors_after"].mean()
        tables.append(screen_table(te, reg.predict(X[te_idx]), base,
                                   tr.loc[m, "floors_after"].mean()))
        print(f"fold {k}/{args.folds} done in {time.time() - start:.0f}s")

    oof = pd.concat(tables)
    rmse = float(np.sqrt(np.mean((oof["pred_picked"] - oof["y"]) ** 2)))
    rmse_base = float(np.sqrt(np.mean(oof["resid"] ** 2)))
    print(f"\nvalue fit, floors survived after the pick: RMSE {rmse:.2f} "
          f"(predicting the average for that floor: {rmse_base:.2f})")
    folds_agree = [t["agree"].mean() for t in tables]
    print(f"agreement with the players' pick: {np.mean(folds_agree):.3f} "
          f"+- {np.std(folds_agree):.3f} (chance {(1 / counts).mean():.3f})")
    print(f"skip rate: players {oof['player_skip'].mean():.3f}, model {oof['model_skip'].mean():.3f}")

    print("\nwhere the player's pick ranks in the model's list, and how those runs went")
    print("(outcome vs typical for that floor, in floors; screens from one run are correlated)")
    oof["bucket"] = np.where(oof["pick_rank"] == 1, "1 (model's top)",
                             np.where(oof["pick_rank"] == 2, "2", "3 or lower"))
    for name, g in oof.groupby("bucket"):
        print(f"  rank {name:<16} n={len(g):<7} {g['resid'].mean():+.3f}")

    point, lo, hi = agree_gap(oof)
    print(f"\npick = model's top vs not: {point:+.3f} floors (95% interval {lo:+.3f} to {hi:+.3f})")
    dis = oof[oof["agree"] == 0]
    print(f"when they differ, the model's top card is predicted "
          f"{(dis['pred_best'] - dis['pred_picked']).mean():.2f} floors better than the player's pick")

    if args.save:
        m = df["picked"].to_numpy() == 1
        final = XGBRegressor(**PARAMS, random_state=args.seed)
        final.fit(X[m], df["floors_after"].to_numpy()[m])
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        final.save_model(args.save)
        print(f"\nsaved final model to {args.save}")


if __name__ == "__main__":
    main()
