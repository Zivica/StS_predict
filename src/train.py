"""Compare card-pick models with cross-validation grouped by run.

Run from the project root:
    python3 src/train.py                                  (your own runs)
    python3 src/train.py --csv data/processed/card_choices_big.csv --sample-runs 5000

Every screen (one card reward) is scored by picking the highest-scoring row among
its candidates (the offered cards plus SKIP). "top-1" is how often that row is
the one that was actually picked.
"""
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRanker, XGBRegressor

CSV = "data/processed/card_choices.csv"
NUMERIC = ["hp", "max_hp", "hp_ratio", "floor", "act", "ascension",
           "deck_size", "n_offered", "cand_in_deck", "is_skip"]


def xgb_params(n_screens):
    """Small, shallow models for tiny data; bigger ones once there is plenty of it."""
    if n_screens < 20000:
        return dict(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, min_child_weight=2, verbosity=0)
    return dict(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                colsample_bytree=0.8, min_child_weight=5, tree_method="hist",
                n_jobs=-1, verbosity=0)


def one_hot(values):
    codes, uniques = pd.factorize(values)
    n = len(codes)
    return sparse.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), codes)),
                             shape=(n, len(uniques)))


def build_features(raw):
    """Sparse feature matrix: numbers, deck card counts, candidate card, character."""
    df = raw.reset_index(drop=True).copy()
    df["n_offered"] = df.groupby("screen_id")["candidate"].transform("count")
    codes, _ = pd.factorize(df["screen_id"])  # screens numbered by first appearance
    first = ~df["screen_id"].duplicated()
    decks = df.loc[first, "deck"].map(json.loads).tolist()  # one deck per screen
    df["cand_in_deck"] = [decks[c].get(k, 0) for c, k in zip(codes, df["candidate"])]
    deck_by_screen = DictVectorizer(dtype=np.float32).fit_transform(decks).tocsr()
    num = sparse.csr_matrix(df[NUMERIC].astype(float).fillna(0).to_numpy(dtype=np.float32))
    X = sparse.hstack([num, deck_by_screen[codes], one_hot(df["candidate"]),
                       one_hot(df["character"])], format="csr")
    return df, X


def evaluate(te, score, kind):
    """kind: 'prob' (scores are probabilities), 'margin' (raw scores), None (no log loss)."""
    t = pd.DataFrame({"screen": te["screen_id"].to_numpy(),
                      "picked": te["picked"].to_numpy(),
                      "score": np.asarray(score, dtype=float)})
    best = t.loc[t.groupby("screen")["score"].idxmax()]
    out = {"top1": float(best["picked"].mean()), "logloss": np.nan}
    if kind == "prob":
        p = t["score"].clip(lower=1e-9)
    elif kind == "margin":
        p = np.exp(t["score"] - t.groupby("screen")["score"].transform("max"))
    else:
        return out
    p = p / p.groupby(t["screen"]).transform("sum")
    out["logloss"] = float(-np.log(p[t["picked"] == 1].clip(lower=1e-9)).mean())
    return out


def random_baseline(te):
    n = te.groupby("screen_id")["candidate"].count()
    return {"top1": float((1 / n).mean()), "logloss": float(np.log(n).mean())}


def run_fold(tr, Xtr, te, Xte, seed, params):
    res = {}
    res["random (expected)"] = random_baseline(te)
    res["always skip"] = evaluate(te, te["is_skip"], None)

    # Popularity, smoothed so cards never seen in training do not get probability 0.
    stats = tr.groupby("candidate")["picked"].agg(["sum", "count"])
    prior = tr["picked"].mean()
    rates = (stats["sum"] + 5 * prior) / (stats["count"] + 5)
    res["popularity"] = evaluate(te, te["candidate"].map(rates).fillna(prior), "prob")

    logreg = make_pipeline(StandardScaler(with_mean=False),
                           LogisticRegression(C=0.1, max_iter=1000))
    logreg.fit(Xtr, tr["picked"])
    res["logistic regression"] = evaluate(te, logreg.predict_proba(Xte)[:, 1], "prob")

    clf = XGBClassifier(**params, random_state=seed)
    clf.fit(Xtr, tr["picked"])
    res["xgb classifier"] = evaluate(te, clf.predict_proba(Xte)[:, 1], "prob")

    # Ranker: rows must be grouped by screen, with one query id per screen.
    order = np.argsort(tr["screen_id"].to_numpy(), kind="stable")
    qid = pd.factorize(tr["screen_id"].to_numpy()[order])[0]
    ranker = XGBRanker(**params, objective="rank:pairwise", random_state=seed)
    ranker.fit(Xtr[order], tr["picked"].to_numpy()[order], qid=qid)
    res["xgb ranker"] = evaluate(te, ranker.predict(Xte), "margin")

    # Regressor: learn floors survived after the pick, from the rows that were really
    # picked (SKIP included), then score every candidate and take the best.
    m = tr["picked"].to_numpy() == 1
    reg = XGBRegressor(**params, random_state=seed)
    reg.fit(Xtr[m], tr["floors_after"].to_numpy()[m])
    pred = reg.predict(Xte)
    res["xgb regressor"] = evaluate(te, pred, None)
    tm = te["picked"].to_numpy() == 1
    y = te["floors_after"].to_numpy()[tm]
    res["xgb regressor"]["rmse"] = float(np.sqrt(np.mean((pred[tm] - y) ** 2)))
    res["xgb regressor"]["rmse_mean"] = float(
        np.sqrt(np.mean((tr["floors_after"].to_numpy()[m].mean() - y) ** 2)))
    return res


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-runs", type=int, default=0,
                    help="use a random subset of this many runs (0 = all); handy for quick tests")
    args = ap.parse_args()

    raw = pd.read_csv(args.csv)
    if args.sample_runs:
        runs = raw["run_id"].unique()
        if len(runs) > args.sample_runs:
            keep = np.random.default_rng(args.seed).choice(runs, args.sample_runs, replace=False)
            raw = raw[raw["run_id"].isin(keep)]
    df, X = build_features(raw)
    n_screens = df["screen_id"].nunique()
    params = xgb_params(n_screens)
    print(f"{n_screens} screens from {df['run_id'].nunique()} runs, {X.shape[1]} features, "
          f"{args.folds}-fold CV grouped by run, xgboost depth {params['max_depth']} "
          f"x {params['n_estimators']} trees\n")

    folds = []
    splitter = GroupKFold(n_splits=args.folds)
    for k, (tr_idx, te_idx) in enumerate(splitter.split(X, groups=df["run_id"]), 1):
        start = time.time()
        folds.append(run_fold(df.iloc[tr_idx], X[tr_idx], df.iloc[te_idx], X[te_idx],
                              args.seed, params))
        print(f"fold {k}/{args.folds} done in {time.time() - start:.0f}s")

    print(f"\n{'model':<22}{'top-1 accuracy':<20}screen log loss")
    for name in folds[0]:
        top1 = [f[name]["top1"] for f in folds]
        ll = [f[name]["logloss"] for f in folds]
        ll_txt = "-" if np.all(np.isnan(ll)) else f"{np.nanmean(ll):.3f}"
        print(f"{name:<22}{np.mean(top1):.3f} +- {np.std(top1):.3f}     {ll_txt}")

    rmse = np.mean([f["xgb regressor"]["rmse"] for f in folds])
    rmse_mean = np.mean([f["xgb regressor"]["rmse_mean"] for f in folds])
    print(f"\nregressor RMSE on floors survived after the pick: {rmse:.2f} "
          f"(predicting the average: {rmse_mean:.2f})")


if __name__ == "__main__":
    main()
