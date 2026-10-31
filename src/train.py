"""Compare card-pick models with cross-validation grouped by run.

Run from the project root:  python3 src/train.py
Input: data/processed/card_choices.csv (made by src/parse_runs.py)

Every screen (one card reward) is scored by picking the highest-scoring row among
its candidates (the offered cards plus SKIP). "top-1" is how often that row is
the one you actually picked.
"""
import argparse
import json
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRanker, XGBRegressor

CSV = "data/processed/card_choices.csv"
NUMERIC = ["hp", "max_hp", "hp_ratio", "floor", "act", "ascension",
           "deck_size", "n_offered", "cand_in_deck", "is_skip"]
XGB = dict(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.8, min_child_weight=2, verbosity=0)


def build_features(raw):
    df = raw.copy()
    df["n_offered"] = df.groupby("screen_id")["candidate"].transform("count")
    decks = df["deck"].map(json.loads)
    df["cand_in_deck"] = [d.get(c, 0) for d, c in zip(decks, df["candidate"])]
    deck_cols = pd.DataFrame(decks.tolist(), index=df.index).fillna(0).add_prefix("deck:")
    cand_cols = pd.get_dummies(df["candidate"], prefix="cand").astype(float)
    char_cols = pd.get_dummies(df["character"], prefix="char").astype(float)
    X = pd.concat([df[NUMERIC].astype(float).fillna(0), deck_cols, cand_cols, char_cols], axis=1)
    X.columns = [re.sub(r"[\[\]<>]", "", c) for c in X.columns]  # xgboost dislikes these
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


def run_fold(tr, Xtr, te, Xte, seed):
    res = {}
    res["random (expected)"] = random_baseline(te)
    res["always skip"] = evaluate(te, te["is_skip"], None)

    rates = tr.groupby("candidate")["picked"].mean()
    res["popularity"] = evaluate(te, te["candidate"].map(rates).fillna(tr["picked"].mean()), "prob")

    logreg = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    logreg.fit(Xtr, tr["picked"])
    res["logistic regression"] = evaluate(te, logreg.predict_proba(Xte)[:, 1], "prob")

    clf = XGBClassifier(**XGB, random_state=seed)
    clf.fit(Xtr, tr["picked"])
    res["xgb classifier"] = evaluate(te, clf.predict_proba(Xte)[:, 1], "prob")

    # Ranker: rows must be grouped by screen, with one query id per screen.
    order = np.argsort(tr["screen_id"].to_numpy(), kind="stable")
    qid = pd.factorize(tr["screen_id"].to_numpy()[order])[0]
    ranker = XGBRanker(**XGB, objective="rank:pairwise", random_state=seed)
    ranker.fit(Xtr.iloc[order], tr["picked"].to_numpy()[order], qid=qid)
    res["xgb ranker"] = evaluate(te, ranker.predict(Xte), "margin")

    # Regressor: learn floors survived after the pick, from the rows that were really
    # picked (SKIP included), then score every candidate and take the best.
    m = tr["picked"].to_numpy() == 1
    reg = XGBRegressor(**XGB, random_state=seed)
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
    args = ap.parse_args()

    df, X = build_features(pd.read_csv(args.csv))
    print(f"{df['screen_id'].nunique()} screens from {df['run_id'].nunique()} runs, "
          f"{X.shape[1]} features, {args.folds}-fold CV grouped by run\n")

    folds = []
    splitter = GroupKFold(n_splits=args.folds)
    for tr_idx, te_idx in splitter.split(X, groups=df["run_id"]):
        folds.append(run_fold(df.iloc[tr_idx], X.iloc[tr_idx],
                              df.iloc[te_idx], X.iloc[te_idx], args.seed))

    print(f"{'model':<22}{'top-1 accuracy':<20}screen log loss")
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
