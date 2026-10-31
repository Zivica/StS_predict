"""Parse the big Slay the Spire metrics dump into one row per (card reward screen, candidate).

The dump is one huge JSON array of {"event": <run>} objects. Runs are streamed, filtered,
and turned into rows with the same deck rebuild and row format as parse_runs.py.

Run from the project root:
    python3 src/parse_big.py
    python3 src/parse_big.py --min-ascension 10 --max-runs 50000
Output: data/processed/card_choices_big.csv
"""
import argparse
import csv
import json
from pathlib import Path

import parse_runs  # same folder: reuses the deck rebuild and the row format


class _Record:
    """Lets parse_runs.parse_run read an in-memory run as if it were a .run file."""

    def __init__(self, stem, data):
        self.stem = stem
        self._text = json.dumps(data)

    def read_text(self):
        return self._text


def iter_events(path, chunk=32 * 1024 * 1024):
    """Yield the runs of a huge JSON array one at a time, without loading the whole file."""
    dec = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        buf = f.read(chunk)
        pos = buf.index("[") + 1
        while True:
            while pos < len(buf) and buf[pos] in ", \n\r\t":
                pos += 1
            if pos >= len(buf):
                more = f.read(chunk)
                if not more:
                    return
                buf, pos = buf[pos:] + more, 0
                continue
            if buf[pos] == "]":
                return
            try:
                obj, end = dec.raw_decode(buf, pos)
            except json.JSONDecodeError:
                more = f.read(chunk)
                if not more:
                    raise
                buf, pos = buf[pos:] + more, 0
                continue
            yield obj.get("event", obj)
            pos = end


def with_ascenders_bane(run):
    """Ascension 10+ runs start with the curse Ascender's Bane, which the start deck in
    parse_runs.py does not include. Record it as a card gained at floor 0."""
    if (run.get("ascension_level") or 0) < 10:
        return run
    run = dict(run)
    run["items_purchased"] = ["AscendersBane"] + list(run.get("items_purchased", []))
    run["item_purchase_floors"] = [0] + list(run.get("item_purchase_floors", []))
    return run


def wanted(run, args):
    if run.get("is_daily") or run.get("is_endless") or run.get("is_trial"):
        return False
    if run.get("character_chosen") not in parse_runs.START_DECK:
        return False
    if not run.get("card_choices"):
        return False
    if (run.get("ascension_level") or 0) < args.min_ascension:
        return False
    if args.wins_only and not run.get("victory"):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/big/november.json")
    ap.add_argument("--out", default="data/processed/card_choices_big.csv")
    ap.add_argument("--min-ascension", type=int, default=15)
    ap.add_argument("--wins-only", action="store_true")
    ap.add_argument("--max-runs", type=int, default=0, help="stop after this many kept runs (0 = all)")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = kept = bad = n_rows = n_screens = exact = 0
    mismatch_sum = 0
    writer = None
    with out.open("w", newline="") as f:
        for run in iter_events(args.input):
            seen += 1
            if seen % 20000 == 0:
                print(f"read {seen} runs, kept {kept}")
            if not wanted(run, args):
                continue
            try:
                rows, mm = parse_runs.parse_run(_Record(str(kept), with_ascenders_bane(run)))
            except (KeyError, TypeError, ValueError):
                bad += 1
                continue
            rows = [r for r in rows if r["floor"] >= 1]  # floor 0 = Neow's start-of-run picks
            if not rows:
                continue
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
            writer.writerows(rows)
            kept += 1
            n_rows += len(rows)
            n_screens += len({r["screen_id"] for r in rows})
            mismatch_sum += mm
            exact += mm == 0
            if args.max_runs and kept >= args.max_runs:
                break

    if not kept:
        print("no runs matched the filters")
        return
    print(f"read {seen} runs, kept {kept} (skipped {bad} unreadable)")
    print(f"{n_screens} screens, {n_rows} rows -> {out}")
    print(f"deck rebuild: {exact}/{kept} runs match the real final deck exactly, "
          f"mean card difference {mismatch_sum / kept:.2f}")


if __name__ == "__main__":
    main()
