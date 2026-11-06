# StS_predict

Predict the best card pick in Slay the Spire (the first game) with XGBoost. Work in progress.

The target is outcome based: which pick leads to a longer run, not which card players usually
take. Copying players is kept only as a reference.

## How it works

A value model predicts how many floors a run survives from its state (deck, upgrades, relics,
HP, floor, ascension). Each offered card, and skipping, is scored by the prediction for the
state it leads to, and the highest score wins.

Two ways to get training data:

- **Human runs.** Own `.run` files plus a large public dump. Player skill confounds the
  outcome: strong players both pick better and survive longer.
- **Simulator runs.** Full Ironclad runs played in
  [sts_lightspeed](https://github.com/gamerpuppy/sts_lightspeed) with random card picks, so
  there is no player skill in the data. It teaches what works for the simulator's combat bot,
  not necessarily for a human.

## Setup

Tested with Python 3.11.

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, XGBoost also needs `brew install libomp`. Run everything from the project root.

### Data

- Own runs: copy your `.run` files into `data/raw/<CHARACTER>/` (`IRONCLAD`, `THE_SILENT`,
  `DEFECT`, `WATCHER`). 52 runs are included.
- Big dump (not in the repo, 619 MB): the Dropbox sample from the 2020 r/slaythespire
  "77 Million Runs" post, saved as `data/big/november.json`. About 120k runs, all characters.
- `parse_big.py` keeps Ascension 15 and up, drops daily, endless and trial runs, and drops
  the floor 0 (Neow) picks.
- The deck at each screen is rebuilt from the start deck, card picks, shop buys and purges and
  event changes. Neow bonuses, transforms and relic effects are not captured, so the rebuilt
  deck matches the real final deck for only about two thirds of runs (196 of 300 checked).
- Boss relics never appear in the relic column of the human data.

### Simulator (optional, Ironclad only)

```
brew install cmake
git clone --recurse-submodules https://github.com/gamerpuppy/sts_lightspeed.git
cd sts_lightspeed/pybind11 && git fetch --depth 1 origin tag v2.13.6 && git checkout v2.13.6 && cd ..
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=$(which python) \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build --target slaythespire -j4
```

Run cmake with the venv active so the module matches the interpreter. The pybind11 checkout
is needed for Python 3.13 and newer, and the policy flag for CMake 4. Built and used on an
Apple Silicon Mac with Python 3.11.

## Scripts

| Script | What it does |
| --- | --- |
| `src/parse_runs.py` | own `.run` files to `data/processed/card_choices.csv` |
| `src/parse_big.py` | the big dump to `data/processed/card_choices_big.csv` |
| `src/train.py` | baselines and XGBoost classifier, ranker, regressor; 5-fold CV grouped by run |
| `src/train_value.py` | the value model; `--save PATH` also writes the model and its `.meta.json` |
| `src/sim_eval.py` | plays full Ironclad runs with different card policies on the same seeds |
| `src/sim_collect.py` | collects random-pick simulator runs in the same CSV format |
| `src/compare_picks.py` | per-card pick rates of the models against the players |

Typical order:

```
python src/parse_big.py
python src/train_value.py --save models/value_model.json
python src/sim_eval.py --sim-build ~/workspace/sts_lightspeed/build --model models/value_model.json --runs 1000
python src/sim_collect.py --sim-build ~/workspace/sts_lightspeed/build
python src/train_value.py --csv data/processed/sim_choices.csv --save models/sim_value_model.json
python src/compare_picks.py
```

`sim_collect.py` starts its seeds at 100000 so training seeds do not overlap the seeds
`sim_eval.py` tests on. `models/` holds the saved models (about 1.3 MB each).

## Results so far

Human data, Ascension 15 and up, grouped CV. Agreement with the player's pick: popularity
0.466, XGBoost classifier 0.488 (random 0.255). The value model on 216,738 screens from
21,890 runs: RMSE 11.88 floors against 12.47 for predicting the average for that floor.

Simulator, Ironclad A5, 1000 seeds, 1000 combat simulations per turn, every policy on the
same seeds. Mean floor, and the difference to the built-in policy with a 95% interval:

| Policy | Mean floor | Difference to built-in |
| --- | --- | --- |
| built-in | 17.5 | |
| always skip | 11.6 | -5.96 (-6.53 to -5.39) |
| random | 16.9 | -0.68 (-1.35 to -0.01) |
| random, never skip | 18.4 | +0.86 (+0.20 to +1.55) |
| popularity | 17.0 | -0.54 (-1.32 to +0.25) |
| value model, trained on human data | 21.5 | +3.93 (+3.21 to +4.69) |
| value model, trained on simulator data | 26.7 | +9.19 (+8.43 to +9.99) |

The human-trained model is still +3.06 (+2.47 to +3.67) ahead of random with no skips, so the
gain is not just a bigger deck. At A10 (seeds 5000 to 5999) it gains +3.44 (+2.76 to +4.12).

## Caveats

- The simulator results say the models pick well for the simulator's combat bot. The
  simulator-trained model was tested on the same bot it learned from.
- Neither model looks like a player. On 20,000 human screens they agree with the player's pick
  about 27% of the time (chance is about 25%) and with each other on 40%. Both skip far less
  than players. The simulator-trained model favors plain attacks and self-damage cards and
  rarely takes setup cards such as Barricade, Feel No Pain or Limit Break.
- Only Ironclad can be checked in the simulator. The other characters have human data only.
- Neither model is ready to give a human card advice.

## Next

- Retrain the human model on high-experience players only (assumes `player_experience`
  tracks skill, not checked).
- A second round of simulator data where the model plays most picks (`--epsilon`).
- Features for card type, cost and rarity; fix the model skipping too rarely.
- Overlay: needs a game mod to read the deck, relics, HP and offered cards.



## Disclaimer 

- LLM model used to for parsing.py modules as well as for research of the available datasets like the one I'm using november.7z
