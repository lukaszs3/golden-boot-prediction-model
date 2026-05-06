# 2026 World Cup Golden Boot Predictor

Small PyTorch project that trains a simple MLP to predict World Cup goals for likely 2026 Golden Boot candidates.

The pipeline intentionally stays uncomplicated:

1. Downloads Kaggle datasets.
2. Builds a recent-only training set from the 2018 and 2022 World Cups.
3. Trains on top-30 FIFA-ranked national teams, using all outfield players for training and attacking players for 2026 scoring.
4. Trains a PyTorch regression MLP on tournament goals.
5. Applies the model to active 2025/26 players and plots top-candidate probabilities.

## Kaggle Data Sources

The code fetches these datasets with `kagglehub`:

- `cashncarry/fifaworldranking` for historical FIFA rankings.
- `joshfjelstul/world-cup-database` for 2018 World Cup squads and goal labels. The dataset is historical, but this project filters it to 2018 only.
- `swaptr/fifa-world-cup-2022-player-data` for 2022 World Cup player performance and goal labels.
- `diegobartoli/top5legauesplayers-statsandphys` for 2017/18 pre-World Cup club form.
- `vivovinco/20212022-football-player-stats` for 2021/22 pre-World Cup club form.
- `hubertsidorowicz/football-players-stats-2025-2026` for current active player form.
- `zkskhurram/fifa-and-football-complete-dataset-19302022` for a Kaggle top-30 January 2026 ranking snapshot used during inference when available.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If Kaggle requires credentials on your machine, configure them in the normal Kaggle way before running the downloader.

## Run

```bash
python scripts/run_pipeline.py --download
```

Useful options:

```bash
python scripts/run_pipeline.py --epochs 250
python scripts/run_pipeline.py --early-stopping-patience 25
python scripts/run_pipeline.py --top-n 15
python scripts/run_pipeline.py --min-current-minutes 500
python scripts/run_pipeline.py --max-candidates-per-country 8
```

## What The Training Dataset Looks Like

The training table is written to `data/processed/training_players.csv`.

Current sample run with the checked-in raw data:

- 765 training rows total.
- 280 rows from the 2018 World Cup.
- 485 rows from the 2022 World Cup.
- 606 rows with `target_goals = 0`.
- 159 rows with `target_goals > 0`.

Each row represents one outfield player in one tournament sample. The target is the number of World Cup goals scored in that tournament.

Columns:

- `source_tournament` - source label (`2018 World Cup` or `2022 World Cup`).
- `player` - player name.
- `country` - national team name.
- `country_code` - FIFA-style team code.
- `position` - tournament or source position label.
- `target_goals` - supervised label for training.
- `age` - player age at the tournament.
- `fifa_rank` - FIFA ranking snapshot near tournament start.
- `club_minutes` - pre-tournament club minutes.
- `club_goals` - pre-tournament club goals.
- `club_non_penalty_goals` - pre-tournament non-penalty goals.
- `club_penalty_goals` - pre-tournament penalty goals.
- `club_assists` - pre-tournament club assists.
- `club_goals_per90` - derived as `club_goals / (club_minutes / 90)`.
- `league_goals_rank_pct` - percentile rank inside the domestic league by total goals.
- `league_non_penalty_goals_rank_pct` - percentile rank inside the domestic league by non-penalty goals.
- `team_goal_share` - share of club goals scored by the player.
- `position_is_defender` - `1.0` if the player row includes a defender role, else `0.0`.
- `position_is_midfielder` - `1.0` if the player row includes a midfielder role, else `0.0`.
- `position_is_forward` - `1.0` if the player row includes an attacking role (`FW/ST/CF/SS/LW/RW`), else `0.0`.

CSV header:

```text
source_tournament,player,country,country_code,position,target_goals,age,fifa_rank,club_minutes,club_goals,club_non_penalty_goals,club_penalty_goals,club_assists,club_goals_per90,league_goals_rank_pct,league_non_penalty_goals_rank_pct,team_goal_share,position_is_defender,position_is_midfielder,position_is_forward
```

## How The Dataset Is Built

### 2018 rows

- FIFA ranking snapshot is taken at `2018-06-14`.
- Only top-30 national teams are kept.
- Player pool comes from `squads.csv` in the World Cup database.
- All outfield players are kept. `GK` rows are removed.
- Goal labels are built from `goals.csv`, counting non-own-goals per player.
- Age is calculated from `birth_date` and tournament start.
- Pre-tournament club features come from the 2017/18 top-five-leagues dataset.
- Missing club stats after joins are filled with `0.0`.

### 2022 rows

- FIFA ranking snapshot is taken at `2022-11-20`.
- Only top-30 national teams are kept.
- Player pool comes from `player_stats.csv` in the 2022 World Cup dataset.
- All outfield players are kept. `GK` rows are removed.
- `goals` from the 2022 dataset becomes `target_goals`.
- Pre-tournament club features come from the 2021/22 player-stats dataset.
- Missing club stats after joins are filled with `0.0`.

### 2026 candidate rows

The inference table is written to `data/processed/current_candidates.csv`.

Default filters:

- active 2025/26 club players only,
- top-30 current FIFA teams,
- attacking roles matching `FW|ST|CF|SS|LW|RW`,
- minimum `300` club minutes,
- maximum `10` candidates per country.

In the current sample run this produces `194` candidate rows across `29` national teams.

## How Training Works

The full run is:

1. Build the merged training table from 2018 and 2022 data.
2. Shuffle rows with the configured random seed.
3. Split the table into `80%` train and `20%` validation.
4. Fit preprocessing only on the training split:
   - median imputation for missing numeric features,
   - mean/std normalization for every feature column.
5. Train `GoldenBootMLP` from `src/golden_boot/model.py`.
6. After every epoch, compute train and validation MSE.
7. Keep the checkpoint from the epoch with the best validation loss.
8. Stop early when validation loss has not improved for the configured patience window.
9. Save the model, scaler metadata, and training summary to `models/golden_boot_model.pt`.

Position is now part of the numeric feature set through three binary flags, so the model can distinguish defenders, midfielders, and forwards even though training uses all outfield players.

To improve generalization across seasons and leagues, the model also gets league-relative and team-relative scoring context:

- domestic-league goal rank percentile,
- domestic-league non-penalty goal rank percentile,
- club goal-share,
- separation of penalty and non-penalty goals.

The model currently uses:

- input size = `14` numeric features,
- hidden layers = `64 -> 32`,
- activations = `ReLU`,
- dropout = `0.20` then `0.10`,
- output = single regression value (`predicted_goals`).

The command-line arguments below directly affect training now:

- `--epochs`
- `--early-stopping-patience`
- `--lr`
- `--batch-size`
- `--seed`

## How Predictions Are Produced

For 2026 candidates, the pipeline:

1. applies the saved scaler,
2. runs the MLP to get `predicted_goals`,
3. clips negatives to `0.0`,
4. converts scores into `golden_boot_probability` with a softmax over the candidate pool,
5. sorts candidates by probability.

That softmax probability is a ranking signal inside the filtered candidate pool, not a calibrated real-world probability.

## Outputs

After a successful run:

- `data/processed/training_players.csv` - model training rows.
- `data/processed/current_candidates.csv` - filtered 2026 candidate rows.
- `models/golden_boot_model.pt` - PyTorch checkpoint with scaler metadata.
- `outputs/learning_curve.png` - train/validation loss curve.
- `outputs/golden_boot_probabilities.png` - top candidate probability chart.
- `outputs/golden_boot_predictions.csv` - ranked candidate predictions.

This is an educational model, not an official squad or betting model. The 2026 candidate list is built from current top-five-league activity plus the top-30 national-team filter, so it should be treated as a reasonable candidate pool rather than a confirmed World Cup roster.
