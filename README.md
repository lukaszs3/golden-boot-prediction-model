# 2026 World Cup Golden Boot Predictor

Small PyTorch project that trains a simple MLP to predict World Cup goals for likely 2026 Golden Boot candidates.

The pipeline intentionally stays uncomplicated:

1. Downloads Kaggle datasets.
2. Builds a recent-only training set from the 2018 and 2022 World Cups.
3. Filters to forwards and attacking midfielders from top-30 FIFA-ranked national teams.
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
python scripts/run_pipeline.py --top-n 15
python scripts/run_pipeline.py --min-current-minutes 500
python scripts/run_pipeline.py --max-candidates-per-country 8
```

## Outputs

After a successful run:

- `data/processed/training_players.csv` - model training rows.
- `data/processed/current_candidates.csv` - filtered 2026 candidate rows.
- `models/golden_boot_mlp.pt` - PyTorch checkpoint with scaler metadata.
- `outputs/learning_curve.png` - train/validation loss curve.
- `outputs/golden_boot_probabilities.png` - top candidate probability chart.
- `outputs/golden_boot_predictions.csv` - ranked candidate predictions.

This is an educational model, not an official squad or betting model. The 2026 candidate list is built from current top-five-league activity plus the top-30 national-team filter, so it should be treated as a reasonable candidate pool rather than a confirmed World Cup roster.
