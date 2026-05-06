# Pipeline.py Documentation

This module orchestrates the entire Golden Boot prediction workflow, from downloading data through training the model to generating predictions and visualizations. It serves as the main entry point and coordinator for the project.

## Table of Contents
1. [Main Functions](#main-functions)
2. [Pipeline Orchestration](#pipeline-orchestration)
3. [Argument Parser](#argument-parser)
4. [Execution Flow](#execution-flow)

---

## Main Functions

### `run_pipeline(args: argparse.Namespace) -> None`
**Purpose:** Executes the complete Golden Boot prediction pipeline in sequence.

**Parameters:**
- `args` (argparse.Namespace): Command-line arguments containing configuration parameters

**Process Flow:**

#### 1. Data Download Phase
```python
if args.download or args.force_download:
    download_all(force=args.force_download)
```
- Downloads required Kaggle datasets if `--download` flag is set or if `--force-download` is used
- `--force-download` will re-download even if files already exist
- Uses the `download_all()` function from `download_data.py`

#### 2. Directory Setup
```python
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
```
- Creates three directories if they don't exist:
  - `PROCESSED_DIR`: Stores processed CSV files
  - `MODEL_DIR`: Stores trained model weights
  - `OUTPUT_DIR`: Stores predictions and visualizations

#### 3. Training Data Preparation
```python
print("Building training table...")
training_df = build_training_table()
training_path = PROCESSED_DIR / "training_players.csv"
training_df.to_csv(training_path, index=False)
print(f"  {len(training_df)} training rows -> {training_path}")
```
**What happens:**
- Calls `build_training_table()` from features.py
- Combines 2018 and 2022 World Cup data with actual goal labels
- Saves to `data/processed/training_players.csv`
- Logs number of training samples and output path

**Output:** `training_players.csv` containing player statistics and actual tournament goals

#### 4. Candidate Preparation
```python
print("Building 2026 candidate table...")
candidates_df = build_current_candidates(
    min_minutes=args.min_current_minutes,
    max_per_country=args.max_candidates_per_country,
)
if candidates_df.empty:
    raise ValueError("No current candidates passed the filters.")
candidates_path = PROCESSED_DIR / "current_candidates.csv"
candidates_df.to_csv(candidates_path, index=False)
print(f"  {len(candidates_df)} candidates -> {candidates_path}")
```
**What happens:**
- Calls `build_current_candidates()` with filtering parameters
- Filters players by:
  - Minimum club minutes played (`--min-current-minutes`, default 300)
  - Maximum players per country (`--max-candidates-per-country`, default 10)
  - Forward/attacking positions only
  - Top 30 FIFA-ranked countries
- Validates that at least some candidates exist
- Saves to `data/processed/current_candidates.csv`
- Logs candidate count and output path

**Output:** `current_candidates.csv` containing filtered 2026 World Cup candidates

#### 5. Model Training
```python
model_path = MODEL_DIR / "golden_boot_model.pt"
learning_curve_path = OUTPUT_DIR / "learning_curve.png"
print("Training MLP model...")
history = train_model(
    training_df,
    model_path=model_path,
    learning_curve_path=learning_curve_path,
    epochs=args.epochs,
    lr=args.lr,
    batch_size=args.batch_size,
    seed=args.seed,
    early_stopping_patience=args.early_stopping_patience,
)
```
**Parameters used:**
- `epochs` (default 200): Number of training epochs
- `lr` (default 5e-4): Learning rate for optimizer
- `batch_size` (default 32): Samples per batch
- `seed` (default 42): Random seed for reproducibility
- `early_stopping_patience` (default 25): Stop if validation doesn't improve for N epochs (0 = disabled)

**What happens:**
- Calls `train_model()` from train.py
- Trains a Multi-Layer Perceptron neural network
- Uses historical data as training set
- Implements early stopping to prevent overfitting
- Saves best model checkpoint to `models/golden_boot_model.pt`
- Returns training history dictionary

**Returns History Dictionary:**
- `best_train_loss`: Best training loss achieved
- `best_val_loss`: Best validation loss achieved
- `best_epoch`: Epoch with best validation loss
- `trained_epochs`: Total epochs trained
- `stopped_early`: Boolean indicating if early stopping was triggered
- `early_stopping_patience`: Patience value used

#### 6. Training Report
```python
print(
    "  best checkpoint:",
    f"train={history['best_train_loss']:.4f}",
    f"val={history['best_val_loss']:.4f}",
    f"epoch={history['best_epoch']}/{history['trained_epochs']}",
)
if history["stopped_early"]:
    print(
        "  early stopping:",
        f"stopped after {history['trained_epochs']} epochs",
        f"(patience={history['early_stopping_patience']})",
    )
print(f"  model -> {model_path}")
print(f"  learning curve -> {learning_curve_path}")
```
- Displays best training and validation losses
- Shows which epoch had best performance
- Reports if early stopping was triggered and after how many epochs
- Shows file paths for model and learning curve visualization

#### 7. Inference/Scoring
```python
print("Scoring current candidates...")
predictions = load_model_predictions(candidates_df, model_path)
predictions_path = OUTPUT_DIR / "golden_boot_predictions.csv"
predictions.to_csv(predictions_path, index=False)
```
**What happens:**
- Calls `load_model_predictions()` from train.py
- Loads the trained model from checkpoint
- Scores all current candidates
- Generates two key outputs:
  - `predicted_goals`: Number of goals player is predicted to score
  - `golden_boot_probability`: Probability they win the Golden Boot
- Saves predictions to CSV
- Logs output path

**Output:** `golden_boot_predictions.csv` containing:
- Player name and country
- Predicted goals
- Golden Boot probability

#### 8. Visualization
```python
chart_path = OUTPUT_DIR / "golden_boot_probabilities.png"
plot_probability_chart(predictions, chart_path, top_n=args.top_n)
```
- Calls `plot_probability_chart()` from train.py
- Creates visualization of top N candidates
- `top_n` parameter controls how many players to show (default 15)
- Saves chart to `outputs/golden_boot_probabilities.png`

#### 9. Final Report
```python
print("\nTop candidates:")
preview_cols = ["player", "country", "predicted_goals", "golden_boot_probability"]
preview_text = predictions[preview_cols].head(args.top_n).to_string(index=False)
stdout_encoding = sys.stdout.encoding or "utf-8"
print(preview_text.encode(stdout_encoding, errors="replace").decode(stdout_encoding))
```
- Displays top N predictions to console
- Shows player, country, predicted goals, and probability
- Handles encoding to avoid character display issues
- Provides quick visual confirmation of results

---

## Argument Parser

### `build_parser() -> argparse.ArgumentParser`
**Purpose:** Constructs the command-line argument parser for the pipeline.

**Arguments:**

#### Data Download Arguments
- `--download` (flag): Download missing Kaggle datasets
- `--force-download` (flag): Re-download all Kaggle datasets even if they exist

#### Training Arguments
- `--epochs` (int, default=200): Number of training epochs
  - Higher = more training iterations, risk of overfitting
  
- `--lr` (float, default=5e-4): Learning rate
  - Higher = faster learning but risk of divergence
  - Lower = slower learning but more stable
  
- `--batch-size` (int, default=32): Number of samples per gradient update
  - Higher = faster training but less frequent updates
  - Lower = more frequent updates but noisier gradients
  
- `--seed` (int, default=42): Random seed for reproducibility
  - Same seed = same results across runs
  
- `--early-stopping-patience` (int, default=25): Patience for early stopping
  - Stop if validation loss doesn't improve for N epochs
  - Set to 0 to disable early stopping

#### Filtering Arguments
- `--top-n` (int, default=15): Number of top candidates to display

- `--min-current-minutes` (int, default=300): Minimum club minutes for candidates
  - Filters out rarely-played players
  
- `--max-candidates-per-country` (int, default=10): Maximum candidates per country
  - Prevents biasing toward large countries
  - Set to 0 for no limit

**Example Usage:**
```bash
# Default settings
python -m golden_boot.pipeline

# With custom training parameters
python -m golden_boot.pipeline --epochs 300 --lr 1e-3 --batch-size 16

# Download data and use stricter filtering
python -m golden_boot.pipeline --download --min-current-minutes 500 --max-candidates-per-country 5

# Force re-download and show top 20 candidates
python -m golden_boot.pipeline --force-download --top-n 20

# Disable early stopping for longer training
python -m golden_boot.pipeline --epochs 500 --early-stopping-patience 0
```

---

## Main Entry Point

### `main() -> None`
**Purpose:** Parses arguments and executes the pipeline.

**Process:**
1. Creates argument parser via `build_parser()`
2. Parses command-line arguments
3. Calls `run_pipeline()` with parsed arguments

**Example:**
```bash
python -m golden_boot.pipeline --help  # Show help
python -m golden_boot.pipeline          # Run with defaults
```

---

## Execution Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│              Command Line Arguments                     │
│  (epochs, lr, batch_size, download, candidates, etc)   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│          run_pipeline(args) Execution                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. DATA DOWNLOAD (Optional)                            │
│     └─> download_all(force=args.force_download)        │
│                                                         │
│  2. DIRECTORY SETUP                                     │
│     ├─> PROCESSED_DIR/                                 │
│     ├─> MODEL_DIR/                                     │
│     └─> OUTPUT_DIR/                                    │
│                                                         │
│  3. TRAINING DATA PREPARATION                           │
│     └─> build_training_table()                         │
│         ├─> 2018 World Cup data with goals             │
│         └─> 2022 World Cup data with goals             │
│         └─> Saved: training_players.csv                │
│                                                         │
│  4. CANDIDATE PREPARATION                              │
│     └─> build_current_candidates()                     │
│         ├─> Filter by position (forwards)              │
│         ├─> Filter by minutes (min_current_minutes)    │
│         ├─> Limit per country (max_candidates)         │
│         └─> Saved: current_candidates.csv              │
│                                                         │
│  5. MODEL TRAINING                                      │
│     └─> train_model(training_df, ...)                  │
│         ├─> Input: Training data with actual goals     │
│         ├─> Epochs: 0 to N with early stopping         │
│         ├─> Output: Best model checkpoint              │
│         └─> Returns: Training history                  │
│         └─> Saved: golden_boot_model.pt                │
│         └─> Saved: learning_curve.png                  │
│                                                         │
│  6. INFERENCE/SCORING                                  │
│     └─> load_model_predictions(candidates, model)      │
│         ├─> Load trained model                         │
│         ├─> Forward pass through candidates            │
│         ├─> Output: Predicted goals + probabilities    │
│         └─> Saved: golden_boot_predictions.csv         │
│                                                         │
│  7. VISUALIZATION                                       │
│     └─> plot_probability_chart(predictions, top_n)     │
│         └─> Saved: golden_boot_probabilities.png       │
│                                                         │
│  8. CONSOLE REPORT                                      │
│     └─> Display top N candidates                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                   OUTPUT FILES                          │
├─────────────────────────────────────────────────────────┤
│ • training_players.csv      (historical training data)  │
│ • current_candidates.csv    (2026 candidates)          │
│ • golden_boot_model.pt      (trained model weights)    │
│ • learning_curve.png        (training visualization)   │
│ • golden_boot_predictions.csv (final predictions)      │
│ • golden_boot_probabilities.png (top predictions viz)  │
└─────────────────────────────────────────────────────────┘
```

---

## Key Design Patterns

### 1. Separation of Concerns
- **Data Preparation**: Handled by `features.py`
- **Model Training**: Handled by `train.py`
- **Orchestration**: Handled by `pipeline.py`
- Each module has a clear, focused responsibility

### 2. Configuration-Driven
- All parameters are command-line configurable
- Default values are sensible but can be overridden
- Enables experimentation without code changes

### 3. Checkpoint-Based Training
- Model is saved at best validation loss
- Early stopping prevents unnecessary training
- Training history is returned for analysis

### 4. Checkpoint Reuse
- Trained model can be used for multiple candidates
- No need to retrain for different filtering

### 5. Defensive Validation
- Checks that candidates exist after filtering
- Raises informative errors if requirements aren't met

### 6. Progressive Output
- Each step prints progress updates
- Console output helps track execution
- Handles encoding issues for international characters

---

## Common Execution Scenarios

### Scenario 1: First-Time Setup
```bash
python -m golden_boot.pipeline --download
```
- Downloads datasets
- Trains model with defaults
- Generates predictions
- ~5-10 minutes depending on network speed

### Scenario 2: Quick Prediction Update
```bash
python -m golden_boot.pipeline
```
- Uses existing datasets
- Trains model with defaults
- ~1-2 minutes

### Scenario 3: Experiment with Parameters
```bash
python -m golden_boot.pipeline --epochs 500 --lr 1e-3 --batch-size 16 --early-stopping-patience 50
```
- Longer training with different hyperparameters
- Tests if model can improve with more training

### Scenario 4: Stricter Candidate Filter
```bash
python -m golden_boot.pipeline --min-current-minutes 900 --max-candidates-per-country 3
```
- Only very active players
- Focuses on elite candidates

### Scenario 5: Extended Analysis
```bash
python -m golden_boot.pipeline --top-n 25 --max-candidates-per-country 15
```
- More candidates included
- Larger visualization and output

---

## Output Files Summary

| File | Purpose | Location |
|------|---------|----------|
| `training_players.csv` | Historical training data (2018/2022 WC) | `data/processed/` |
| `current_candidates.csv` | Filtered 2026 World Cup candidates | `data/processed/` |
| `golden_boot_model.pt` | Trained neural network weights | `models/` |
| `learning_curve.png` | Training loss over epochs | `outputs/` |
| `golden_boot_predictions.csv` | Final predictions with probabilities | `outputs/` |
| `golden_boot_probabilities.png` | Top candidates visualization | `outputs/` |

---

## Error Handling

### Empty Candidates Error
```python
if candidates_df.empty:
    raise ValueError("No current candidates passed the filters.")
```
- Validates that filtering didn't eliminate everyone
- Suggests relaxing filter constraints if this occurs
- Example fix: Reduce `--min-current-minutes` or increase `--max-candidates-per-country`

### Missing Data Handling
- `download_all()` manages dataset download with fallback logic
- `build_training_table()` handles missing values gracefully
- Missing player stats are filled with 0 (conservative assumption)

---

## Performance Notes

- **Data Loading**: ~30 seconds for all datasets
- **Training**: ~2-5 minutes depending on epochs and early stopping
- **Inference**: ~1 second for ~300 candidates
- **Total Runtime**: ~3-10 minutes depending on parameters

---

## Extending the Pipeline

To add new functionality:

1. **Add new argument** to `build_parser()`
2. **Add logic** to `run_pipeline()` using the new argument
3. **Implement helper function** in appropriate module
4. **Update documentation** here

Example: Adding validation set metrics
```python
# In run_pipeline() after training
if args.show_validation_metrics:
    print_validation_metrics(history)
```

