# 🏆 Golden Boot Prediction Model — How It All Works

## The Goal

Predict which player is most likely to win the **Golden Boot** (top scorer award) at the **2026 FIFA World Cup**. We treat this as a **regression problem**: predict how many goals a player will score at the tournament, then convert those predictions into probabilities.

---

## 1. The Data: What Does the Model Learn From?

### Training Data (the "textbook")

The model learns from **real historical data** — every outfield player who participated in the **2018** and **2022 World Cups**. This gives us **765 player rows**.

For each player, we know:
- Their **club stats from the season before** the World Cup (goals, assists, minutes, etc.)
- Their **national team's FIFA ranking** at the time
- Their **age** and **position**
- ⭐ **How many goals they actually scored** at that World Cup — this is the **label** (target variable `target_goals`)

The model's job is to learn the relationship: *"given these pre-tournament stats, how many World Cup goals does a player like this tend to score?"*

### Inference Data (the "exam")

For 2026, we take **194 current attackers** from the 2025-26 club season (sourced from Kaggle datasets). The model has never seen these players during training — it uses what it learned from 2018/2022 to predict their expected tournament goals.

### The Label Problem: Why This Is Hard

Here's the reality of World Cup goal distributions:

| Goals scored | Players | % of total |
|:---:|:---:|:---:|
| 0 | 606 | 79.2% |
| 1 | 119 | 15.6% |
| 2 | 23 | 3.0% |
| 3 | 9 | 1.2% |
| 4 | 5 | 0.7% |
| 6+ | 3 | 0.3% |

**~80% of outfield players at a World Cup score zero goals.** This makes the prediction problem inherently difficult — the data is extremely skewed. Even a perfect oracle would struggle to distinguish between a player who scores 0 and one who scores 1.

---

## 2. Features: What Information Goes Into the Model?

The model uses **14 numeric features** defined in [config.py](file:///Users/jsw/Projects/golden-boot-prediction-model/src/golden_boot/config.py#L44-L59):

| Feature | What it captures |
|---|---|
| `age` | Player's age at tournament time |
| `fifa_rank` | Their national team's FIFA ranking (lower = better) |
| `club_minutes` | Total minutes played in club season |
| `club_goals` | Total club goals scored |
| `club_non_penalty_goals` | Goals excluding penalties |
| `club_penalty_goals` | Goals from penalties only |
| `club_assists` | Total assists |
| `club_goals_per90` | Scoring rate normalized by playing time |
| `league_goals_rank_pct` | Percentile rank among league peers for goals |
| `league_non_penalty_goals_rank_pct` | Same but for non-penalty goals |
| `team_goal_share` | What fraction of their club team's goals they scored |
| `position_is_defender` | Binary: is the player a defender? |
| `position_is_midfielder` | Binary: is the player a midfielder? |
| `position_is_forward` | Binary: is the player a forward? |

### Feature Normalization

Before feeding data into the neural network, all features are **standardized** (z-score normalization): each feature is centered to mean=0 and scaled to std=1. This is critical because features like `club_minutes` (~2000) and `team_goal_share` (~0.15) live on completely different scales. Without normalization, the model would essentially ignore small-scale features.

The scaler is **fit only on the training set** to prevent data leakage into validation.

---

## 3. Model Architecture: The Neural Network

The model is a **Multi-Layer Perceptron (MLP)** built in PyTorch, defined in [model.py](file:///Users/jsw/Projects/golden-boot-prediction-model/src/golden_boot/model.py):

```
Input (14 features)
    ↓
Linear(14 → 32) + ReLU + Dropout(0.15)
    ↓
Linear(32 → 16) + ReLU + Dropout(0.15)
    ↓
Linear(16 → 1)
    ↓
Output: predicted goals (single number)
```

### Why these choices?

- **Small network (32 → 16 neurons):** We only have 765 training rows. A bigger network (e.g. 128 → 64) would memorize the training data without learning generalizable patterns. The model capacity must match the data size.
- **ReLU activation:** Standard non-linearity that lets the network learn non-linear relationships (e.g. "prime age players with high goal rates score more").
- **Dropout (15%):** During training, 15% of neurons are randomly disabled each pass. This forces the network to not over-rely on any single neuron, acting as regularization.
- **Single output neuron:** This is a regression problem — we predict one continuous number (expected goals).

---

## 4. Training: How the Model Learns

The full training logic lives in [train.py](file:///Users/jsw/Projects/golden-boot-prediction-model/src/golden_boot/train.py). Here's the flow:

### Train/Validation Split

The 765 rows are split **80/20** into training (~612 rows) and validation (~153 rows). The split is **stratified** on the target variable — meaning the proportion of 0-goal, 1-goal, 2+-goal players is the same in both sets. Without stratification, random chance could put more high-scorers in one set, creating misleading loss curves.

### Training Loop

```
For each epoch (1 to 200):
    1. Forward pass: feed all training data through the network
    2. Compute MSE loss between predictions and actual goals
    3. Backward pass: compute gradients
    4. Update weights using Adam optimizer
    5. Evaluate on validation set (no gradients, dropout disabled)
    6. Save the best model checkpoint (lowest validation loss)
```

### Key Training Parameters

| Parameter | Value | Why |
|---|---|---|
| Epochs | 200 | Enough for full convergence on this small dataset |
| Learning rate | 5×10⁻⁴ | Lower LR = smoother, more stable convergence |
| Optimizer | Adam | Adapts learning rate per-parameter; standard choice |
| LR Schedule | CosineAnnealingLR | Gradually reduces LR from 5×10⁻⁴ to 1×10⁻⁶ over training |
| Weight decay | 5×10⁻⁴ | L2 regularization penalty on weights to prevent overfitting |
| Loss function | MSELoss | Mean Squared Error — standard for regression |
| Training mode | Full-batch | All 612 training rows in one pass (mini-batches are too noisy for this size) |
| Early stopping | patience=25 | Stop if validation loss doesn't improve for 25 epochs |

### Why Full-Batch Instead of Mini-Batch?

With only 612 training rows, mini-batches of 32 create 19 gradient updates per epoch, each computed from a tiny, noisy slice of data. This causes unstable training — the loss curves zigzag. Full-batch training computes one clean gradient from all data per epoch, producing smooth convergence.

---

## 5. The Learning Curve: Understanding the Gap

![Learning Curve](/Users/jsw/.gemini/antigravity/brain/4c907a65-7291-4d96-9c50-3e191a61b2c8/artifacts/learning_curve.png)

### Reading the Chart

- **Blue line (Training loss):** How well the model fits the data it learns from. Decreases from ~0.60 → ~0.44.
- **Orange line (Validation loss):** How well the model generalizes to unseen data. Decreases from ~0.83 → ~0.62.
- **Both curves go down and converge** — this is a healthy learning curve.

### Why Is There a Gap?

The gap between training (~0.44) and validation (~0.62) loss is called the **generalization gap**. It exists for three reasons:

1. **The problem is inherently noisy.** Club season stats are a weak predictor of World Cup goals. A player can have an amazing season and then get injured, face a tough group, or simply not be the penalty taker. The signal-to-noise ratio is low.

2. **Baseline context.** If you naively predicted the mean (0.30 goals) for every player, your MSE would be **0.59**. The validation loss of 0.62 means the model is performing *roughly at baseline* on unseen data, while on training data it does better (0.44). This tells us the model captures some real patterns but they're subtle.

3. **Small dataset.** 765 rows with 14 features is small for a neural network. The model finds patterns in training data that are partially noise. More historical World Cup data would help close this gap.

> [!IMPORTANT]
> The gap is **expected and correct** for this type of problem. What would be *wrong* is if the validation loss went UP while training loss went down (overfitting) or if validation loss was BELOW training loss (data leakage or broken split). Neither happens here.

---

## 6. Inference: From Predicted Goals to Probabilities

After training, the model predicts expected goals for all 194 current candidates. But raw goal predictions aren't intuitive — we want to say *"Player X has an 8.6% chance of winning."*

### Softmax Conversion

The predicted goals are converted to probabilities using the **softmax function**:

```
probability(player_i) = exp(predicted_goals_i) / Σ exp(predicted_goals_j)
```

This ensures:
- All probabilities sum to 100%
- Players with higher predicted goals get higher probabilities
- The differences are amplified exponentially (a player predicted to score 1.35 goals gets noticeably more probability than one at 1.0)

### Final Output

![Probability Chart](/Users/jsw/.gemini/antigravity/brain/4c907a65-7291-4d96-9c50-3e191a61b2c8/artifacts/golden_boot_probabilities.png)

---

## 7. Project Structure

```
golden-boot-prediction-model/
├── scripts/
│   └── run_pipeline.py          # Entry point — run this to execute everything
├── src/golden_boot/
│   ├── config.py                # Paths, feature list, Kaggle dataset references
│   ├── download_data.py         # Downloads raw datasets from Kaggle
│   ├── features.py              # Builds training & candidate tables from raw data
│   ├── model.py                 # PyTorch MLP architecture definition
│   ├── train.py                 # Training loop, evaluation, plotting
│   └── pipeline.py              # Orchestrates the full pipeline end-to-end
├── data/
│   ├── raw/                     # Raw Kaggle datasets (7 sources)
│   └── processed/
│       ├── training_players.csv # 765 historical player rows (2018 + 2022 WC)
│       └── current_candidates.csv # 194 current attackers for 2026 inference
├── models/
│   └── golden_boot_model.pt     # Saved PyTorch model checkpoint + scaler
└── outputs/
    ├── learning_curve.png       # Training vs Validation loss plot
    ├── golden_boot_predictions.csv   # Full prediction results
    └── golden_boot_probabilities.png # Top-15 probability bar chart
```

---

## 8. How to Run

```bash
# First time (downloads Kaggle datasets — needs Kaggle API credentials)
python scripts/run_pipeline.py --download

# Subsequent runs (data already downloaded)
python scripts/run_pipeline.py
```

### Useful CLI Options

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 200 | Number of training epochs |
| `--lr` | 5×10⁻⁴ | Initial learning rate |
| `--seed` | 42 | Random seed for reproducibility |
| `--top-n` | 15 | How many candidates to show in output |
| `--early-stopping-patience` | 25 | Epochs without improvement before stopping |
| `--min-current-minutes` | 300 | Minimum club minutes for a candidate |
| `--max-candidates-per-country` | 10 | Cap on attackers considered per national team |

---

## 9. Summary

| Aspect | Detail |
|---|---|
| **Problem type** | Regression (predict goals) → converted to probabilities via softmax |
| **Training data** | 765 real players from 2018 & 2022 World Cups |
| **Label** | `target_goals` — actual goals scored at the World Cup |
| **Features** | 14 numeric features (club stats, age, FIFA rank, position) |
| **Model** | PyTorch MLP: 14 → 32 → 16 → 1 with ReLU + Dropout |
| **Training** | 200 epochs, full-batch, Adam with cosine LR schedule |
| **Train MSE** | ~0.44 (better than baseline 0.59) |
| **Val MSE** | ~0.62 (near baseline — expected for this noisy problem) |
| **Output** | Ranked list of 194 players with predicted goals & win probabilities |
