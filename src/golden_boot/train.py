from __future__ import annotations

import os
import random
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / ".matplotlib-cache"),
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import FEATURE_COLUMNS
from .model import GoldenBootMLP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_scaler(df: pd.DataFrame) -> dict:
    values = df[FEATURE_COLUMNS].astype(float)
    medians = values.median().fillna(0.0)
    filled = values.fillna(medians)
    means = filled.mean()
    stds = filled.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    return {
        "features": list(FEATURE_COLUMNS),
        "medians": medians.to_dict(),
        "means": means.to_dict(),
        "stds": stds.to_dict(),
    }


def transform_features(df: pd.DataFrame, scaler: dict) -> np.ndarray:
    values = df.copy()
    for col in scaler["features"]:
        if col not in values:
            values[col] = np.nan
        values[col] = pd.to_numeric(values[col], errors="coerce")
        values[col] = values[col].fillna(float(scaler["medians"][col]))
        values[col] = (values[col] - float(scaler["means"][col])) / float(scaler["stds"][col])
    return values[scaler["features"]].to_numpy(dtype=np.float32)


def train_model(
    training_df: pd.DataFrame,
    model_path: Path,
    learning_curve_path: Path,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 42,
    early_stopping_patience: int = 25,
) -> dict:
    if len(training_df) < 10:
        raise ValueError("Training data is too small after filtering.")

    set_seed(seed)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    learning_curve_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    train_idx_list, val_idx_list = [], []
    
    # Stratify split based on target_goals to ensure similar distributions
    strat_keys = np.clip(training_df["target_goals"].fillna(0).to_numpy(), 0, 2)
    for key in np.unique(strat_keys):
        idx = np.where(strat_keys == key)[0]
        rng.shuffle(idx)
        v_size = max(1 if len(idx) > 1 else 0, int(0.2 * len(idx)))
        val_idx_list.extend(idx[:v_size])
        train_idx_list.extend(idx[v_size:])
        
    train_idx = np.array(train_idx_list)
    val_idx = np.array(val_idx_list)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    train_df = training_df.iloc[train_idx].reset_index(drop=True)
    val_df = training_df.iloc[val_idx].reset_index(drop=True)
    scaler = fit_scaler(train_df)

    x_train = transform_features(train_df, scaler)
    y_train = train_df["target_goals"].to_numpy(dtype=np.float32)
    x_val = transform_features(val_df, scaler)
    y_val = val_df["target_goals"].to_numpy(dtype=np.float32)
    x_train_tensor = torch.from_numpy(x_train)
    y_train_tensor = torch.from_numpy(y_train)
    x_val_tensor = torch.from_numpy(x_val)
    y_val_tensor = torch.from_numpy(y_val)

    model = GoldenBootMLP(input_size=x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # --- Training Phase (full-batch for small dataset) ---
        model.train()
        optimizer.zero_grad()
        predictions = model(x_train_tensor)
        loss = loss_fn(predictions, y_train_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # --- Validation Phase ---
        model.eval()
        with torch.no_grad():
            train_predictions = model(x_train_tensor)
            val_predictions = model(x_val_tensor)
            train_loss = float(loss_fn(train_predictions, y_train_tensor).item())
            val_loss = float(loss_fn(val_predictions, y_val_tensor).item())

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            break

    if best_state_dict is None:
        raise RuntimeError("MLP training did not produce a valid checkpoint.")

    model.load_state_dict(best_state_dict)

    history["best_epoch"] = best_epoch
    history["best_train_loss"] = best_train_loss
    history["best_val_loss"] = best_val_loss
    history["trained_epochs"] = len(history["train_loss"])
    history["stopped_early"] = len(history["train_loss"]) < epochs
    history["early_stopping_patience"] = early_stopping_patience

    torch.save(
        {
            "model_type": "mlp",
            "model_state_dict": model.state_dict(),
            "input_size": x_train.shape[1],
            "feature_columns": list(FEATURE_COLUMNS),
            "scaler": scaler,
            "epochs": epochs,
            "trained_epochs": len(history["train_loss"]),
            "best_epoch": best_epoch,
            "best_train_loss": best_train_loss,
            "best_val_loss": best_val_loss,
            "early_stopping_patience": early_stopping_patience,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
        },
        model_path,
    )
    plot_learning_curve(history, learning_curve_path)
    return history


def load_model_predictions(candidates_df: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    x = transform_features(candidates_df, checkpoint["scaler"])
    model = GoldenBootMLP(input_size=int(checkpoint["input_size"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        raw_predictions = model(torch.from_numpy(x)).numpy()

    predictions = candidates_df.copy()
    predictions["predicted_goals"] = np.maximum(raw_predictions, 0.0)
    predictions["golden_boot_probability"] = softmax_probabilities(
        predictions["predicted_goals"].to_numpy()
    )
    return predictions.sort_values("golden_boot_probability", ascending=False)


def softmax_probabilities(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return scores
    stable_scores = scores - np.max(scores)
    exp_scores = np.exp(stable_scores)
    return exp_scores / exp_scores.sum()


def _try_seaborn_theme() -> None:
    try:
        import seaborn as sns

        sns.set_theme(style="whitegrid", context="talk")
    except ImportError:
        plt.style.use("seaborn-v0_8-whitegrid")


def plot_learning_curve(history: dict, output_path: Path) -> None:
    _try_seaborn_theme()
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Training loss", linewidth=2.2)
    ax.plot(epochs, history["val_loss"], label="Validation loss", linewidth=2.2)

    # Mark the best epoch
    if "best_epoch" in history:
        best_ep = history["best_epoch"]
        best_val = history["best_val_loss"]
        ax.axvline(best_ep, color="gray", linestyle="--", alpha=0.6, label=f"Best epoch ({best_ep})")
        ax.scatter([best_ep], [best_val], color="orange", zorder=5, s=60)

    ax.set_title("MLP Regression Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_probability_chart(predictions: pd.DataFrame, output_path: Path, top_n: int = 15) -> None:
    _try_seaborn_theme()
    top = predictions.head(top_n).iloc[::-1].copy()
    labels = top["player"] + " (" + top["country_code"] + ")"
    chart_probs = softmax_probabilities(top["predicted_goals"].to_numpy())
    probabilities = pd.Series(chart_probs, index=top.index) * 100

    fig_height = max(6, 0.42 * len(top) + 2)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(top)))
    ax.barh(labels, probabilities, color=colors)
    ax.set_title("2026 World Cup Golden Boot Probability")
    ax.set_xlabel("Softmax probability among displayed candidates (%)")
    ax.set_ylabel("")

    for i, value in enumerate(probabilities):
        ax.text(value + 0.05, i, f"{value:.1f}%", va="center", fontsize=10)

    ax.set_xlim(0, max(probabilities.max() * 1.18, 1.0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
