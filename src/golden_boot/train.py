from __future__ import annotations

import os
import random
from copy import deepcopy
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
from torch.utils.data import DataLoader, TensorDataset

from .config import FEATURE_COLUMNS
from .model import GoldenBootMLP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_scaler(df: pd.DataFrame) -> dict:
    """Fit a tiny standardizer with pandas/numpy."""
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
    features = scaler["features"]
    values = df.copy()
    for col in features:
        if col not in values:
            values[col] = np.nan
        values[col] = pd.to_numeric(values[col], errors="coerce")
        values[col] = values[col].fillna(float(scaler["medians"][col]))
        values[col] = (values[col] - float(scaler["means"][col])) / float(scaler["stds"][col])
    return values[features].to_numpy(dtype=np.float32)


def train_model(
    training_df: pd.DataFrame,
    model_path: Path,
    learning_curve_path: Path,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 42,
) -> dict:
    if len(training_df) < 10:
        raise ValueError("Training data is too small after filtering.")

    set_seed(seed)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    learning_curve_path.parent.mkdir(parents=True, exist_ok=True)

    indices = np.arange(len(training_df))
    np.random.default_rng(seed).shuffle(indices)
    val_size = max(1, int(0.2 * len(indices)))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_df = training_df.iloc[train_idx].reset_index(drop=True)
    val_df = training_df.iloc[val_idx].reset_index(drop=True)
    scaler = fit_scaler(train_df)

    x_train = torch.tensor(transform_features(train_df, scaler))
    y_train = torch.tensor(train_df["target_goals"].to_numpy(dtype=np.float32))
    x_val = torch.tensor(transform_features(val_df, scaler))
    y_val = torch.tensor(val_df["target_goals"].to_numpy(dtype=np.float32))

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    model = GoldenBootMLP(input_size=len(FEATURE_COLUMNS))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn: nn.Module = nn.MSELoss()
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_preds = model(x_val)
            val_loss = float(loss_fn(val_preds, y_val).item())

        history["train_loss"].append(float(np.mean(batch_losses)))
        history["val_loss"].append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": len(FEATURE_COLUMNS),
            "feature_columns": list(FEATURE_COLUMNS),
            "scaler": scaler,
            "epochs": epochs,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "lr": lr,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
        },
        model_path,
    )
    plot_learning_curve(history, learning_curve_path)
    return history


def load_model_predictions(candidates_df: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = GoldenBootMLP(input_size=int(checkpoint["input_size"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x = torch.tensor(transform_features(candidates_df, checkpoint["scaler"]))
    with torch.no_grad():
        raw_predictions = model(x).numpy()

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
    ax.set_title("MLP Learning Curve")
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
