from __future__ import annotations

import argparse
import sys

from .config import MODEL_DIR, OUTPUT_DIR, PROCESSED_DIR
from .download_data import download_all
from .features import build_current_candidates, build_training_table
from .train import load_model_predictions, plot_probability_chart, train_model


def run_pipeline(args: argparse.Namespace) -> None:
    if args.download or args.force_download:
        download_all(force=args.force_download)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building training table...")
    training_df = build_training_table()
    training_path = PROCESSED_DIR / "training_players.csv"
    training_df.to_csv(training_path, index=False)
    print(f"  {len(training_df)} training rows -> {training_path}")

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

    print("Scoring current candidates...")
    predictions = load_model_predictions(candidates_df, model_path)
    predictions_path = OUTPUT_DIR / "golden_boot_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    chart_path = OUTPUT_DIR / "golden_boot_probabilities.png"
    plot_probability_chart(predictions, chart_path, top_n=args.top_n)
    print(f"  predictions -> {predictions_path}")
    print(f"  probability chart -> {chart_path}")

    print("\nTop candidates:")
    preview_cols = ["player", "country", "predicted_goals", "golden_boot_probability"]
    preview_text = predictions[preview_cols].head(args.top_n).to_string(index=False)
    stdout_encoding = sys.stdout.encoding or "utf-8"
    print(preview_text.encode(stdout_encoding, errors="replace").decode(stdout_encoding))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and run the Golden Boot model.")
    parser.add_argument("--download", action="store_true", help="Download missing Kaggle data.")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload Kaggle data even if files already exist.",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=25,
        help="Stop training when validation loss has not improved for this many epochs. Use 0 to disable.",
    )
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument(
        "--min-current-minutes",
        type=int,
        default=300,
        help="Minimum 2025/26 club minutes for a player to be scored.",
    )
    parser.add_argument(
        "--max-candidates-per-country",
        type=int,
        default=10,
        help="Keep only the top current attackers per national team. Use 0 for no cap.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    run_pipeline(parser.parse_args())


if __name__ == "__main__":
    main()
