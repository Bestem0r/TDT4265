import argparse
from pathlib import Path
from typing import Sequence

import torch

from data import DEFAULT_ARTIFACT_PATH, DEFAULT_DATA_ROOT, DEFAULT_PREDICTIONS_PATH, DEVICE
from model import BreastCNN
from pipeline import predict_cases_pytorch, train_model_pytorch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ODELIA breast MRI 3D CNN")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a 3D CNN model")
    train_parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    train_parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH)
    train_parser.add_argument("--fit-splits", nargs="+", default=("train",))
    train_parser.add_argument("--val-splits", nargs="+", default=("val",))
    train_parser.add_argument("--max-breasts", type=int, default=None)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--monitor-every", type=int, default=20)
    train_parser.add_argument("--checkpoint-path", type=Path, default=None)
    train_parser.add_argument("--best-checkpoint-path", type=Path, default=None)
    train_parser.add_argument("--monitor-csv-path", type=Path, default=None)
    train_parser.add_argument("--target-specificity", type=float, default=0.90)
    train_parser.add_argument("--target-sensitivity", type=float, default=0.90)
    train_parser.add_argument("--selection-metric", choices=("aggregate", "val_loss"), default="aggregate")

    pred_parser = subparsers.add_parser("predict", help="Generate predictions")
    pred_parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH)
    pred_parser.add_argument("--cases-root", type=Path, default=DEFAULT_DATA_ROOT)
    pred_parser.add_argument("--output-file", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    pred_parser.add_argument("--max-breasts", type=int, default=None)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        print("Training 3D CNN model...")
        checkpoint_path = args.checkpoint_path if args.checkpoint_path is not None else args.artifact
        best_checkpoint_path = (
            args.best_checkpoint_path
            if args.best_checkpoint_path is not None
            else args.artifact.with_name(f"{args.artifact.stem}_best{args.artifact.suffix}")
        )

        model = train_model_pytorch(
            data_root=args.data_root,
            fit_splits=args.fit_splits,
            val_splits=args.val_splits,
            max_breasts=args.max_breasts,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            monitor_every=args.monitor_every,
            checkpoint_path=checkpoint_path,
            best_checkpoint_path=best_checkpoint_path,
            monitor_csv_path=args.monitor_csv_path,
            target_specificity=args.target_specificity,
            target_sensitivity=args.target_sensitivity,
            selection_metric=args.selection_metric,
        )
        torch.save(model.state_dict(), args.artifact)
        print(f"Model saved to {args.artifact}")
        return 0

    if args.command == "predict":
        print("Loading model...")
        model = BreastCNN(in_channels=5, num_classes=3).to(DEVICE)
        model.load_state_dict(torch.load(args.artifact, map_location=DEVICE))
        model.eval()

        print("Generating predictions...")
        output_path = predict_cases_pytorch(
            model=model,
            cases_root=args.cases_root,
            output_file=args.output_file,
            max_breasts=args.max_breasts,
        )
        print(f"Predictions saved to {output_path}")
        return 0

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
