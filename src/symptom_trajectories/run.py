from pathlib import Path
import argparse

from .config import load_config
from .build_features.build import build_features
from .build_datasets.build_training import build_training_from_dir


def run_build_features(cfg):
    daily, events, pats, enc, cond, med, obs = build_features(
        raw_dir=cfg.paths.raw_dir,
        out_dir=cfg.paths.processed_dir,
        pdc_window_days=cfg.build.pdc_window_days,
        max_sequence_days=cfg.build.max_sequence_days,
    )
    print(f"Wrote features to {cfg.paths.processed_dir}")
    print(f"Rows: daily={len(daily)}, events={len(events)}, patients={pats['patient_id'].nunique()}")


def run_prepare_splits(cfg):
    model_table, splits, posw, cookbook = build_training_from_dir(
        train_split=cfg.prepare_splits.train_split,
        val_split=cfg.prepare_splits.val_split,
        test_split=cfg.prepare_splits.test_split,
        seed=cfg.prepare_splits.seed,
        dx_adherence_threshold=cfg.prepare_splits.dx_adherence_threshold,
        phq9_spike=cfg.prepare_splits.phq9_spike,
        utilization_spike=cfg.prepare_splits.utilization_spike,
        relapse_horizon_days=cfg.prepare_splits.relapse_horizon_days,
    )
    print(f"Wrote train artifacts to {cfg.paths.train_output_dir} (rows={len(model_table)}, patients={model_table['patient_id'].nunique()})")


def run_train(cfg, model = ""):
    return



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Symptom trajectories data pipeline")
    parser.add_argument("--config", "-c", type=str, default="config/default.yaml", help="YAML config file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-features", help="Parse raw synthea files into feature tables.")
    build.set_defaults(func=run_build_features)

    prepare = subparsers.add_parser("prepare-splits", help="Build model-ready tables and train/val/test splits.")
    prepare.set_defaults(func=run_prepare_splits)

    xgb = subparsers.add_parser("train", help="Build model-ready tables and train/val/test splits.")
    xgb.add_argument("--model", type=Path, default="xgb", help="Name of model to train (xgb, lstm, lstt)")
    xgb.set_defaults(func=run_train, model="xgb")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config, model=args.model)
    args.func(cfg)


if __name__ == "__main__":
    main()
