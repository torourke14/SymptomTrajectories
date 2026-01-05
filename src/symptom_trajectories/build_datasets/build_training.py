from pathlib import Path
import json
import pandas as pd
from .encoding import encode_codes_factorize
from .labels import compute_labels
from .splits import build_splits, compute_pos_weights


def build_training_from_dir(
    train_split: float = 0.75,
    val_split: float = 0.10,
    test_split: float = 0.15,
    seed: int = 42,
    dx_adherence_threshold: float = 0.8,
    phq9_spike: float = 5.0,
    utilization_spike: float = 2.0,
    relapse_horizon_days: int = 30,
):
  parsed_dir = Path(parsed_dir)
  train_dir = Path(train_dir)
  train_dir.mkdir(parents=True, exist_ok=True)

  patients = pd.read_csv(parsed_dir / "patients.csv", parse_dates=["birthdate"], low_memory=False, compression="infer")
  encounters = pd.read_csv(parsed_dir / "encounters.csv", parse_dates=["start_time"], low_memory=False, compression="infer")
  conditions = pd.read_csv(parsed_dir / "conditions.csv", parse_dates=["start_time"], low_memory=False, compression="infer")
  meds = pd.read_csv(parsed_dir / "medications.csv", parse_dates=["start_time"], low_memory=False, compression="infer")
  observations = pd.read_csv(parsed_dir / "observations.csv", parse_dates=["obs_time"], low_memory=False, compression="infer")
  daily = pd.read_csv(parsed_dir / "daily.csv", parse_dates=["date"], low_memory=False, compression="infer")
  events = pd.read_csv(parsed_dir / "events.csv", parse_dates=["date"], usecols=["patient_id", "date", "event_type", "code"], low_memory=False, compression="infer")

  per_day_codes, cookbook = encode_codes_factorize(events)
  daily_labeled = compute_labels(
    daily, encounters, conditions,
    dx_adherence_threshold=dx_adherence_threshold,
    phq9_spike=phq9_spike,
    utilization_spike=utilization_spike,
    relapse_horizon_days=relapse_horizon_days,
  )

  model_table = daily_labeled.merge(per_day_codes, how="left", on=["patient_id", "date"])
  model_table["code_ids"] = model_table["code_ids"].fillna("[]")
  model_table.sort_values(["patient_id","date"], inplace=True)

  num_cols = model_table.select_dtypes(include=["number","bool"]).columns
  model_table[num_cols] = model_table[num_cols].fillna(0)

  splits = build_splits(model_table, train_split=train_split, val_split=val_split, test_split=test_split, seed=seed)
  posw   = compute_pos_weights(model_table)

  cookbook.to_csv(train_dir / "dx_uniques.csv", index=False)
  model_table.to_csv(train_dir / "model_table.csv", index=False)
  splits.to_csv(train_dir / "splits.csv", index=False)
  with open(train_dir / "pos_weights.json", "w") as f:
    json.dump(posw, f)

  return model_table, splits, posw, cookbook
