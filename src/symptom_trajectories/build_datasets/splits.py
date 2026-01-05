import numpy as np
import pandas as pd


def build_splits(model_table, train_split=0.75, val_split=0.10, test_split=0.15, seed=42):
  cols = [c for c in ["y_relapse", "y_det"] if c in model_table.columns]
  rng = np.random.default_rng(seed)

  per_patient_max = model_table.groupby("patient_id")[cols].max(min_count=1).fillna(0)
  has_pos = (per_patient_max.max(axis=1) >= 1).to_numpy()

  pos_pids = per_patient_max.index[has_pos].to_numpy()
  neg_pids = per_patient_max.index[~has_pos].to_numpy()
  rng.shuffle(pos_pids)
  rng.shuffle(neg_pids)

  def _split_group(arr):
    n = len(arr)
    n_tr = int(round(n * train_split))
    n_val = int(round(n * val_split))
    if n_tr + n_val > n:
        n_val = max(0, n - n_tr)
    n_te = max(0, n - n_tr - n_val)
    return arr[:n_tr], arr[n_tr:n_tr+n_val], arr[n_tr+n_val:]

  pos_tr, pos_val, pos_te = _split_group(pos_pids)
  neg_tr, neg_val, neg_te = _split_group(neg_pids)

  train_ids = np.array(list(pos_tr) + list(neg_tr))
  val_ids   = np.array(list(pos_val) + list(neg_val))
  test_ids  = np.array(list(pos_te) + list(neg_te))
  rng.shuffle(train_ids)
  rng.shuffle(val_ids)
  rng.shuffle(test_ids)
  split = (
    [(pid, "train") for pid in train_ids] +
    [(pid, "val")   for pid in val_ids] +
    [(pid, "test")  for pid in test_ids]
  )
  return pd.DataFrame(split, columns=["patient_id","split"])


def compute_pos_weights(model_table):
  def _pos_weight(series):
    counts = series.astype("float32").value_counts()
    pos = int(counts.get(1.0, 0))
    neg = int(counts.get(0.0, 0))
    return float(neg / pos) if pos > 0 and neg > 0 else None

  scalar_cols = [c for c in ["y_relapse", "y_det", "non_adherent_flag"] if c in model_table.columns]
  scalar = { c: _pos_weight(model_table[c]) for c in scalar_cols }
  scalar = { k: v for k, v in scalar.items() if v is not None }

  cb_cols = [c for c in model_table.columns if c.startswith("cb_")]
  cb = { c: _pos_weight(model_table[c]) for c in cb_cols }
  cb = { k: v for k, v in cb.items() if v is not None }

  return { "scalar": scalar, "cb": cb }
