import os, json
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from google.colab import drive
import xgboost as xgb
from xgboost.callback import EarlyStopping
import google.generativeai as genai

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, classification_report
from sklearn.calibration import calibration_curve

DRIVE_MOUNT = "/content/drive"
TRAIN_DIR = '/content/drive/MyDrive/UT-AIHC/HRP/training-data'
ARTIFACTS_DIR = '/content/drive/MyDrive/UT-AIHC/HRP/artifacts'
drive.mount(DRIVE_MOUNT, force_remount=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

genai.configure(api_key="AIzaSyBYahAZ6aAHR8H0KA83qnWmbUAkaq0iJVE")
model = genai.GenerativeModel("gemma-3n-e2b-it")

EARLY_STOPPING_ROUNDS = 50
N_ESTIMATORS          = 600
LEARNING_RATE         = 0.05
MAX_DEPTH             = 4
SUBSAMPLE             = 0.8
COLSAMPLE_BYTREE      = 0.8

BINARY_HEADS = ["y_relapse", "y_det", "non_adherent_flag"]
REG_TARGET     = "y_gap_days"

early_stop_bin = EarlyStopping(rounds=EARLY_STOPPING_ROUNDS, save_best=True, maximize=True)
early_stop_reg = EarlyStopping(rounds=EARLY_STOPPING_ROUNDS, save_best=True, maximize=False)



def parse_code_ids_column(series):
  return series.apply(lambda s: json.loads(s) if isinstance(s, str) else [])

def select_numeric_feature_columns(df):
  candidate_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
  drop_cols = set(["y_relapse", "y_det", "non_adherent_flag", "y_gap_days"])  # labels
  # Exclude one-hot leak-y columns if desired; keeping all engineered numerics is fine
  numeric_cols = [c for c in candidate_cols if c not in drop_cols]
  return numeric_cols

def get_training_data():
  model_table_path = os.path.join(TRAIN_DIR, "model_table.csv")
  splits_path = os.path.join(TRAIN_DIR, "splits.csv")
  posw_path = os.path.join(TRAIN_DIR, "pos_weights.json")  # optional

  model_table = pd.read_csv(model_table_path, parse_dates=["date"])
  splits = pd.read_csv(splits_path)
  try:
    with open(posw_path, "r") as f:
      pos_weights = json.load(f)
  except:
    pos_weights = {"scalar": {}}

  model_table["patient_id"] = model_table["patient_id"].astype(str)
  splits["patient_id"]      = splits["patient_id"].astype(str)

  return model_table, splits, pos_weights

"""
  Build sparse design matrices for train/val/test.
  - Numeric features: dense -> CSR
  - Code IDs: MultiLabelBinarizer(sparse_output=True)
  Returns: (X_train, X_val, X_test, y_dict, feature_names)
  """
def build_feature_matrices(model_table, splits):
  # Patient splits -> row masks
  train_pids = set(splits.loc[splits["split"] == "train", "patient_id"])
  val_pids   = set(splits.loc[splits["split"] == "val",   "patient_id"])
  test_pids  = set(splits.loc[splits["split"] == "test",  "patient_id"])

  df_train = model_table.loc[model_table["patient_id"].isin(train_pids)].copy()
  df_val   = model_table.loc[model_table["patient_id"].isin(val_pids)].copy()
  df_test  = model_table.loc[model_table["patient_id"].isin(test_pids)].copy()

  # Numeric features
  numeric_cols = select_numeric_feature_columns(model_table)
  Xn_train = csr_matrix(df_train[numeric_cols].to_numpy(dtype=np.float32))
  Xn_val   = csr_matrix(df_val[numeric_cols].to_numpy(dtype=np.float32))
  Xn_test  = csr_matrix(df_test[numeric_cols].to_numpy(dtype=np.float32))

  # Codes from codebook
  code_lists_train = parse_code_ids_column(df_train["code_ids"])
  code_lists_val   = parse_code_ids_column(df_val["code_ids"])
  code_lists_test  = parse_code_ids_column(df_test["code_ids"])

  all_code_lists = parse_code_ids_column(model_table["code_ids"])
  all_classes = sorted({cid for lst in all_code_lists for cid in lst})
  mlb = MultiLabelBinarizer(classes=all_classes, sparse_output=True)

  Xc_train = mlb.fit_transform(parse_code_ids_column(df_train["code_ids"]))
  Xc_val   = mlb.transform(parse_code_ids_column(df_val["code_ids"]))
  Xc_test  = mlb.transform(parse_code_ids_column(df_test["code_ids"]))

  # Combine
  X_train = hstack([Xn_train, Xc_train], format="csr")
  X_val   = hstack([Xn_val,   Xc_val],   format="csr")
  X_test  = hstack([Xn_test,  Xc_test],  format="csr")

  # Combined feature names: numeric + code#<id>
  numeric_feature_names = list(numeric_cols)
  code_feature_names    = [f"code#{int(i)}" for i in mlb.classes_]
  feature_names = numeric_feature_names + code_feature_names

  # Targets
  y = {
    "y_relapse": df_train["y_relapse"].to_numpy(dtype=np.float32),
    "y_det": df_train["y_det"].to_numpy(dtype=np.float32),
    "non_adherent_flag": df_train["non_adherent_flag"].to_numpy(dtype=np.float32),
    "y_gap_days": df_train.get("y_gap_days", pd.Series(0, index=df_train.index)).to_numpy(dtype=np.float32),
  }
  y_val = {
    "y_relapse": df_val["y_relapse"].to_numpy(dtype=np.float32),
    "y_det": df_val["y_det"].to_numpy(dtype=np.float32),
    "non_adherent_flag": df_val["non_adherent_flag"].to_numpy(dtype=np.float32),
    "y_gap_days": df_val.get("y_gap_days", pd.Series(0, index=df_val.index)).to_numpy(dtype=np.float32),
  }
  y_test = {
    "y_relapse": df_test["y_relapse"].to_numpy(dtype=np.float32),
    "y_det": df_test["y_det"].to_numpy(dtype=np.float32),
    "non_adherent_flag": df_test["non_adherent_flag"].to_numpy(dtype=np.float32),
    "y_gap_days": df_test.get("y_gap_days", pd.Series(0, index=df_test.index)).to_numpy(dtype=np.float32),
  }

  row_sets = {
    "train_rows": df_train[["patient_id", "date"]].reset_index(drop=True),
    "val_rows": df_val[["patient_id", "date"]].reset_index(drop=True),
    "test_rows": df_test[["patient_id", "date"]].reset_index(drop=True),
  }

  return X_train, X_val, X_test, y, y_val, y_test, feature_names, row_sets




def get_xgb_clf(scale_pos_weight=1.0, random_state=42):
  return xgb.XGBClassifier(
    objective="binary:logistic",
    n_estimators=N_ESTIMATORS,
    learning_rate=LEARNING_RATE,
    max_depth=MAX_DEPTH,
    subsample=SUBSAMPLE,
    colsample_bytree=COLSAMPLE_BYTREE,
    eval_metric="aucpr",
    tree_method="hist",
    random_state=random_state,
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1,
  )

def make_xgb_regressor(random_state=42):
  return xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=N_ESTIMATORS,
    learning_rate=LEARNING_RATE,
    max_depth=MAX_DEPTH,
    subsample=SUBSAMPLE,
    colsample_bytree=COLSAMPLE_BYTREE,
    eval_metric="rmse",
    tree_method="hist",
    random_state=random_state,
    n_jobs=-1,
)