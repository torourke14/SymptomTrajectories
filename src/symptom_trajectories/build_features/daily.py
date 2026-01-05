import pandas as pd
import numpy as np
import re
from .constants import ENCOUNTER_TYPES


def dense_grid(activity_df):
  df = activity_df[["patient_id", "date"]].copy()
  df = df.dropna(subset=["patient_id", "date"])
  df["patient_id"] = df["patient_id"].astype(str)
  df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")
  df = df.dropna(subset=["date"])

  agg = (df
    .groupby("patient_id")["date"]
    .agg(["min","max"])
    .reset_index()
  )

  rows = []
  for _, row in agg.iterrows():
    for d in pd.date_range(row["min"], row["max"], freq="D"):
      rows.append((row["patient_id"], d))

  return pd.DataFrame(rows, columns=["patient_id","date"])


def build_daily_features(
  pats_df, enc, cond, med, obs,
  smoke, preg, alcohol, ad_daily,
  phq9, phq2, gad7, auditc, dast10,
  max_seq_days, pdc_window_days
):
  def require_cols(df, name, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
      raise ValueError(f"{name} missing columns: {missing}")

  dates_enc = enc.assign(date=enc["start_time"].dt.floor("D"))[ ["patient_id","date"] ]
  dates_cond= cond.assign(date=cond["start_time"].dt.floor("D"))[ ["patient_id","date"] ]
  dates_med = med.assign(date=med["start_time"].dt.floor("D"))[ ["patient_id","date"] ]
  dates_obs = obs.assign(date=obs["obs_time"].dt.floor("D"))[ ["patient_id","date"] ]

  activity = (
    pd.concat([dates_enc, dates_cond, dates_med, dates_obs], ignore_index=True)
      .dropna()
      .drop_duplicates()
  )

  require_cols(activity, "activity", ["patient_id","date"])
  daily = (dense_grid(activity)
    .reset_index(drop=True)
    .sort_values(["patient_id","date"])
    .groupby("patient_id", group_keys=False, as_index=False).tail(max_seq_days)
  )

  daily = daily.merge(pats_df[["patient_id","birthdate"]], how="left", on="patient_id")
  daily["age_years"] = (daily["date"] - daily["birthdate"]).dt.days // 365
  daily = daily.drop(columns=["birthdate"])

  keep_keys = daily[["patient_id","date"]].drop_duplicates()

  daily = daily.merge(pats_df[["patient_id","birthdate", "sex", "race"]], how="left", on="patient_id")
  daily["age_years"] = ((daily["date"] - daily["birthdate"]).dt.days // 365).astype("int16")
  daily = daily.drop(columns=["birthdate"])

  enc_day = enc.assign(date=enc["start_time"].dt.floor("D"))
  enc_day = enc_day.merge(keep_keys, on=["patient_id","date"], how="inner")
  enc_counts = (
    pd.crosstab(index=[enc_day["patient_id"], enc_day["date"]],
                columns=enc_day["enc_class_bucket"])
    .reset_index()
    .rename_axis(None, axis=1)
  )

  daily = daily.merge(enc_counts, how="left", on=["patient_id","date"])
  bucket_cols = [c for c in ["EMERGENCY","INPATIENT","OUTPATIENT","URGENTCARE","WELLNESS","OTHER"] if c in daily.columns]
  for buck in bucket_cols:
    if buck not in daily.columns:
      daily[buck] = 0
  if bucket_cols:
    daily[bucket_cols] = daily[bucket_cols].fillna(0).astype("int16")
    daily["visits_per_day"] = daily[bucket_cols].sum(axis=1).astype("int16")
  else:
    daily["visits_per_day"] = 0

  daily = daily.sort_values(["patient_id","date"])
  daily["util_7d"] = (
    daily.groupby("patient_id", observed=True)["visits_per_day"]
      .transform(lambda s: s.rolling(7, min_periods=1).sum())
      .astype("float32")
  )
  daily = daily.drop(columns=["visits_per_day"])

  def _merge_score(base, score_df):
    return base.merge(score_df.merge(keep_keys, on=["patient_id","date"], how="inner"), on=["patient_id","date"], how="left")

  daily = _merge_score(daily, phq9)
  daily = _merge_score(daily, phq2)
  daily = _merge_score(daily, gad7)
  daily = _merge_score(daily, auditc)
  daily = _merge_score(daily, dast10)

  smo  = smoke.merge(keep_keys, on=["patient_id","date"], how="inner")
  daily   = daily.merge(smo, on=["patient_id","date"], how="left")

  prego   = preg.merge(keep_keys, on=["patient_id","date"], how="inner")
  preg_agg = prego.groupby(["patient_id","date"], as_index=False)["pregnancy_pos"].max()
  daily   = daily.merge(preg_agg, on=["patient_id","date"], how="left")

  alc    = alcohol.merge(keep_keys, on=["patient_id","date"], how="inner")
  daily   = daily.merge(alc, on=["patient_id","date"], how="left")

  for col in [c for c in daily.columns if c.startswith(("smoke_", "alcohol_", "pregnancy_pos"))]:
    daily[col] = daily[col].fillna(0).astype("int8")

  add_    = ad_daily.merge(keep_keys, on=["patient_id","date"], how="inner")
  daily   = daily.merge(add_, on=["patient_id","date"], how="left")
  daily[["ad_covered", f"pdc_{pdc_window_days}", "ad_gap_days"]] = (
    daily[["ad_covered", f"pdc_{pdc_window_days}", "ad_gap_days"]].fillna(0))

  daily["ad_covered"] = daily["ad_covered"].astype("int8")
  daily["ad_gap_days"] = daily["ad_gap_days"].astype("int16")
  daily[f"pdc_{pdc_window_days}"] = daily[f"pdc_{pdc_window_days}"].astype("float32")

  daily = daily.sort_values(["patient_id","date"])
  daily["prev_date"] = daily.groupby("patient_id", observed=True)["date"].shift(1)
  daily["days_since_prev"] = (daily["date"] - daily["prev_date"]).dt.days.fillna(0).astype("int16")
  daily = daily.drop(columns=["prev_date"])

  daily["sex_M"] = (daily["sex"] == "M").astype("int8")
  daily["sex_F"] = (daily["sex"] == "F").astype("int8")

  for r in daily["race"].value_counts(dropna=False).head(6).index.tolist():
    key = re.sub("[^A-Za-z0-9]+","_", str(r)).lower()
    daily[f"race_{key}"] = (daily["race"] == r).astype("int8")
  daily = daily.drop(columns=["race", "sex"])

  for c in daily.select_dtypes(include=["float64","float32"]).columns:
    daily[c] = daily[c].astype("float32")
  for c in daily.select_dtypes(include=["int64"]).columns:
    daily[c] = daily[c].astype("int16")

  return daily.sort_values(["patient_id","date"]).reset_index(drop=True)


def clean_daily(pat, enc, con, med, obs, dayyy, events):
  dayly = dayyy.copy()

  if "phq9_screen" in dayly.columns:
    if "phq9" not in dayly.columns:
        dayly["phq9"] = np.nan
    dayly["phq9"] = dayly["phq9"].fillna(dayly["phq9_screen"])
    dayly = dayly.drop(columns=["phq9_screen"])

  num_fills = [
    "phq9", "phq2", "gad7", "auditc", "dast10",
    "util_7d", "ad_gap_days", "ad_covered", "pregnancy_pos"
  ]
  for col in num_fills:
    if col in dayly.columns:
      dayly[col] = dayly[col].astype("float32").fillna(0.0)

  expected_enc_buckets = set(ENCOUNTER_TYPES.values()).union({"OTHER"})
  for buck in expected_enc_buckets:
    if buck not in dayly.columns:
      dayly[buck] = 0
    dayly[buck] = dayly[buck].fillna(0).astype("int16")

  for c in [col for col in dayly.columns if col.startswith("sex_") or col.startswith("race_")]:
    dayly[c] = dayly[c].fillna(0).astype("int8")

  if "days_since_prev" in dayly.columns:
    dayly["days_since_prev"] = dayly["days_since_prev"].fillna(0).astype("int32")

  return dayly


def one_hot(df_in, col, prefix, classes):
  out = df_in.copy()
  for k in classes:
    out[f"{prefix}_{k}"] = (out[col] == k).astype("int8")
  out[f"{prefix}_unknown"] = (~out[col].isin(classes)).astype("int8")
  return out.drop(columns=[col])
