import pandas as pd


def build_coverage_calendar(med_df):
  rows = []
  for _, row in med_df.iterrows():
    pid = row["patient_id"]
    start = row["start_time"]
    stop = row["stop_time"]
    if pd.isna(start):
      continue
    if pd.isna(stop) or stop < start:
      stop = start + pd.Timedelta(days=30)
    for day in pd.date_range(start.floor("D"), stop.floor("D"), freq="D"):
      rows.append((row["patient_id"], day, 1))

  cov = pd.DataFrame(rows, columns=["patient_id","date","ad_covered"])
  cov = (cov
    .groupby(["patient_id","date"], as_index=False)["ad_covered"]
    .max()
  )
  return cov


def add_pdc_and_gaps(cov, window_days):
  cov = cov.sort_values(["patient_id","date"])
  cov["date"] = pd.to_datetime(cov["date"], errors="coerce").dt.floor("D")
  cov = cov.dropna(subset=["patient_id","date"])
  cov["patient_id"] = cov["patient_id"].astype(str)
  cov["ad_covered"] = cov["ad_covered"].astype("int8").fillna(0).astype("int8")

  def per_patient(g):
    g = g.set_index("date").asfreq("D", fill_value=0).sort_index()
    pid = str(g["patient_id"].iloc[0])
    pdc = g["ad_covered"].rolling(window_days, min_periods=1).mean().astype("float32")
    zero = (g["ad_covered"] == 0).astype("int16")
    groups = g["ad_covered"].cumsum()
    gap = zero.groupby(groups).cumsum().astype("int16")

    return pd.DataFrame({
      "patient_id": pid,
      "date": g.index,
      "ad_covered": g["ad_covered"].astype("int8").values,
      f"pdc_{window_days}": pdc.values,
      "ad_gap_days": gap.values,
    })

  out = (cov
    .groupby("patient_id", group_keys=False)
    .apply(per_patient)
  ).reset_index(drop=True)

  out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.floor("D")
  out["patient_id"] = out["patient_id"].astype(str)
  return out
