import numpy as np
import pandas as pd


def compute_labels(
    daily,
    encounters,
    conditions,
    dx_adherence_threshold: float = 0.8,
    phq9_spike: float = 5.0,
    utilization_spike: float = 2.0,
    relapse_horizon_days: int = 30,
):
  daily_df = daily.copy()

  encounter_days_df  = encounters.assign(date=encounters["start_time"].dt.floor("D"))[ ["patient_id","date","enc_class_bucket"] ]
  condition_days_df  = conditions.assign(date=conditions["start_time"].dt.floor("D"))[ ["patient_id","date","description"] ]

  pdc_col = next(c for c in daily_df.columns if c.startswith("pdc_"))
  daily_df["non_adherent_flag"] = (
      (daily_df["ad_gap_days"].fillna(0).astype(float) >= 1) |
      (daily_df[pdc_col].fillna(0).astype(float) < dx_adherence_threshold)
  ).astype("int8")

  mental_health_regex     = r"depress|anxiet|bipolar|schizo|psych|suicid|ptsd|panic|ocd|substance|addict"
  severe_condition_regex  = r"severe|psychosis|suicid|mania|catatonia|acute"

  mh_conditions      = condition_days_df.loc[
      condition_days_df["description"].str.contains(mental_health_regex, case=False, na=False),
      ["patient_id","date"]
  ]
  severe_conditions  = condition_days_df.loc[
      condition_days_df["description"].str.contains(severe_condition_regex, case=False, na=False),
      ["patient_id","date"]
  ]
  acute_encounters   = encounter_days_df.loc[
      encounter_days_df["enc_class_bucket"].str.upper().isin(["EMERGENCY","INPATIENT"]),
      ["patient_id","date"]
  ]

  ed_with_mh_same_day   = mh_conditions.merge(acute_encounters, on=["patient_id","date"], how="inner").drop_duplicates()
  long_medication_gap   = daily_df.loc[daily_df["ad_gap_days"].fillna(0).astype(float) >= 30, ["patient_id","date"]]
  relapse_anchor_dates  = pd.concat([ed_with_mh_same_day, severe_conditions, long_medication_gap], ignore_index=True).drop_duplicates()

  daily_df["y_relapse"] = 0
  for patient_id, patient_days in daily_df.groupby("patient_id", sort=False):
      row_index   = patient_days.index
      day_dates   = patient_days["date"].to_numpy(dtype="datetime64[D]")
      anchor_dates= relapse_anchor_dates.loc[
          relapse_anchor_dates["patient_id"] == patient_id, "date"
      ].to_numpy(dtype="datetime64[D]")
      anchor_dates = np.sort(anchor_dates)

      next_anchor_idx     = np.searchsorted(anchor_dates, day_dates + np.timedelta64(1, 'D'), side='left')
      has_future_anchor   = (next_anchor_idx < anchor_dates.size)
      within_horizon      = np.zeros_like(has_future_anchor, dtype=bool)
      valid_rows          = np.where(has_future_anchor)[0]
      if valid_rows.size:
        deltas = anchor_dates[next_anchor_idx[valid_rows]] - day_dates[valid_rows]
        within_horizon[valid_rows] = deltas <= np.timedelta64(relapse_horizon_days, 'D')
      daily_df.loc[row_index, "y_relapse"] = (has_future_anchor & within_horizon).astype("int8")

  daily_df["y_det"] = 0
  for patient_id, patient_days in daily_df.groupby("patient_id", sort=False):
      row_index        = patient_days.index
      day_dates        = patient_days["date"].to_numpy(dtype="datetime64[D]")
      phq9_values      = patient_days.get("phq9", pd.Series(index=row_index, dtype=float)).to_numpy(dtype=float)
      util7_values     = patient_days.get("util_7d", pd.Series(index=row_index, dtype=float)).to_numpy(dtype=float)
      adherence_values = patient_days.get("ad_covered", pd.Series(index=row_index, dtype=float)).to_numpy(dtype=float)

      n_days                 = len(patient_days)
      future_phq9_max        = np.full(n_days, np.nan, dtype=float)
      future_util7_max       = np.zeros(n_days, dtype=float)
      future_any_med_covered = np.zeros(n_days, dtype=bool)

      for i in range(n_days):
          in_horizon = (day_dates > day_dates[i]) & (day_dates <= day_dates[i] + np.timedelta64(relapse_horizon_days, 'D'))
          idx_future = np.where(in_horizon)[0]
          if idx_future.size:
              future_phq_vals   = phq9_values[idx_future]
              future_phq9_max[i]= np.nanmax(future_phq_vals) if np.isfinite(future_phq_vals).any() else np.nan
              future_util7_max[i]= util7_values[idx_future].max()
              future_any_med_covered[i] = (adherence_values[idx_future] == 1).any()

      future_phq9_rise   = (~np.isnan(future_phq9_max)) & (~np.isnan(phq9_values)) & ((future_phq9_max - phq9_values) >= phq9_spike)
      utilization_spike_flag  = (future_util7_max - util7_values) >= utilization_spike
      medication_start   = (adherence_values == 0) & future_any_med_covered
      deterioration_flag = future_phq9_rise | utilization_spike_flag | medication_start
      daily_df.loc[row_index, "y_det"] = deterioration_flag.astype("int8")

  return daily_df
