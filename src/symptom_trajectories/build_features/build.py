from pathlib import Path
from typing import NoReturn
import pandas as pd
from .constants import SCREENING_TOT_CODES, SCREENING_KWS, ANTIDEPRESSANT_REGEX, DEFAULT_PDC_WINDOW_DAYS, DEFAULT_MAX_SEQUENCE_DAYS
from .observations import get_screening_scores
from .lifestyle import extract_life_obs
from .meds import build_coverage_calendar, add_pdc_and_gaps
from .daily import build_daily_features, one_hot, clean_daily
from .events import build_event_stream
from .synthea import configure_tables, load_raw_tables
from tqdm import tqdm


"""

"""

def build_feature_tables(
    prog_bar: tqdm[NoReturn],
    pat_view: pd.DataFrame, enc_view: pd.DataFrame, 
    cnd_view: pd.DataFrame, med_view: pd.DataFrame, 
    obs_view: pd.DataFrame,
    pdc_window_days: int = DEFAULT_PDC_WINDOW_DAYS,
    max_sequence_days: int = DEFAULT_MAX_SEQUENCE_DAYS,
    
):
  """Convert configure(raw) data frames into feature tables 
    - pat_view: patient demographics
  """

  prog_bar.update(10); prog_bar.set_description_str("Building screening scores..")
  # phq9, phq2, gad7, auditc, dast10 = get_screening_scores(obs_view)
  df_obs_screen_scores = get_screening_scores(obs_view)
  print("Retrieved screening scores from observations.csv")

  smoke, preg, alcohol = extract_life_obs(obs_view)
  smoke = one_hot(smoke, col="bucket", prefix="smoke", classes=("never", "former", "current"))
  alcohol = one_hot(alcohol, col="bucket", prefix="alcohol", classes=("none", "moderate", "heavy"))
  print("Retrieved lifestyle choices from observations.csv")

  med_view["is_antidepressant"] = med_view["description"].astype(str).str.contains(ANTIDEPRESSANT_REGEX)
  ad_med = med_view.loc[med_view["is_antidepressant"]].copy()
  ad_cov = build_coverage_calendar(ad_med)
  ad_daily = add_pdc_and_gaps(ad_cov, pdc_window_days).reset_index(drop=True)

  daily = build_daily_features(
    pat_view, enc_view, cnd_view, med_view, obs_view,
    smoke, preg, alcohol, ad_daily,
    phq9, phq2, gad7, auditc, dast10,
    max_sequence_days, pdc_window_days
  )
  daily_keys = daily[["patient_id", "date"]].drop_duplicates()

  events = build_event_stream(cnd_view, med_view, enc_view, obs_view).merge(daily_keys, on=["patient_id","date"], how="inner")
  enc_f = (enc_view
    .assign(date=enc_view["start_time"].dt.floor("D"))
    .merge(daily_keys, on=["patient_id", "date"], how="inner")
  )
  cond_f = (cnd_view
    .assign(date=cnd_view["start_time"].dt.floor("D"))
    .merge(daily_keys, on=["patient_id", "date"], how="inner")
  )

  daily = clean_daily(pat_view, enc_f, cond_f, med_view, obs_view, daily, events)
  return daily, events, pat_view, enc_f, cond_f, med_view, obs_view


def write_feature_outputs(out_dir: Path, daily, events, patients, encounters, conditions, med, obs):
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  def write(df: pd.DataFrame, name: str):
    fp = out_dir / name
    df.to_csv(fp, index=False, compression="gzip")

  out_dir.mkdir(parents=True, exist_ok=True)
  write(daily, "daily.csv")
  write(events, "events.csv")
  write(patients, "patients.csv")
  write(encounters, "encounters.csv")
  write(conditions, "conditions.csv")
  write(med, "medications.csv")
  write(obs, "observations.csv")


def build_features(
  raw_dir: Path,
  out_dir: Path,
  pdc_window_days: int = DEFAULT_PDC_WINDOW_DAYS,
  max_sequence_days: int = DEFAULT_MAX_SEQUENCE_DAYS,
):
  prog_bar = tqdm(total=100)
  
  prog_bar.set_description_str("Loading tables")
  raw = load_raw_tables(raw_dir)

  prog_bar.set_description_str("Configuring tables")
  pat_view, enc_view, cnd_view, med_view, obs_view = configure_tables(
    raw["patients"], raw["encounters"], raw["conditions"], raw["medications"], raw["observations"])

  daily, events, patients, encounters, conditions, med, obs = build_feature_tables(
    prog_bar,
    raw["patients"], raw["encounters"], raw["conditions"], raw["medications"], raw["observations"],
    pdc_window_days=pdc_window_days,
    max_sequence_days=max_sequence_days,
  )
  write_feature_outputs(out_dir, daily, events, patients, encounters, conditions, med, obs)
  return daily, events, patients, encounters, conditions, med, obs
