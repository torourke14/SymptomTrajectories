from pathlib import Path
import pandas as pd
from ..build_features.constants import ENCOUNTER_TYPES


def load_raw_tables(base_dir: Path):
    """
    Load raw Synthea data.
    - Parse dates, convert text to lower
    """

    def _find_table(name: str):
        base = Path(base_dir)
        for ext in (".csv.gz", ".csv"):
            path = base / f"{name}{ext}"
            if path.exists():
                return path
        return None
    
    raw = {}
    for name in ["patients", "encounters", "conditions", "medications", "observations"]:
        path = _find_table(name)
        if not path:
            raise FileNotFoundError(f"Missing {name} under {base_dir}")
        raw[name] = pd.read_csv(path, compression="infer", low_memory=False)

    return raw


def configure_tables(
    patients: pd.DataFrame, 
    encounters: pd.DataFrame, 
    conditions: pd.DataFrame,
    medications: pd.DataFrame, 
    observations: pd.DataFrame
):
    """ Collect and configure all data tables for processing
        - select necessary columns
        - rename more appropriately
        - map certain class values to processable sets (e.g, reduce encounter classes)
        - ..and most importantly.. CAST TO CORRECT DATATYPES
    """
    # -------------------------
    # Patients
    # -------------------------
    pat = (
        patients.loc[:, ["id", "birthdate", "deathdate", "race", "ethnicity", "gender"]]
        .rename(columns={
            "id":"patient_id",
            "gender":"sex",
            "birthdate": "date_birth",
            "deathdate": "date_death"
        })
        .copy(deep=True)
    )
    pat["sex"] = (patients["sex"]
        .astype("string")
        .str.upper()
        .map({"M": "M", "F": "F"})
        .fillna("UNK")
    )
    pat["race"] = patients["race"].astype("string").fillna("UNK")
    pat["ethnicity"] = pat["ethnicity"].astype("string")
    pat["date_birth"] = pd.to_datetime(pat["date_birth"], errors="coerce").dt.tz_localize(None)
    pat["date_death"] = pd.to_datetime(pat["date_death"], errors="coerce").dt.tz_localize(None)
    pat = pat.astype({
        "patient_id": "string",
        "sex": "string",
        "race": "string",
        "ethnicity": "string",
        "date_birth": "datetime64[ns]",
        "date_death": "datetime64[ns]",
    })

    # -------------------------
    # Encounters
    # -------------------------
    enc = (
        encounters.loc[:, ["id", "patient", "start", "stop", "encounterclass", "code", "description", "payer"]]
        .rename(columns={
            "patient":"patient_id", 
            "start":"start_time", 
            "stop":"stop_time", 
            "id":"encounter_id" 
        })
        .copy(deep=True)
    )
    enc["start_time"] = pd.to_datetime(enc["start_time"], errors="coerce").dt.tz_localize(None)
    enc["stop_time"] = pd.to_datetime(enc["stop_time"], errors="coerce").dt.tz_localize(None)
    enc["enc_class_bucket"] = enc["encounterclass"].map(ENCOUNTER_TYPES).fillna("OTHER")
    enc = enc.astype({
        "patient_id": "string",
        "encounter_id": "string",
        "start_time": "datetime64[ns]",
        "stop_time": "datetime64[ns]",
        "encounterclass": "string",
        "enc_class_bucket": "category",
        "code": "string",
        "description": "string",
        "payer": "string",
    })

    # -------------------------
    # Conditions
    # -------------------------
    cond = (
        conditions.loc[:, ["patient", "encounter", "start", "stop", "code", "description"]]
        .rename(columns={
            "patient": "patient_id",
            "start": "start_time",
            "stop": "stop_time",
            "encounter": "encounter_id",
            "code": "snomed_code"
        })
        .copy(deep=True)
    )
    cond["start_time"] = pd.to_datetime(cond["start_time"], errors="coerce").dt.tz_localize(None)
    cond["stop_time"] = pd.to_datetime(cond["stop_time"], errors="coerce").dt.tz_localize(None)
    cond = cond.astype({
        "patient_id": "string",
        "encounter_id": "string",
        "start_time": "datetime64[ns]",
        "stop_time": "datetime64[ns]",
        "snomed_code": "string",
        "description": "string",
    })

    # -------------------------
    # Medications
    # -------------------------
    med = (
        medications.loc[:, ["patient", "encounter", "start", "stop", "code", "description", 
                            "dispenses", "totalcost", "reasoncode", "reasondescription"]]
        .rename(columns={
            "patient": "patient_id",
            "start": "start_time",
            "stop": "stop_time",
            "encounter": "encounter_id",
            "code": "rx_code"
        })
        .copy(deep=True)
    )      
    med["start_time"] = pd.to_datetime(med["start_time"], errors="coerce").dt.tz_localize(None)
    med["stop_time"] = pd.to_datetime(med["stop_time"], errors="coerce").dt.tz_localize(None)
    med["dispenses"] = pd.to_numeric(med["dispenses"], errors="coerce")
    med["totalcost"] = pd.to_numeric(med["totalcost"], errors="coerce")
    med = med.astype({
        "patient_id": "string",
        "encounter_id": "string",
        "start_time": "datetime64[ns]",
        "stop_time": "datetime64[ns]",
        "rx_code": "string",
        "description": "string",
        "reasoncode": "string",
        "reasondescription": "string",
        "dispenses": "Int64",
        "totalcost": "Float64",
    })

    # -------------------------
    # Observations
    # -------------------------
    obs = (observations.loc[:, ["date", "patient", "encounter", "category", "code",
                        "description", "value", "units", "type"]]
        .rename(columns={
            "patient": "patient_id",
            "date": "obs_time",
            "code": "loinc_code",
        })
        .copy(deep=True)
    )
    obs["obs_date"] = pd.to_datetime(obs["obs_time"], errors="coerce").dt.tz_localize(None).dt.floor("D")
    obs["val_num"] = pd.to_numeric(obs["value"], errors="coerce")
    obs = obs.astype({
        "patient_id": "string",
        "encounter_id": "string",
        "obs_date": "datetime64[ns]",
        "category": "string",
        "loinc_code": "string",
        "description": "string",
        "value": "string",
        "units": "string",
        "type": "string",
        "val_str": "string",
        "val_num": "Float64",
    })

    # -------------------------
    # convert to lowercacse strings
    # -------------------------
    def to_lower(df: pd.DataFrame) -> pd.DataFrame:
        string_cols = df.select_dtypes(include=["string", "object"]).columns
        df[string_cols] = df[string_cols].apply(lambda s: s.str.strip().lower())
        return df
    pat = to_lower(pat)
    enc = to_lower(enc)
    cond = to_lower(cond)
    med = to_lower(med)
    obs = to_lower(obs)

    # -------------------------
    # Drop data with any 'NA's' or 'UNK'
    # -------------------------
    def drop4_any_na(df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop any column that contains at least one missing value (NaN/NaT/<NA>/None).
        """
        keep_cols = df.columns[df.notna().all(axis=0)]
        return df.loc[:, keep_cols]
    pat = drop4_any_na(pat)
    enc = drop4_any_na(enc)
    cond = drop4_any_na(cond)
    med = drop4_any_na(med)
    obs = drop4_any_na(obs)

    return pat, enc, cond, med, obs
