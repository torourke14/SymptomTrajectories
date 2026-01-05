import numpy as np
import pandas as pd
from .constants import SMOKING_STATUS_CODE, SMOKER_COLLAPSE, PREGNANCY_CODES, ALCOHOL_COLLAPSE


def extract_life_obs(obs_in: pd.DataFrame):
    obs = obs_in.copy()
    obs["patient_id"]  = obs["patient_id"].astype(str)
    obs["obs_time"]    = pd.to_datetime(obs["obs_time"], errors="coerce")
    obs["date"]        = obs["obs_time"].dt.floor("D")
    obs["desc_lc"]     = obs["description"].astype(str).str.lower().str.strip()
    obs["val_str"]     = obs["value"].astype(str).str.lower().str.strip()
    obs["val_num"]     = pd.to_numeric(obs["value"], errors="coerce")

    def last_daily_rec(rec):
        if rec.empty:
            return rec
        return (rec.dropna(subset=["date"])
            .sort_values(["patient_id", "date", "obs_time"])
            .groupby(["patient_id", "date"], as_index=False)
            .tail(1))

    smok = last_daily_rec(
        obs.loc[obs["loinc_code"].eq(SMOKING_STATUS_CODE), ["patient_id", "date", "obs_time", "val_str"]]
    )
    smoke_df = (smok
        .assign(bucket = smok["val_str"].replace(SMOKER_COLLAPSE))
        .rename(columns={"val_str": "smoking_status"})
        [["patient_id", "date", "bucket"]]
    )


    preg = last_daily_rec(obs.loc[
        obs["loinc_code"].isin(PREGNANCY_CODES), ["patient_id", "date", "obs_time","val_str"]]
    )
    preg_df = preg.assign(pregnancy_pos=preg["val_str"].str.contains("pos", na=False).astype("int8"))[
        ["patient_id","date","pregnancy_pos"]
    ]

    alc = last_daily_rec(obs.loc[
    obs["loinc_code"].eq("75626-2"),
    ["patient_id",  "date", "obs_time", "val_num"]]
    )
    if alc.empty:
        a_txt = last_daily_rec(obs.loc[
            obs["desc_lc"].isin(ALCOHOL_COLLAPSE.keys()) | obs["val_str"].isin(ALCOHOL_COLLAPSE.keys()),
            ["patient_id", "date", "obs_time", "desc_lc", "val_str"]])
        if a_txt.empty:
            alcohol_df = pd.DataFrame(columns=["patient_id", "date", "bucket"])
        else:
            raw = a_txt["val_str"].where(a_txt["val_str"].isin(ALCOHOL_COLLAPSE), a_txt["desc_lc"])
            alcohol_df = (a_txt
                .assign(bucket = raw.replace(ALCOHOL_COLLAPSE))
                [["patient_id",  "date", "bucket"]]
            )
    else:
        alcohol_df = (alc
            .assign(
            bucket=pd.cut(alc["val_num"],
            bins=[-np.inf, 0, 3, np.inf],
            labels=["none", "moderate", "heavy"])
            )
            [["patient_id", "date", "bucket"]]
        )

    return smoke_df, preg_df, alcohol_df
