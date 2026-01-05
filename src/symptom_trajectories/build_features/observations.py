import pandas as pd
import re
from .constants import OBS_LOINC_CODES, SCREENING_KWS, ANTIDEPRESSANT_REGEX, DEFAULT_PDC_WINDOW_DAYS, DEFAULT_MAX_SEQUENCE_DAYS


def extract_observations(obs_view: pd.DataFrame):

    return


def get_screening_scores(obs_view: pd.DataFrame, ):
    obs = obs_view[["patient_id", "obs_date", "loinc_code", "description", "value"]].copy()

    # create NA column for screen type to fill in
    obs["obs_type"] = pd.NA

    # assign by LOINC code match
    for screen_key, code in OBS_LOINC_CODES.items():
        obs.loc[obs["loinc_code"].isin(code), "obs_type"] = screen_key

    # assign by keyword match ONLY where screen is still empty
    screen_empty = obs["screen"].isna()
    for screen_key, keywords in SCREENING_KWS.items():
        # pattern = r"(?<!\w)(?:%s)(?!\w)" % "|".join(map(re.escape, [k.lower() for k in keywords]))
        pattern = "|".join([re.escape(k) for k in keywords]) # "|".join(map(re.escape, keywords)),
        matches = obs["description"].str.contains(
        pattern, 
        case=False, na=False, regex=True
        )
        obs.loc[matches & screen_empty, "screen"] = screen_key
        screen_empty = obs["screen"].isna()  # refresh after assignments

    # 3) keep only desired columns + only rows that got assigned
    obs = obs.loc[
        obs["screen"].notna() & obs["obs_date"].notna() & obs["value"].notna(), 
        ["patient_id", "obs_date", "screen", "value"]
    ].copy()
    
    obs = (obs
        .sort_values(["patient_id", "date", "screen", "obs_date"])
        .rename()
    )
    obs["value"] = obs["value"].astype("float32")
    
    return obs[["patient_id", "date", "screen", "value"]]



