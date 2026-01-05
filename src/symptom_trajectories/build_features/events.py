import pandas as pd


def build_event_stream(cond, med, enc, obs):
    events = []

    dx_cond = cond.copy()
    dx_cond["date"] = dx_cond["start_time"].dt.floor("D")
    events.append(
      dx_cond.assign(
        event_type="dx",
        code=dx_cond["snomed_code"].astype(str)
      )[["patient_id","date","event_type","code"]]
    )

    med = med.copy()
    med["date"] = med["start_time"].dt.floor("D")
    events.append(
      med.assign(
        event_type="med",
        code=med["rx_code"].astype(str)
      )[["patient_id","date","event_type","code"]]
    )

    enc_codes = enc.assign(date=enc["start_time"].dt.floor("D"))
    enc_codes["code"] = "ADM:" + enc_codes["enc_class_bucket"].astype(str)
    events.append(
      enc_codes[["patient_id", "date", "enc_class_bucket", "code"]]
        .rename(columns={"enc_class_bucket": "event_type"})
        .assign(event_type="adm")
    )

    obs_ev = obs.copy()
    obs_ev["date"] = obs_ev["obs_time"].dt.floor("D")
    obs_ev = obs_ev.assign(
      event_type="obs",
      code=obs_ev["loinc_code"].astype(str)
    )
    events.append(obs_ev[["patient_id","date","event_type","code"]])

    events = (
      pd.concat(events, ignore_index=True)
      .dropna(subset=["patient_id", "date", "event_type", "code"])
      .drop_duplicates()
    ).sort_values(["patient_id", "date", "event_type", "code"]).reset_index(drop=True)

    return events
