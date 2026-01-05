import json
import numpy as np
import pandas as pd


def encode_codes_factorize(events):
  keys = events["event_type"] + "|" + events["code"]
  ids, uniques = pd.factorize(keys, sort=True)
  ids = (ids.astype(np.int32) + 1) if ids.size else ids

  freq = pd.Series(keys).value_counts()
  cookbook = (pd.DataFrame({"key": uniques})
    .assign(id=lambda d: np.arange(1, len(d)+1, dtype=np.int32))
    .assign(type=lambda d: d["key"].str.split("|", n=1).str[0],
            code=lambda d: d["key"].str.split("|", n=1).str[1],
            freq=lambda d: d["key"].map(freq).fillna(0).astype(int))
    [["id", "type", "code", "key", "freq"]]
  )

  events = events.assign(code_id=ids) if ids.size else events.assign(code_id=np.array([], dtype=np.int32))

  def _uniq_json(s):
    return json.dumps(sorted(set(map(int, s.tolist()))))

  per_day = (events
    .groupby(["patient_id", "date"], observed=True)["code_id"]
      .apply(_uniq_json)
      .reset_index()
      .rename(columns={"code_id": "code_ids"})
      if len(events) else pd.DataFrame(columns=["patient_id","date","code_ids"])
  )
  return per_day, cookbook
