import pandas as pd
import numpy as np


def match_daypart(spot_row, programs_df, rate_card_df):
    """
    Find the selling daypart row for a spot.
    Priority:
      1. program row: same network & program_name == spot's program_name
      2. time row: weekday + minute within [start_time, end_time)
    Returns a rate-card row (Series) or None.
    """
    net = spot_row['network_code']
    # --- 1. program row ---
    # spot has telecast_id → join to programs to get episode_name; caller passes it
    prog_name = spot_row.get('episode_name')
    if prog_name:
        pr = rate_card_df[
            (rate_card_df.network_code == net)
            & (rate_card_df.program_name == prog_name)
        ]
        if len(pr) > 0:
            return pr.iloc[0]

    # --- 2. time row ---
    # broadcast weekday
    bcast_dt = spot_row['start_ts'] - pd.Timedelta(hours=6)
    weekday = bcast_dt.weekday()          # 0=Mon
    # minute since 06:00 of broadcast day
    minute = (spot_row['start_ts'] - bcast_dt.normalize()
              - pd.Timedelta(hours=6)).total_seconds() / 60
    minute = int(round(minute))

    def to_min(s):
        h, m = s.split(':')
        return int(h) * 60 + int(m) - 360   # 06:00→0, 25:37→1537-360

    cand = rate_card_df[
        (rate_card_df.network_code == net)
        & (rate_card_df.program_name.isna())          # only time rows
    ]
    for _, r in cand.iterrows():
        if weekday not in r['weekdays']:
            continue
        lo = to_min(r['start_time'])
        hi = to_min(r['end_time'])
        if lo <= minute <= hi:
            return r
    return None


def build_exposures(viewings_df, spots_df, people_df=None):
    """
    One row per (spot, person) where the person was watching the network
    at the spot's start_ts (session start <= spot start <= session end).
    persons only (R, L). Returns DataFrame with
      ad_airing_id, telecast_id, network_code, respondent_id, weight,
      + person attrs if provided.
    """
    v = viewings_df[viewings_df.resp_type.isin(['R', 'L'])].copy()
    if people_df is not None:
        v = v.merge(people_df, on='respondent_id', how='left')

    out = []
    # vectorised-ish: for each spot, filter once
    for _, s in spots_df.iterrows():
        m = (v.network_code == s.network_code) & \
            (v.start_ts <= s.start_ts) & (v.end_ts >= s.start_ts)
        sub = v[m]
        if len(sub) == 0:
            continue
        sub = sub.copy()
        sub['ad_airing_id'] = s.ad_airing_id
        sub['telecast_id'] = s.telecast_id
        out.append(sub[['ad_airing_id', 'telecast_id',
                        'network_code', 'respondent_id', 'weight',
                        'start_ts', 'end_ts']])
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)