import pandas as pd
import numpy as np


# INPUT
# hourly data
path = "/Users/pohle/Dropbox/Central_Asia/DISCHARGE/L3/"
filename_wl = '/Users/pohle/Dropbox/Central_Asia/DISCHARGE/L3/KG_BS-old_cat_df.csv'
filename_q = "/Users/pohle/Dropbox/Central_Asia/DISCHARGE/Q-FL/KG_BatyshSook-old_Q-FL.xlsx"

# OUTPUT
filename_out = '/Users/pohle/Dropbox/Central_Asia/DISCHARGE/L3/KG_BS-old_cat_df_withQv2.csv'

# -------
df = pd.read_csv(filename_wl)
# parse datetime and use as index
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.set_index("datetime").sort_index()
start_date = df.index.min()
end_date   = df.index.max()
start_wl = df['wl_final'][df.index == start_date]


q = pd.read_excel(filename_q, sheet_name="clean")  # adjust filename[web:23]
# treat empty strings as NaN (optional, if your sheet has "" instead of real NaN)
q = q.replace("", pd.NA)

# drop rows where no discharge columns are available
q = q.dropna(
    subset=["discharge (calc) [m3/s]",
            "discharge (calc with extrapolation) [m3/s]"],
    how="all",
)

# ensure UTC datetime
q["dt_utc"] = pd.to_datetime(
    q["date"].astype(str) + " " + q["time (UTC+0)"].astype(str),
    format="%Y-%m-%d %H:%M:%S",
    utc=True,
)

# compute mean discharge (if two values are provided, e.g. in R from extrapolation)
q["discharge"] = q[
    ["discharge (calc) [m3/s]", "discharge (calc with extrapolation) [m3/s]"]
].mean(axis=1)

# Remove Q values if no WL is available within 30‑minute tolerance
tol = pd.Timedelta("30min")

mask = (
    (q["dt_utc"].dt.year.isin(df.index.year.unique())) &   # matching years
    (q["dt_utc"] >= start_date - tol) &
    (q["dt_utc"] <= end_date   + tol)
)

q = q.loc[mask].set_index("dt_utc").sort_index()
q = q.dropna(subset='discharge')


# union of hourly timestamps and discharge timestamps
new_index = df.index.union(q.index).sort_values()
df_interp = df.reindex(new_index).interpolate(method="time")
df_interp["discharge"] = q["discharge"]
result = df_interp.reset_index().rename(columns={"index": "datetime"})

# Remove Q data at start and end if outside of initial time period of WL
dates_before = result['datetime'][result['datetime'] < start_date]
hours_before = (start_date - dates_before)

result = result.sort_values("datetime").reset_index(drop=True)


# mask rows BEFORE the first hourly timestamp, within 30 minutes, with discharge
pre_mask = (
    (result["datetime"] < start_date) &
    (result["datetime"] >= start_date - tol) &
    (~result["discharge"].isna())
)

pre_vals = result.loc[pre_mask, "discharge"]

if not pre_vals.empty:
    pre_mean = pre_vals.mean()

    # locate the row in result that corresponds to start_date
    idx_start = result.index[result["datetime"] == start_date]

    if len(idx_start) == 1:
        idx_start = idx_start[0]
        # only overwrite if it is currently NaN
        if pd.isna(result.at[idx_start, "discharge"]):
            result.at[idx_start, "discharge"] = pre_mean

# Remove discharge values where wl_final is NaN and WL is NaN
mask_wl = ~result["wl_final"].isna()
result = result.loc[mask_wl] 

# to csv
result.to_csv(filename_out)
