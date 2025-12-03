import pandas as pd
import numpy as np


def write_to_csv(dataset, output_filename):
    import copy
    df_out = copy.copy(dataset)
    df_out['datetime'] = df_out['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S%z')
    df_out['datetime'] = df_out['datetime'].str.replace(r'([+-]\d{2})(\d{2})$', r'\1:\2', regex=True)
    df_out.to_csv(output_filename, index=False)

def write_wl_with_iso_tz(df, out_path):
    out = df.copy()
    # move index to column
    out = out.reset_index().rename(columns={"index": "datetime"})
    # format with offset like +0000 then insert colon
    out["datetime"] = out["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    out["datetime"] = out["datetime"].str.replace(
        r"([+-]\d{2})(\d{2})$", r"\1:\2", regex=True
    )
    out.to_csv(out_path, index=False)


stations = ["KG_G354-new","KG_G354-old", "KG_BS-old","KG_BS-new", "KG_Arabel"]

for station in stations:
    # station = "KG_BS-old"
    # station = "KG_BS-new" # moved station 2025-08-11 06:00:00+00:00 >> BS_new-1 and BS_new-2
    # station = "KG_Arabel"
    # station = "KG_G354-old"
    # station = "KG_G354-new"

    # INPUT
    path = "/Users/pohle/Dropbox/Central_Asia/DISCHARGE/L3/"
    filename_wl = f"/Users/pohle/Dropbox/Central_Asia/DISCHARGE/L3/{station}_cat_df.csv"
    filename_q  = f"/Users/pohle/Dropbox/Central_Asia/DISCHARGE/Q-FL/{station}_Q-FL.xlsx"

    # OUTPUT
    filename_out = f"/Users/pohle/Dropbox/Central_Asia/DISCHARGE/L3/{station}_cat_df_withQv2.csv"
    filename_out_mini = f"/Users/pohle/Dropbox/Central_Asia/DISCHARGE/L3/{station}_cat_df_WL-Q.csv"

    # -------------------------------
    # 1) Read hourly water-level data
    df = pd.read_csv(filename_wl)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()

    start_date = df.index.min()
    end_date   = df.index.max()

    # -------------------------------
    # 2) Read and clean discharge data
    q = pd.read_excel(filename_q, sheet_name="clean")
    q = q.replace("", pd.NA)

    # drop rows where both discharge columns are missing
    q = q.dropna(
        subset=["discharge (calc) [m3/s]",
                "discharge (calc with extrapolation) [m3/s]"],
        how="all",
    )

    # build UTC datetime
    q["dt_utc"] = pd.to_datetime(
        q["date"].astype(str) + " " + q["time (UTC+0)"].astype(str),
        format="%Y-%m-%d %H:%M:%S",
        utc=True,
    )

    # mean discharge of the two columns
    # q["discharge (calc) [m3/s]"]
    # q["discharge (calc with extrapolation) [m3/s]"]
    q["discharge"] = q[
        ["discharge (calc) [m3/s]",
         "discharge (calc with extrapolation) [m3/s]"]
    ].mean(axis=1)

    # keep only measurements in matching years and within ±30 min of WL period
    tol = pd.Timedelta("30min")
    mask_q = (
        q["dt_utc"].dt.year.isin(df.index.year.unique())
        & (q["dt_utc"] >= start_date - tol)
        & (q["dt_utc"] <= end_date + tol)
    )

    q = q.loc[mask_q].set_index("dt_utc").sort_index()
    q = q.dropna(subset=["discharge"])

    # -------------------------------
    # 3) Merge and interpolate onto combined time index
    new_index = df.index.union(q.index).sort_values()
    df_interp = df.reindex(new_index).interpolate(method="time")

    # put measured discharges at their timestamps
    df_interp["discharge"] = q["discharge"]

    # work in a datetime column for convenience
    result = df_interp.reset_index().rename(columns={"index": "datetime"})

    # -------------------------------
    # 4) If first WL timestep has NaN discharge,
    #    fill with mean of discharge values within 30 min *before* it
    result = result.sort_values("datetime").reset_index(drop=True)

    pre_mask = (
        (result["datetime"] < start_date)
        & (result["datetime"] >= start_date - tol)
        & (~result["discharge"].isna())
    )
    pre_vals = result.loc[pre_mask, "discharge"]

    if not pre_vals.empty:
        pre_mean = pre_vals.mean()

        idx_start = result.index[result["datetime"] == start_date]
        if len(idx_start) == 1:
            idx_start = idx_start[0]
            if pd.isna(result.at[idx_start, "discharge"]):
                result.at[idx_start, "discharge"] = pre_mean

    # -------------------------------
    # 5) Drop rows where no water level exists
    #    (and thus discharge should not be used)
    result = result.loc[~result["wl_final"].isna()]

    # drop all values with wl_final <=0
    result.loc[result["wl_final"] <= 0, "wl_final"] = pd.NA  # or np.nan
    # -------------------------------
    # 6) Export
    # result.to_csv(filename_out, index=False)
    # write_to_csv(result,filename_out)
    # after reading and preparing df
    write_wl_with_iso_tz(df, filename_out)  # original WL with ISO UTC timestamps

    # # only write filtered WL–Q pairs
    # result_filtered = result.loc[~result['discharge'].isna() & ~result['wl_final'].isna()]
    # write_wl_with_iso_tz(result_filtered, filename_out_mini)

    # only with entries where WL and Q are available
    result_filtered = result.loc[~result['discharge'].isna() & ~result['wl_final'].isna()]
    write_to_csv(result_filtered, filename_out_mini)