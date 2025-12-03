

# Imports for OSX plotting only:
import matplotlib
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
matplotlib.use('TkAgg')
# - END -

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import matplotlib.pyplot as plt


station = "KG_Arabel"
station = "KG_BS-old"
station = "KG_BS-new" # moved station 2025-08-11 06:00:00+00:00 >> BS_new-1 and BS_new-2
station = "KG_G354-old"
station = "KG_G354-new"


# INPUT
filename = f"/Users/pohle/Dropbox/Central_Asia/DISCHARGE/L3/{station}_cat_df_withQv2.csv"
filename_mini = f"/Users/pohle/Dropbox/Central_Asia/DISCHARGE/L3/{station}_cat_df_WL-Q.csv"

wl_q_df_mini = pd.read_csv(filename_mini)
# wl_q_df_mini = wl_q_df_mini[['datetime','wl_final', 'discharge']]
wl_q_df_mini["datetime"] = pd.to_datetime(wl_q_df_mini["datetime"])

wl_q_df_full = pd.read_csv(filename)
# wl_q_df_full = wl_q_df_full[['datetime','wl_final', 'discharge']]
wl_q_df_full["datetime"] = pd.to_datetime(wl_q_df_full["datetime"])

# OUTPUT
output_dir = f"/Users/pohle/Dropbox/Central_Asia/DISCHARGE/L4/"
output_dir_figure = f"/Users/pohle/Dropbox/Central_Asia/DISCHARGE/figures/"

def make_H0_grid_from_data(H0_min, H0_max, n=200):
    return np.linspace(H0_min, H0_max, n)


@dataclass
class RatingCurveParams:
    H0: float
    a: float
    b: float


class RatingCurve:
    """
    Power-law rating curve: Q = a * (H - H0)**b

    H is wl_final, Q is discharge.
    """
    def __init__(self,
                 params: RatingCurveParams,
                 rel_q_uncertainty: float = 0.3):
        """
        rel_q_uncertainty: relative discharge uncertainty (e.g. 0.3 for ±30%).
        """
        self.params = params
        self.rel_q_uncertainty = rel_q_uncertainty

    # ---------- fitting internals ----------

    @staticmethod
    def _fit_for_fixed_H0(H: np.ndarray,
                          Q: np.ndarray,
                          H0: float,
                          min_valid_frac: float = 0.8
                          ) -> Optional[RatingCurveParams]:
        """
        Fit log(Q) = b*log(H-H0) + log(a) for a fixed H0.

        Returns RatingCurveParams or None if too few/too many points dropped.
        """
        x = H - H0
        mask = (x > 0) & (Q > 0)
        valid_frac = mask.mean()

        # Require that we keep a decent fraction of points
        if valid_frac < min_valid_frac or mask.sum() < 3:
            return None

        x_log = np.log(x[mask])
        y_log = np.log(Q[mask])

        A = np.vstack([x_log, np.ones_like(x_log)]).T
        b, c = np.linalg.lstsq(A, y_log, rcond=None)[0]
        a = float(np.exp(c))

        return RatingCurveParams(H0=H0, a=a, b=float(b))

    def fit_from_dataframe(df: pd.DataFrame,
                           h_col: str = "wl_final",
                           q_col: str = "discharge",
                           H0_grid: Optional[np.ndarray] = None,
                           rel_q_uncertainty: float = 0.3,
                           min_valid_frac: float = 0.8
                           ) -> "RatingCurve":
        """
        Fit rating curve to data, searching over a grid of H0.

        If H0_grid is None, build a default band just below min(H).
        Otherwise, use the user-supplied H0_grid.
        """
        H = df[h_col].to_numpy(dtype=float)
        Q = df[q_col].to_numpy(dtype=float)

        H_min = np.nanmin(H)
        H_max = np.nanmax(H)

        if H0_grid is None:
            # default band just below H_min
            delta = max(0.05 * (H_max - H_min), 0.2)
            eps = 0.01
            H0_low = H_min - delta
            H0_high = H_min - eps
            H0_grid = np.linspace(H0_low, H0_high, 40)

        best_params: Optional[RatingCurveParams] = None
        best_ssr = np.inf

        for H0 in H0_grid:
            params = RatingCurve._fit_for_fixed_H0(
                H, Q, H0, min_valid_frac=min_valid_frac
            )
            if params is None:
                continue

            x = H - params.H0
            mask = (x > 0) & (Q > 0)
            x_log = np.log(x[mask])
            y_log = np.log(Q[mask])
            y_pred = params.b * x_log + np.log(params.a)

            ssr = float(np.sum((y_log - y_pred) ** 2))

            if ssr < best_ssr:
                best_ssr = ssr
                best_params = params

        if best_params is None:
            raise ValueError("Could not fit rating curve with given constraints.")

        return RatingCurve(best_params, rel_q_uncertainty=rel_q_uncertainty)

    # ---------- prediction ----------

    def predict(self,
                H: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict discharge for given water levels H.

        Returns (Q_mid, Q_low, Q_high) where Q_low/high implement
        the relative discharge uncertainty.
        """
        H = np.asarray(H, dtype=float)
        x = H - self.params.H0
        Q_mid = self.params.a * np.maximum(x, 0.0) ** self.params.b

        factor = self.rel_q_uncertainty
        Q_low = Q_mid * (1.0 - factor)
        Q_high = Q_mid * (1.0 + factor)
        return Q_mid, Q_low, Q_high

    def __call__(self,
                 H: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.predict(H)

    # ---------- plotting ----------

    def plot_fit(self,
                 df: pd.DataFrame,
                 h_col: str = "wl_final",
                 q_col: str = "discharge",
                 ax: Optional[plt.Axes] = None,
                 n_points: int = 100,
                 loglog: bool = False,
                 station: Optional[str] = None,
                 show_point_errors: bool = True):
        """
        Scatter observed H–Q and overlay fitted curve (+ envelope).
        Optionally show vertical error bars on gauging points.
        """
        if ax is None:
            fig, ax = plt.subplots()

        H_obs = df[h_col].to_numpy(float)
        Q_obs = df[q_col].to_numpy(float)

        # x-range for curve
        H_min, H_max = np.nanmin(H_obs), np.nanmax(H_obs)
        H_grid = np.linspace(H_min, H_max, n_points)
        Q_mid, Q_low, Q_high = self.predict(H_grid)

        # scatter observations
        ax.scatter(H_obs, Q_obs, s=20, alpha=0.7, label="gauging points")

        # optional error bars on points (vertical, from relative Q-uncertainty)
        if show_point_errors:
            q_err = self.rel_q_uncertainty * Q_obs
            ax.errorbar(H_obs, Q_obs,
                        yerr=q_err,
                        fmt="none",
                        ecolor="gray",
                        elinewidth=1,
                        capsize=2,
                        alpha=0.7)

        # curve and envelope
        ax.plot(H_grid, Q_mid, "r-", label="rating curve")
        ax.fill_between(H_grid, Q_low, Q_high,
                        color="r", alpha=0.2, label="±uncertainty")

        ax.set_xlabel("Water level [cm]")
        ax.set_ylabel("Discharge [m$^3$ s$^{-1}$]")

        if loglog:
            ax.set_xscale("log")
            ax.set_yscale("log")

        title = "Rating curve fit"
        if station is not None:
            title += f" – {station}"
        ax.set_title(title)

        ax.legend()
        return ax


def build_yearly_rating_curves(
        df: pd.DataFrame,
        datetime_col: str = "datetime",
        h_col: str = "wl_final",
        q_col: str = "discharge",
        min_points: int = 5,
        rel_q_uncertainty: float = 0.3,
        min_valid_frac: float = 0.8
    ) -> Dict[int, RatingCurve]:
    """
    For one station's wl_q_df, return a dict {year: RatingCurve},
    using the closest previous year with data if a year has too few points.
    """
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df["year"] = df[datetime_col].dt.year

    curves: Dict[int, RatingCurve] = {}
    years = sorted(df["year"].unique())

    last_curve: Optional[RatingCurve] = None

    for year in years:
        df_y = df[df["year"] == year]

        if df_y.shape[0] >= min_points:
            try:
                curve = RatingCurve.fit_from_dataframe(
                    df_y,
                    h_col=h_col,
                    q_col=q_col,
                    rel_q_uncertainty=rel_q_uncertainty,
                    min_valid_frac=min_valid_frac
                )
                curves[year] = curve
                last_curve = curve
            except ValueError:
                if last_curve is not None:
                    curves[year] = last_curve
        else:
            if last_curve is not None:
                curves[year] = last_curve

    return curves



def process_station(
    station: str,
    wl_q_df_mini: pd.DataFrame,
    wl_q_df_full: pd.DataFrame,
    years: list[int],
    H0_bounds_min: dict,
    H0_bounds_max: dict,
    min_points: int = 5,
    rel_q_uncertainty: float = 0.3,
    output_dir: str = ".",
):
    # Ensure datetime and year columns
    for df in (wl_q_df_mini, wl_q_df_full):
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["year"] = df["datetime"].dt.year

    last_curve: Optional[RatingCurve] = None

    for year in years:
        df_year_pairs = wl_q_df_mini[wl_q_df_mini["year"] == year]
        df_year_full = wl_q_df_full[wl_q_df_full["year"] == year]

        if df_year_full.empty:
            continue

        curve: Optional[RatingCurve] = None

        if len(df_year_pairs) >= min_points:
            # --- H0 bounds for this station/year ---
            H0_min_global = H0_bounds_min.get(station, None)
            H0_max_global = H0_bounds_max.get(station, None)

            wl_min_flow = df_year_full["wl_final"].min()

            if H0_max_global is None:
                H0_max = wl_min_flow
            else:
                H0_max = min(H0_max_global, wl_min_flow)

            if H0_min_global is None:
                H0_min = H0_max - 40.0
            else:
                H0_min = H0_min_global

            H0_grid = np.linspace(H0_min, H0_max, 200)

            try:
                curve = RatingCurve.fit_from_dataframe(
                    df_year_pairs,
                    h_col="wl_final",
                    q_col="discharge",
                    rel_q_uncertainty=rel_q_uncertainty,
                    min_valid_frac=0.5,
                    H0_grid=H0_grid,
                )
                last_curve = curve
            except ValueError:
                curve = last_curve
        else:
            curve = last_curve

        if curve is None:
            continue

        # ---------- NEW: save rating-curve plot ----------
        if not df_year_pairs.empty:
            fig, ax = plt.subplots()
            curve.plot_fit(
                df_year_pairs,
                ax=ax,
                loglog=False,
                station=station,
            )
            fig_path = f"{output_dir_figure}/{station}_rating-curve_{year}.pdf"
            fig.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)
        # fig, ax = plt.subplots()
        # curve.plot_fit(
        #     df_year_pairs,
        #     ax=ax,
        #     loglog=False,
        #     station=station,        # station name in title
        # )
        # fig_path = f"{output_dir_figure}/{station}_rating-curve_{year}.pdf"
        # fig.savefig(fig_path, bbox_inches="tight")
        # plt.close(fig)
        # -----------------------------------------------

        # Apply curve to full WL time series for that year
        H = df_year_full["wl_final"].to_numpy(float)
        Q_mid, Q_low, Q_high = curve.predict(H)

        df_out = df_year_full.copy()
        df_out["Q_mid"] = Q_mid
        df_out["Q_low"] = Q_low
        df_out["Q_high"] = Q_high

        # Format datetime as "YYYY-MM-DDTHH:MM:SS+00:00"
        df_out["datetime"] = pd.to_datetime(df_out["datetime"])
        if getattr(df_out["datetime"].dt, "tz", None) is not None:
            dt_utc = df_out["datetime"].dt.tz_convert("UTC")
        else:
            dt_utc = df_out["datetime"]
        df_out["datetime"] = dt_utc.dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

        out_path = f"{output_dir}/{station}_{year}_wl_q_data.csv"
        out_path_nice = f"{output_dir}/{station}_{year}_wl_q_data_clean.csv"
        # Drop "year" column
        df_out.drop(columns="year", inplace=True)
        df_out_nice = df_out.rename(columns={
            "datetime" : "datetime (UTC+0)",
            "pressure" : "pressure_water (cmH2O)",
            "temp" : "temperature_water (deg.C)",
            "pressure_atmo" : "pressure_atmo (cmH2O)",
            "temp_atmo" : "temperature_atmo (deg.C)",
            "wl_corr" : "wl_corr (cmH2O)",
            "wl_final" :"wl_final (cmH2O)",
            "Q_mid" : "Q_mid (m^3 s^{-1})",
            "Q_low" : "Q_low (m^3 s^{-1})",
            "Q_high" :"Q_high (m^3 s^{-1})"
        })
        df_out.to_csv(out_path, index=False)
        df_out_nice.to_csv(out_path_nice, index=False)


def plot_station_year_timeseries(
    station: str,
    year: int,
    df_year: pd.DataFrame,
    output_dir: str = ".",
):
    # ensure datetime index
    df = df_year.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    t = df["datetime"]
    q_mid = df["Q_mid"]
    q_low = df["Q_low"]
    q_high = df["Q_high"]

    fig, ax = plt.subplots(figsize=(10, 4))

    # middle estimate
    LW = 0.4
    ax.plot(t, q_mid, color="C0", label="Q_mid", linewidth=LW)

    # polygon band (upper/lower boundary)
    ax.fill_between(t, q_low, q_high, color="C0", alpha=0.2,
                    label="Q_low / Q_high")

    ax.set_xlabel("Time")
    ax.set_ylabel("Discharge [m$^3$ s$^{-1}$]")
    ax.set_title(f"{station} – discharge time series {year}")
    ax.legend()
    fig.autofmt_xdate()

    out_path = f"{output_dir_figure}/{station}_timeseries_{year}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def plot_station_all_years_timeseries(
    station: str,
    all_year_dfs: list[pd.DataFrame],
    output_dir: str = ".",
):
    if not all_year_dfs:
        return

    # concatenate all years for this station
    df = pd.concat(all_year_dfs, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    t = df["datetime"]
    q_mid = df["Q_mid"]
    q_low = df["Q_low"]
    q_high = df["Q_high"]

    fig, ax = plt.subplots(figsize=(12, 4))
    LW= 0.4
    ax.plot(t, q_mid, color="C0", label="Q_mid", linewidth=LW)
    ax.fill_between(t, q_low, q_high, color="C0", alpha=0.2,
                    label="Q_low / Q_high")

    ax.set_xlabel("Time")
    ax.set_ylabel("Discharge [m$^3$ s$^{-1}$]")
    ax.set_title(f"{station} – discharge time series (all years)")
    ax.legend()
    fig.autofmt_xdate()

    out_path = f"{output_dir_figure}/{station}_timeseries_all-years.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def plot_station_all_years_timeseries_shifted(
    station: str,
    years: list[int],
    output_dir: str = ".",
):
    dfs = []
    for year in years:
        path = f"{output_dir}/{station}_{year}_wl_q_data_shifted.csv"
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            continue
        dfs.append(df)

    if not dfs:
        return

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["datetime"] = pd.to_datetime(df_all["datetime"])
    df_all = df_all.sort_values("datetime")

    t = df_all["datetime"]
    q_mid = df_all["Q_mid"]
    q_low = df_all["Q_low"]
    q_high = df_all["Q_high"]

    fig, ax = plt.subplots(figsize=(12, 4))
    LW = 0.4
    ax.plot(t, q_mid, color="C0", label="Q_mid", linewidth=LW)
    ax.fill_between(t, q_low, q_high, color="C0", alpha=0.2,
                    label="Q_low / Q_high")

    ax.set_xlabel("Time")
    ax.set_ylabel("Discharge [m$^3$ s$^{-1}$]")
    ax.set_title(f"{station} – discharge time series (all years, shifted)")
    ax.legend()
    fig.autofmt_xdate()

    out_path = f"{output_dir}/{station}_timeseries_all-years_shifted.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

# -------- Process loop ---------

years = list(range(2019, 2026))
stations = ["KG_G354-new","KG_G354-old","KG_BS-old","KG_BS-new","KG_Arabel"]

H0_bounds_min = {
    "KG_G354-new": -10.0,
    "KG_G354-old": -30.0,
    "KG_BS-old":   -15.0,
    "KG_BS-new":  -15.0,
    "KG_Arabel":   -25.0,
}
H0_bounds_max = {
    "KG_G354-new": 15.0,
    "KG_G354-old": 5.0,
    "KG_BS-old":   0.0,
    "KG_BS-new":  0.0,
    "KG_Arabel":   15.0,
}

for station in stations:
    # station = "KG_G354-new"
    filename_mini = f"/Users/pohle/Dropbox/Central_Asia/DISCHARGE/L3/{station}_cat_df_WL-Q.csv"
    filename_full = f"/Users/pohle/Dropbox/Central_Asia/DISCHARGE/L3/{station}_cat_df_withQv2.csv"

    wl_q_df_mini = pd.read_csv(filename_mini)#[["datetime", "wl_final", "discharge"]]
    wl_q_df_full = pd.read_csv(filename_full)#[["datetime", "wl_final"]]

    process_station(
        station=station,
        wl_q_df_mini=wl_q_df_mini,
        wl_q_df_full=wl_q_df_full,
        years=years,
        H0_bounds_min=H0_bounds_min,
        H0_bounds_max=H0_bounds_max,
        min_points=5,
        rel_q_uncertainty=0.3,
        output_dir=output_dir,
    )

# Figures
years = list(range(2019, 2026))
stations = ["KG_G354-new","KG_G354-old","KG_BS-old","KG_BS-new","KG_Arabel"]


for station in stations:
    # station = "KG_G354-new"
    year_dfs = []
    for year in years:
        path = f"{output_dir}/{station}_{year}_wl_q_data.csv"
        try:
            df_year = pd.read_csv(path)
        except FileNotFoundError:
            continue

        # 1) individual year plot
        plot_station_year_timeseries(station, year, df_year, output_dir)

        # collect for all-years plot
        year_dfs.append(df_year)

    # 2) combined all-years plot
    plot_station_all_years_timeseries(station, year_dfs, output_dir)

# Special case Arabel: New stage level - discharge measurement in 2025 suggests reorganization of river bed
#with an increase in elevation of the river bed.
#
# Base level shift ∆H:
# ΔH=H2025 − Hequiv,old


station = "KG_Arabel"

# read mini and full data for this station
filename_mini = f"/Users/pohle/Dropbox/Central_Asia/DISCHARGE/L3/{station}_cat_df_WL-Q.csv"
filename_full = f"/Users/pohle/Dropbox/Central_Asia/DISCHARGE/L3/{station}_cat_df_withQv2.csv"

wl_q_df_mini = pd.read_csv(filename_mini)#[["datetime", "wl_final", "discharge"]]
wl_q_df_full = pd.read_csv(filename_full)#[["datetime", "wl_final"]]

wl_q_df_mini["datetime"] = pd.to_datetime(wl_q_df_mini["datetime"])
wl_q_df_full["datetime"] = pd.to_datetime(wl_q_df_full["datetime"])

wl_q_df_mini["year"] = wl_q_df_mini["datetime"].dt.year
wl_q_df_full["year"] = wl_q_df_full["datetime"].dt.year

# use all pairs BEFORE 2025
df_old_pairs = wl_q_df_mini[wl_q_df_mini["year"] < 2025]

print("KG_Arabel pre-2025 pairs:", len(df_old_pairs))

# choose H0 bounds for Arabel old curve (you can tune these)
H0_min = -25.0
H0_max =  15.0
H0_grid = np.linspace(H0_min, H0_max, 200)

curve_old = RatingCurve.fit_from_dataframe(
    df_old_pairs,
    h_col="wl_final",
    q_col="discharge",
    rel_q_uncertainty=0.1,
    min_valid_frac=0.5,
    H0_grid=H0_grid,
)

print("Old curve (pre-2025) params:")
print("  H0_old:", curve_old.params.H0)
print("  a_old:",  curve_old.params.a)
print("  b_old:",  curve_old.params.b)


# 2025 single gauging
df_2025_pairs = wl_q_df_mini[wl_q_df_mini["year"] == 2025]
print("KG_Arabel 2025 pairs:", len(df_2025_pairs))

if len(df_2025_pairs) != 1:
    raise ValueError("Expected exactly one 2025 pair for KG_Arabel.")

H_2025 = float(df_2025_pairs["wl_final"].iloc[0])
Q_2025 = float(df_2025_pairs["discharge"].iloc[0])

H0_old = curve_old.params.H0
a_old  = curve_old.params.a
b_old  = curve_old.params.b

# equivalent old-stage producing Q_2025
H_equiv_old = H0_old + (Q_2025 / a_old) ** (1.0 / b_old)

# stage shift
dH = H_2025 - H_equiv_old
print("H_2025:", H_2025)
print("H_equiv_old:", H_equiv_old)
print("ΔH (2025 - old):", dH)

from dataclasses import replace

H0_new = H0_old + dH
params_2025 = RatingCurveParams(H0=H0_new, a=a_old, b=b_old)
curve_2025 = RatingCurve(params_2025, rel_q_uncertainty=0.1)

print("2025 shifted curve params:")
print("  H0_new:", curve_2025.params.H0)
print("  a_new:",  curve_2025.params.a)
print("  b_new:",  curve_2025.params.b)


# output_dir = "/path/to/output"
years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]  # whatever applies

for year in years:
    df_year_full = wl_q_df_full[wl_q_df_full["year"] == year]
    if df_year_full.empty:
        continue

    if year < 2025:
        curve_year = curve_old
    else:
        curve_year = curve_2025

    H = df_year_full["wl_final"].to_numpy(float)
    Q_mid, Q_low, Q_high = curve_year.predict(H)

    df_out = df_year_full.copy()
    df_out["Q_mid"] = Q_mid
    df_out["Q_low"] = Q_low
    df_out["Q_high"] = Q_high

    # datetime formatting
    df_out["datetime"] = pd.to_datetime(df_out["datetime"])
    if getattr(df_out["datetime"].dt, "tz", None) is not None:
        dt_utc = df_out["datetime"].dt.tz_convert("UTC")
    else:
        dt_utc = df_out["datetime"]
    df_out["datetime"] = dt_utc.dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    # Drop year column
    df_out.drop(columns="year", inplace=True)
    # Drop "year" column
    df_out_nice = df_out.rename(columns={
        "datetime": "datetime (UTC+0)",
        "pressure": "pressure_water (cmH2O)",
        "temp": "temperature_water (deg.C)",
        "pressure_atmo": "pressure_atmo (cmH2O)",
        "temp_atmo": "temperature_atmo (deg.C)",
        "wl_corr": "wl_corr (cmH2O)",
        "wl_final": "wl_final (cmH2O)",
        "Q_mid": "Q_mid (m^3 s^{-1})",
        "Q_low": "Q_low (m^3 s^{-1})",
        "Q_high": "Q_high (m^3 s^{-1})"
    })
    out_path_nice = f"{output_dir}/{station}_{year}_wl_q_data_shifted_clean.csv"
    df_out_nice.to_csv(out_path_nice, index=False)

    out_path = f"{output_dir}/{station}_{year}_wl_q_data_shifted.csv"
    df_out.to_csv(out_path, index=False)

    # optional: rating-curve plot for this year
    df_pairs_this = df_old_pairs if year < 2025 else df_2025_pairs
    if not df_pairs_this.empty:
        fig, ax = plt.subplots()
        curve_year.plot_fit(df_pairs_this, ax=ax, loglog=False, station=station)
        fig.savefig(f"{output_dir_figure}/{station}_rating-curve_{year}_shifted.pdf",
                    bbox_inches="tight")
        plt.close(fig)

# Replot entire time series
import pandas as pd
import matplotlib.pyplot as plt

station = "KG_Arabel"
years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
# output_dir = "/path/to/output"
plot_station_all_years_timeseries_shifted(station, years, output_dir)




