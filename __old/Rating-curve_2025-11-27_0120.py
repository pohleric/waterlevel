# Imports for OSX plotting only:
import matplotlib
matplotlib.use('TkAgg')
# - END -

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
import matplotlib.colors as mcolors
import matplotlib.cm as cm


station = "KG_BS-old"
station = "KG_BS-new" # moved station 2025-08-11 06:00:00+00:00 >> BS_new-1 and BS_new-2
station = "KG_Arabel"
station = "KG_G354-old"
station = "KG_G354-new"


# INPUT
filename = f"/Users/pohle/Dropbox/Central_Asia/DISCHARGE/L3/{station}_cat_df_withQv2.csv"
filename_mini = f"/Users/pohle/Dropbox/Central_Asia/DISCHARGE/L3/{station}_cat_df_WL-Q.csv"

# --- PARAMETERS ---

@dataclass
class RatingCurveParams:
    a: float      # scale in Q = a (h - h0)^b
    b: float      # exponent
    h0: float     # zero-flow level
    log_a: float  # convenience: log(a)


# --- HELPER FOR INITIAL h0 ---

def _initial_h0(wl: np.ndarray, q: np.ndarray) -> float:
    """Initial guess h0: min wl where Q>0."""
    mask = q > 0
    if not np.any(mask):
        raise ValueError("No positive discharges found for initial h0.")
    return float(np.min(wl[mask]))


# --- FITTING FUNCTION ---

def fit_power_law_rating_curve(
    wl: np.ndarray,
    q: np.ndarray,
    q_sigma: np.ndarray | None = None,
    h0_fixed: float | None = None,
    min_q_positive: float = 1e-6,
) -> RatingCurveParams:
    """
    Fit Q = a (h - h0)^b via weighted linear regression in log-space:

        log Q = log a + b log(h - h0)

    Algorithm:
    1) choose/estimate h0,
    2) keep h>h0 and Q>0,
    3) form x = log(h-h0), y = log(Q),
    4) do weighted least squares for y = c0 + c1 x.
    """
    wl = np.asarray(wl, dtype=float)
    q = np.asarray(q, dtype=float)

    # 1) choose h0
    if h0_fixed is None:
        h0 = _initial_h0(wl, q)
    else:
        h0 = float(h0_fixed)

    # 2) filter to h>h0 and Q>0
    mask = (wl > h0) & (q > min_q_positive)
    if np.sum(mask) < 3:
        raise ValueError("Not enough points above h0 with positive Q to fit rating curve.")

    wl_use = wl[mask]
    q_use = q[mask]

    # 3) log-transform
    x = np.log(wl_use - h0)
    y = np.log(q_use)

    # 4) weights from q_sigma if given
    if q_sigma is not None:
        q_sigma = np.asarray(q_sigma, dtype=float)[mask]
        rel_sigma = np.maximum(q_sigma / q_use, 1e-3)
        sigma_log_q = rel_sigma      # ~ sigma_Q / Q
        w = 1.0 / (sigma_log_q ** 2) # weights = 1 / var(logQ)
    else:
        w = np.ones_like(x)

    # Weighted least squares: y = c0 + c1 x
    X = np.vstack([np.ones_like(x), x]).T
    W = np.diag(w)
    XtW = X.T @ W
    beta = np.linalg.solve(XtW @ X, XtW @ y)
    c0, c1 = beta

    log_a = float(c0)
    a = float(np.exp(c0))
    b = float(c1)

    if a <= 0 or b <= 0:
        raise RuntimeError(
            f"Unphysical parameters from fit: a={a}, b={b}. "
            "Try adjusting h0_fixed or filtering data."
        )

    return RatingCurveParams(a=a, b=b, h0=h0, log_a=log_a)


# --- PREDICTION ---

def predict_discharge(
    wl: np.ndarray,
    params: RatingCurveParams,
    clip_zero: bool = True
) -> np.ndarray:
    wl = np.asarray(wl, dtype=float)
    h_eff = wl - params.h0
    if clip_zero:
        h_eff = np.where(h_eff > 0, h_eff, np.nan)
    q_pred = params.a * np.power(h_eff, params.b)
    if clip_zero:
        q_pred = np.where(np.isfinite(q_pred), q_pred, 0.0)
    return q_pred


# --- PLOTTING HELPERS ---

def plot_data_with_q_errorbars(wl, q, q_sigma):
    """Stage vs discharge with Q error bars (your 10% rule)."""
    wl = np.asarray(wl)
    q = np.asarray(q)
    q_sigma = np.asarray(q_sigma)

    plt.figure()
    plt.errorbar(wl, q, yerr=q_sigma, fmt='o', capsize=3, alpha=0.7)
    plt.xlabel("Water level h (wl_final)")
    plt.ylabel("Discharge Q")
    plt.title("Stage–discharge data with 10% Q uncertainty")
    plt.grid(True)
    plt.tight_layout()

def plot_data_with_q_errorbars_by_year(df):
    """
    Stage vs discharge with 10% Q error bars,
    colored by year from the 'datetime' column.
    """
    # ensure datetime
    dt = pd.to_datetime(df["datetime"])
    years = dt.dt.year.to_numpy()

    wl = df["wl_final"].to_numpy()
    q  = df["discharge"].to_numpy()
    q_sigma = 0.10 * q   # 10% relative Q uncertainty

    uniq_years = np.sort(np.unique(years))
    cmap = cm.get_cmap("tab10", len(uniq_years))
    year_to_idx = {y: i for i, y in enumerate(uniq_years)}
    idx = np.array([year_to_idx[y] for y in years])

    plt.figure()
    # scatter with color by year
    plt.scatter(wl, q, c=idx, cmap=cmap, alpha=0.9)

    # error bars with matching colors
    for i in range(len(wl)):
        plt.errorbar(
            wl[i], q[i],
            yerr=q_sigma[i],
            fmt='none',
            ecolor=cmap(idx[i]),
            alpha=0.7,
            capsize=3
        )

    plt.xlabel("Water level h (wl_final)")
    plt.ylabel("Discharge Q")
    plt.title("Stage–discharge with 10% Q uncertainty, colored by year")
    plt.grid(True)

    # legend mapping color -> year
    handles = [
        plt.Line2D([], [], marker='o', linestyle='', color=cmap(i), label=str(y))
        for i, y in enumerate(uniq_years)
    ]
    plt.legend(handles=handles, title="Year", loc="best")
    plt.tight_layout()

def plot_loglog_fit(wl, q, params, q_sigma=None):
    """
    Log–log plot: log(Q) vs log(h - h0) with fitted line.
    This directly shows the linear regression we used.
    """
    wl = np.asarray(wl)
    q = np.asarray(q)
    h0 = params.h0

    mask = wl > h0
    wl_pos = wl[mask]
    q_pos = q[mask]
    x = np.log(wl_pos - h0)
    y = np.log(q_pos)

    plt.figure()
    if q_sigma is not None:
        q_sigma = np.asarray(q_sigma)[mask]
        rel_sigma = np.maximum(q_sigma / q_pos, 1e-3)
        yerr = rel_sigma  # ~ sigma(logQ)
        plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=3, alpha=0.7, label="Data (log space)")
    else:
        plt.plot(x, y, 'o', label="Data (log space)")

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = params.log_a + params.b * x_line
    plt.plot(x_line, y_line, 'r-', label=f"Fit: logQ = {params.log_a:.2f} + {params.b:.2f} log(h-h0)")

    plt.xlabel("log(h - h0)")
    plt.ylabel("log(Q)")
    plt.title("Log–log rating-curve fit (linear model in log space)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_rating_curve(wl, q, params, q_sigma=None):
    """
    Original space: Q vs h, with fitted curve and data (with Q error bars).
    """
    wl = np.asarray(wl, dtype=float)
    q = np.asarray(q, dtype=float)

    wl_grid = np.linspace(wl.min(), wl.max(), 200)
    q_grid = predict_discharge(wl_grid, params)

    plt.figure()
    if q_sigma is not None:
        plt.errorbar(wl, q, yerr=q_sigma, fmt='o', capsize=3, alpha=0.7, label="Data (10% Q)")
    else:
        plt.plot(wl, q, 'o', label="Data")

    plt.plot(wl_grid, q_grid, 'r-', label="Fitted rating curve")

    plt.xlabel("Water level h (wl_final)")
    plt.ylabel("Discharge Q")
    plt.title("Stage–discharge rating curve in original space")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


# ----------------------
wl_q_df = pd.read_csv(filename_mini)
wl_q_df = wl_q_df[['datetime','wl_final', 'discharge']]
wl_q_df["datetime"] = pd.to_datetime(wl_q_df["datetime"])


# wl_q_df as in your example
wl = wl_q_df["wl_final"].to_numpy()
q  = wl_q_df["discharge"].to_numpy()

# your rules:
q_sigma = 0.30 * q                  # 10% relative Q uncertainty
wl_sigma = np.full_like(wl, 2.0)    # 2 cm wl uncertainty (stored, not yet used)

# fit
params = fit_power_law_rating_curve(wl, q, q_sigma=q_sigma, h0_fixed=None)
print(params)

# predictions
wl_q_df["q_fit"] = predict_discharge(wl_q_df["wl_final"].to_numpy(), params)

# plots
plot_data_with_q_errorbars(wl, q, q_sigma)
plot_data_with_q_errorbars_by_year(wl_q_df)
plot_loglog_fit(wl, q, params, q_sigma=q_sigma)
plt.savefig(f"{station}_wl-q.png")
plot_rating_curve(wl, q, params, q_sigma=q_sigma)

plt.show()
