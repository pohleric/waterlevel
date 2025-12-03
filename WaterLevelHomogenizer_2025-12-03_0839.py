
# Imports for OSX plotting only:
import matplotlib
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
matplotlib.use('TkAgg')
# - END -

import pandas as pd
import yaml
import re
from pathlib import Path


import pandas as pd
import yaml
import re
from pathlib import Path


class WaterLevelHomogenizer:
    def __init__(self, param_file):
        with open(param_file, 'r') as f:
            self.params = yaml.safe_load(f)
        self.main_data_path = Path(self.params['main_data_path'])
        # normalize stations and convert base_level_dates from UTC+6 to UTC
        self.stations = self._normalize_station_params(self.params['stations'])
        self.results = {}
        self.pressure_to_cmH2O = {
            "kPa": 1 / 0.0980665,
            "hPa": 0.1 / 0.0980665,
            "cmHg": 1.33322 / 0.0980665,
            "cmH2O": 1,
        }
        # For plotting and concatenation: effective calibration info per (station, year)
        # Each entry: {'levels': [...], 'dates': [...], 'uncertainties': [...], 'offset': float}
        self.base_level_metadata = {}

    @staticmethod
    def _ensure_list(v):
        if v is None:
            return []
        if isinstance(v, list):
            return v
        # scalar -> single-element list
        return [v]

    def _normalize_station_params(self, stations_dict):
        """
        Ensure base_levels, base_level_dates, base_level_uncertainties
        are always lists per year, even if YAML has scalars.
        Additionally:
        - Interpret all base_level_dates from YAML as local UTC+6
          and immediately convert them to UTC ISO strings.
        """
        norm = {}
        for sk, info in stations_dict.items():
            info = info.copy()

            # base_levels and uncertainties as lists
            for key in ['base_levels', 'base_level_uncertainties']:
                if key in info:
                    new_sub = {}
                    for y, v in info[key].items():
                        new_sub[int(y)] = self._ensure_list(v)
                    info[key] = new_sub

            # base_level_dates: list of strings, but convert +06:00 -> UTC once
            if 'base_level_dates' in info:
                new_dates = {}
                for y, v in info['base_level_dates'].items():
                    year_int = int(y)
                    dates_list = self._ensure_list(v)
                    utc_dates = []
                    for d in dates_list:
                        # parse as tz-aware from string, then convert to UTC
                        ts_local = pd.to_datetime(d, utc=True)  # respects +06:00 in the string
                        # store as ISO UTC string (e.g. 2024-08-20T03:53:00+00:00)
                        utc_dates.append(ts_local.isoformat())
                    new_dates[year_int] = utc_dates
                info['base_level_dates'] = new_dates

            norm[sk] = info
        return norm

    @staticmethod
    def write_to_csv(dataset, output_filename):
        import copy
        df_out = copy.copy(dataset)
        df_out['datetime'] = df_out['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S%z')
        df_out['datetime'] = df_out['datetime'].str.replace(r'([+-]\d{2})(\d{2})$', r'\1:\2', regex=True)
        df_out.to_csv(output_filename, index=False)

    @staticmethod
    def extract_years(filename, only_start=False):
        match = re.search(r'(\d{4})-(\d{4})', filename)
        if match:
            if only_start:
                return match.group(1)
            if match.group(2):
                return f"{match.group(1)}-{match.group(2)}"
            return match.group(1)
        return None

    @staticmethod
    def extract_type(filename):
        m = re.search(r'(atmo|water|cond)', filename, re.IGNORECASE)
        return m.group(1).lower() if m else None

    @staticmethod
    def extract_level(filename):
        m = re.search(r'L([1234])', filename, re.IGNORECASE)
        return f"L{m.group(1)}" if m else None

    @staticmethod
    def select_files(file_list, year, file_type):
        candidates = [
            f for f in file_list
            if WaterLevelHomogenizer.extract_type(f) == file_type
               and WaterLevelHomogenizer.extract_years(f, only_start=True) == year
        ]
        levels = ['L2', 'L1', None]
        for lvl in levels:
            for f in candidates:
                if lvl is None:
                    if WaterLevelHomogenizer.extract_level(f) is None:
                        return f
                elif WaterLevelHomogenizer.extract_level(f) == lvl:
                    return f
        return None

    def get_station_info(self, station_key):
        return self.stations[station_key]

    def get_file_list(self, station_key, measurement_type):
        info = self.get_station_info(station_key)
        folder = info[f"{measurement_type}_folder"]
        path = self.main_data_path / folder
        _fnames = [f.name for f in path.glob("*.csv")] + [f.name for f in path.glob("*.CSV")]
        return _fnames

    # ---- helpers that always return lists ----

    def get_base_levels(self, station_key, year):
        info = self.get_station_info(station_key)
        return info.get('base_levels', {}).get(int(year), [])

    def get_base_level_uncertainties(self, station_key, year):
        info = self.get_station_info(station_key)
        return info.get('base_level_uncertainties', {}).get(int(year), [])

    def get_base_level_dates(self, station_key, year):
        info = self.get_station_info(station_key)
        return info.get('base_level_dates', {}).get(int(year), [])

    # ------------------------------------------------

    def detect_file_type(self, filepath):
        try:
            with open(filepath, 'r', encoding="utf-8") as f:
                lines = [next(f) for _ in range(8)]
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding="latin1") as f:
                lines = [next(f) for _ in range(8)]
        if "Plot Title" in lines[0] and "GMT" in lines[1]:
            return "hobo"
        elif "Data file for DataLogger." in lines[0] or "Diver-Office" in "".join(lines):
            return "vanEssen"
        else:
            raise ValueError(f"Cannot detect filetype for {filepath}")

    def read_with_timezone(self, filepath, filetype=None):
        if filetype is None:
            filetype = self.detect_file_type(filepath)
        if filetype == "hobo":
            return self._read_hobo(filepath)
        elif filetype == "vanEssen":
            return self._read_vanessen(filepath)
        else:
            raise ValueError("Unknown filetype: {}".format(filetype))

    def _read_hobo(self, filepath):
        print(f"Reading: {filepath}")
        with open(filepath, 'r', encoding="utf-8") as f:
            header = f.readline()
            header2 = f.readline()
        tz_match = re.search(r'GMT([+-]\d+):', header2)
        if tz_match:
            tz_offset = int(tz_match.group(1))
        else:
            raise ValueError(f"No timezone information found. Please check input file {filepath}")

        unit_match = re.search(r'Abs Pres, *([a-zA-Z0-9]+)', header2)
        if unit_match:
            unit = unit_match.group(1)
            conversion = self.pressure_to_cmH2O.get(unit, None)
        else:
            raise ValueError(f"No unit information found for pressure. Please check input file {filepath}")

        df = pd.read_csv(filepath, header=1)
        df.drop(columns="#", inplace=True)
        df.rename(columns={df.columns[0]: "datetime", df.columns[1]: "pressure", df.columns[2]: "temp"}, inplace=True)
        df = df.iloc[:, 0:3]
        df["datetime"] = pd.to_datetime(df["datetime"], format="%m/%d/%Y %I:%M:%S %p")
        # Convert from local GMT+offset to UTC
        df["datetime"] = df["datetime"] - pd.to_timedelta(tz_offset, unit="h")
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
        df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce") * conversion
        df = df.dropna(subset=['datetime'])
        return df

    def _read_vanessen(self, filepath):
        start = 0
        with open(filepath, "r", encoding="latin-1") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "Date/time" in line:
                start = i
                break
        tz_match = re.search(r'UTC([+-]\d+)', "".join(lines[:start]))
        if tz_match:
            tz_offset = int(tz_match.group(1))
        else:
            raise ValueError(f"No timezone information found. Please check input file {filepath}")

        header_line = lines[start].strip()
        col_names = [c.strip() for c in header_line.split(',')]
        pressure_col = next((col for col in col_names if "Pressure" in col), None)
        unit_match = re.search(r'Pressure\[(.*?)\]', pressure_col) if pressure_col else None
        unit = unit_match.group(1) if unit_match else "cmH2O"
        conversion = self.pressure_to_cmH2O.get(unit, None)
        if conversion is None:
            raise ValueError(f"Unknown pressure unit '{unit}' in file header: {header_line}")

        df = pd.read_csv(filepath, skiprows=start + 1, names=col_names, encoding="latin1")
        df.rename(columns={col_names[0]: "datetime", col_names[1]: "pressure", col_names[2]: "temp"}, inplace=True)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=['datetime'])
        # Convert from local UTC+offset to UTC
        df["datetime"] = df["datetime"] - pd.to_timedelta(tz_offset, unit="h")
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
        df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce") * conversion
        df = df[["datetime", "pressure", "temp"]]
        df = df.dropna(subset=['datetime'])
        return df

    def correct_water_level(self, wl_df, atmo_df):
        atmo_df = atmo_df.sort_values('datetime')
        wl_df = wl_df.sort_values('datetime')
        merged = pd.merge_asof(wl_df, atmo_df, on="datetime", direction="nearest", suffixes=('', '_atmo'))
        merged['wl_corr'] = merged['pressure'] - merged['pressure_atmo']
        return merged

    # --------- multi-base-level, per-year calibration in UTC ---------

    def adjust_base_level(self, df, station_key, start_year):
        """
        Calibrate wl_corr -> wl_final using base levels:
        - Supports multiple base levels per year.
        - Works across years inside this df (e.g. Aug 2024–Aug 2025).
        - offset = mean(base_i - measured_i).
        - Each calendar year segment is calibrated only from its own base levels.
        - Years without any usable base point keep wl_corr (no shift).
        Assumes df['datetime'] is tz-aware UTC, and base_level_dates are stored as UTC strings.
        """
        df = df.sort_values('datetime').copy()
        years_in_df = sorted(df['datetime'].dt.year.unique())

        used_years = {}
        year_offsets = {}

        # How far outside the series we still accept a base-level timestamp
        window = pd.Timedelta('1h')

        for y in years_in_df:
            base_levels = self.get_base_levels(station_key, y)             # list
            base_dates = self.get_base_level_dates(station_key, y)         # list of UTC strings
            base_uncs  = self.get_base_level_uncertainties(station_key, y) # list

            if not base_levels or not base_dates:
                continue

            df_year_mask = df['datetime'].dt.year == y
            if not df_year_mask.any():
                continue
            df_year = df.loc[df_year_mask]
            tmin, tmax = df_year['datetime'].min(), df_year['datetime'].max()  # tz-aware UTC

            # parse dates as UTC-aware; they are already UTC strings now
            parsed_dates = [pd.to_datetime(d, utc=True) for d in base_dates]

            valid_pairs = [
                (b, d)
                for b, d in zip(base_levels, parsed_dates)
                if (d >= (tmin - window)) and (d <= (tmax + window))
            ]
            if not valid_pairs:
                continue

            bases, dates = zip(*valid_pairs)
            bases = pd.Series(bases, dtype="float64")
            dates = pd.to_datetime(list(dates), utc=True)

            df_year_idx = df_year.set_index('datetime')  # tz-aware UTC
            nearest_idx = df_year_idx.index.get_indexer(dates, method='nearest')
            measured = df_year_idx['wl_corr'].iloc[nearest_idx].reset_index(drop=True)

            deviations = bases.reset_index(drop=True) - measured
            offset = deviations.mean()

            # apply offset only to this calendar year
            df.loc[df_year_mask, 'wl_final'] = df.loc[df_year_mask, 'wl_corr'] + offset

            used_years[y] = {
                'levels': list(bases),
                'dates': [d.isoformat() for d in dates],  # UTC ISO strings
                'uncertainties': base_uncs,
                'offset': float(offset),
            }
            year_offsets[y] = float(offset)

        # fill in wl_final for rows without calibration
        if 'wl_final' not in df.columns:
            df['wl_final'] = df['wl_corr']
        else:
            mask_na = df['wl_final'].isna()
            df.loc[mask_na, 'wl_final'] = df.loc[mask_na, 'wl_corr']

        # store metadata for all years in this df
        for y in years_in_df:
            meta = used_years.get(y, None)
            if meta is None:
                meta = {
                    'levels': [],
                    'dates': [],
                    'uncertainties': [],
                    'offset': None,
                }
            self.base_level_metadata[(station_key, y)] = meta

        return df

    # ---------------------------------------------------------------

    def process_year(self, station_key, year):
        water_files = self.get_file_list(station_key, "wl")
        atmo_files = self.get_file_list(station_key, "atmo")

        year_str = str(year)
        water_file = self.select_files(water_files, year_str, "water")
        atmo_file = self.select_files(atmo_files, year_str, "atmo")

        if water_file is None:
            print(f"[SKIP] No water file for {station_key}, {year_str}")
            return None
        if atmo_file is None:
            print(f"[SKIP] No atmo file for {station_key}, {year_str}")
            return None

        info = self.get_station_info(station_key)
        water_fullpath = self.main_data_path / info['wl_folder'] / water_file
        atmo_fullpath = self.main_data_path / info['atmo_folder'] / atmo_file

        df_wl = self.read_with_timezone(water_fullpath)
        df_atmo = self.read_with_timezone(atmo_fullpath)
        self.df_corr = self.correct_water_level(df_wl, df_atmo)
        self.df_final = self.adjust_base_level(self.df_corr, station_key, year)
        self.results[(station_key, int(year))] = self.df_final
        return self.df_final

    def plot_ts_single_year(self, station_key, year):
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, ax = plt.subplots(figsize=(12, 6))
        meta = self.base_level_metadata.get((station_key, year), {})
        LW = 0.4
        # Primary axis: water level
        self.df_final.plot(x="datetime", y="wl_final", legend=False, ax=ax, linewidth=LW, color='b',
                                label="Water level [cm]")
        ax.set_ylabel("Water level [cm]", color='b')

        # Secondary axis: temperature
        ax2 = ax.twinx()
        colax2 = "#33333377"
        self.df_final.plot(x="datetime", y="temp", legend=False, ax=ax2, linewidth=LW, color=colax2,
                                  label="Temperature [°C]", zorder=1)
        ax2.set_ylabel("Temperature [°C]", color=colax2)
        ax2.set_ylim(-1, 10)
        ax2.axhline(0, color=colax2, linestyle=':', label="", zorder=1)

        # Axis tick colors
        ax.tick_params(axis='y', colors='b')
        ax2.tick_params(axis='y', colors=colax2)

        # Monthly ticks
        months = pd.date_range(self.df_final['datetime'].min().normalize(),
                               self.df_final['datetime'].max().normalize(),
                               freq='MS', tz="UTC")
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.set_xticks(months)
        ax.set_xticklabels([d.strftime('%Y-%m') for d in months], rotation=45)
        ax.set_xlabel("")
        ax.axhline(0, color='b', linestyle=':')

        # Plot base level reference info
        extra_handles = []
        extra_labels = []
        if meta.get('level') and meta.get('date'):
            l_base = ax.axhline(meta['level'], color='r', linestyle='--', label='Base level')
            extra_handles.append(l_base)
            extra_labels.append('Base level')
            if meta.get('uncertainty'):
                l_unc = ax.axhspan(meta['level'] - meta['uncertainty'], meta['level'] + meta['uncertainty'],
                                   color='r', alpha=0.15, label="Base level uncertainty", zorder=-10)
                extra_handles.append(l_unc)
                extra_labels.append('Base level uncertainty')
            l_date = ax.axvline(pd.Timestamp(meta['date']), color='k', linestyle=':', label='Base level date')
            extra_handles.append(l_date)
            extra_labels.append('Base level date')

        # Combine handles from both y-axes for legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2) #, loc='upper left'
        ax.set_title(f"{station_key} - {year}")
        plt.tight_layout()
        plt.show()
        plt.savefig(f'figures/{year}_{station_key}_water-level-plot.pdf')
        plt.close()

    @staticmethod
    def plot_ts_multi_years(h, df, station_key, close=False, legend_location='upper left'):
        """
        Plot concatenated multi-year time series for a station, with:
        - wl_final on primary y-axis
        - temp on secondary y-axis
        - All used base-level calibration points from h.base_level_metadata (red).
        - All original base_levels from the param file (orange).
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, ax = plt.subplots(figsize=(12, 6))
        LW = 0.4

        # --- primary axis: water level ---
        x = df['datetime']
        y = df["wl_final"]
        y2 = df["temp"]

        ax.plot(x.values, y.values, linewidth=LW, color='b', label="Water level [cm]", zorder=4)
        ax.set_ylabel("Water level [cm]", color='b')

        # --- secondary axis: temperature ---
        ax2 = ax.twinx()
        colax2 = "#33333355"
        ax2.plot(x.values, y2.values, linewidth=LW, color=colax2, label="Temperature [°C]", zorder=-1)
        ax2.set_ylabel("Temperature [°C]", color=colax2)
        ax2.set_ylim(-1, 10)
        ax2.axhline(0, color=colax2, linestyle=':', label="")

        # Axis tick colors
        ax.tick_params(axis='y', colors='b')
        ax2.tick_params(axis='y', colors=colax2)

        # --- monthly ticks ---
        months = pd.date_range(
            df['datetime'].min().normalize(),
            df['datetime'].max().normalize(),
            freq='MS',
            tz="UTC"
        )
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.set_xticks(months)
        ax.set_xticklabels([d.strftime('%Y-%m') for d in months], rotation=45)
        ax.set_xlabel("")
        ax.axhline(0, color='b', linestyle=':', zorder=4)

        # ------------------------------------------------------------------
        # 1) USED calibration points from base_level_metadata  (RED)
        # ------------------------------------------------------------------
        station_meta = {
            (st, yr): meta
            for (st, yr), meta in h.base_level_metadata.items()
            if st == station_key
        }

        used_rows = []
        for (name, year), meta in station_meta.items():
            levels = meta.get('levels', []) or []
            dates = meta.get('dates', []) or []
            uncs = meta.get('uncertainties', []) or []
            if not uncs:
                uncs = [None] * len(levels)

            for lvl, dt_str, uc in zip(levels, dates, uncs):
                try:
                    dt = pd.to_datetime(dt_str)
                    if hasattr(dt, "tzinfo") and dt.tzinfo is not None:
                        dt = dt.tz_convert('UTC').tz_localize(None)
                except Exception:
                    continue
                used_rows.append({
                    'year': year,
                    'level': float(lvl) if lvl is not None else None,
                    'date': dt,
                    'uncertainty': uc,
                })

        if used_rows:
            used_df = pd.DataFrame(used_rows).sort_values(["year", "date"])
            first_idx_used = used_df.index[0]
            for j, row in used_df.iterrows():
                lvl = row['level']
                dt = row['date']
                if pd.isna(lvl) or pd.isna(dt):
                    continue
                #
                # # make dt naive UTC for plotting
                # if hasattr(dt, "tzinfo") and dt.tzinfo is not None:
                #     dt_plot = dt.tz_convert('UTC').tz_localize(None)
                # else:
                #     dt_plot = dt
                # dt_plot = pd.Timestamp(dt_plot)
                dt_plot = dt
                dt_plot = pd.Timestamp(dt_plot)

                ax.plot(dt_plot, lvl, marker='o', color='r', markersize=6, mfc='none', zorder=2)
                if j == first_idx_used:
                    ax.axvline(
                        dt_plot, color='r', linestyle='--',
                        label='$in situ$ (used)', linewidth=.5
                    )
                else:
                    ax.axvline(
                        dt_plot, color='r', linestyle='--', linewidth=.5, zorder=2
                    )

        # ------------------------------------------------------------------
        # 2) ORIGINAL base_levels from YAML  (ORANGE)
        # ------------------------------------------------------------------
        info = h.get_station_info(station_key)
        orig_rows = []
        base_levels_all = info.get('base_levels', {})
        base_dates_all = info.get('base_level_dates', {})

        for year, levels_list in base_levels_all.items():
            # levels_list and dates_list are lists by construction of _normalize_station_params
            levels_list = levels_list or []
            dates_list = base_dates_all.get(year, []) or []
            # zip -> one point per original (level, date) pair
            for lvl, dt_str in zip(levels_list, dates_list):
                try:
                    dt = pd.to_datetime(dt_str)
                except Exception:
                    continue
                orig_rows.append({
                    'year': year,
                    'level': float(lvl) if lvl is not None else None,
                    'date': dt,
                })

        if orig_rows:
            orig_df = pd.DataFrame(orig_rows).sort_values(["year", "date"])
            first_idx_orig = orig_df.index[0]
            for j, row in orig_df.iterrows():
                lvl = row['level']
                dt = row['date']
                if pd.isna(lvl) or pd.isna(dt):
                    continue

                if hasattr(dt, "tzinfo") and dt.tzinfo is not None:
                    dt_plot = dt.tz_convert('UTC').tz_localize(None)
                else:
                    dt_plot = dt
                dt_plot = pd.Timestamp(dt_plot)

                ax.plot(dt_plot, lvl, marker='x', color='orange', markersize=5, mfc='none', zorder=3)
                if j == first_idx_orig:
                    ax.axvline(
                        dt_plot, color='orange', linestyle=':',
                        label='original base level', linewidth=.7
                    )
                else:
                    ax.axvline(
                        dt_plot, color='orange', linestyle=':', linewidth=.7, zorder=3
                    )

        # --- legend ---
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        leg = ax.legend(lines1 + lines2, labels1 + labels2,
                        facecolor='white', framealpha=1.0,loc=legend_location)
        leg.set_zorder(10)
        # Make sure frame is fully opaque
        frame = leg.get_frame()
        frame.set_alpha(1.0)

        ax.set_title(f"{station_key}")
        plt.tight_layout()
        plt.show()
        plt.savefig(f'figures/{station_key}_water-level-plot.pdf')
        if close:
            plt.close()

    def concatenate_series_with_baselevel(self, station):
        """
        Concatenate time series segments for a station, ensuring:
        - New segment's base level calibration point always matches the prescribed value.
        - Overlap: adjusts the previous segment's tail for continuity, not the new segment.
        - 1h ≤ gap ≤ 5d: uses 24h means to determine shift.
        - gap > 5d: no shift is made, segments are independent.
        No base level uncertainties applied in this version.
        """
        # station = "KG_Arabel"
        # keys = [k for k in h.results if k[0] == station]
        # if not keys:
        #     print(f"[Concat] No results for station {station}")
        #     return None
        #
        # # Collect and sort all segment DataFrames and metadata
        # segments = []
        # for k in keys:
        #     df = h.results[k].copy()
        #     year = k[1]
        #     meta = h.base_level_metadata.get((station, year), {})
        #     base_level_date = meta.get('date')
        #     start = df['datetime'].min()
        #     segments.append({
        #         "year": year,
        #         "df": df,
        #         "base_level_date": pd.to_datetime(base_level_date) if base_level_date else start,
        #         "meta": meta
        #     })
        # segments.sort(key=lambda s: s['base_level_date'])
        #
        keys = [k for k in self.results if k[0] == station]
        if not keys:
            print(f"[Concat] No results for station {station}")
            return None

        # Collect and sort all segment DataFrames and metadata
        segments = []
        for k in keys:
            df = self.results[k].copy()
            year = k[1]
            meta = self.base_level_metadata.get((station, year), {})
            base_level_date = meta.get('date')
            start = df['datetime'].min()
            segments.append({
                "year": year,
                "df": df,
                "base_level_date": pd.to_datetime(base_level_date) if base_level_date else start,
                "meta": meta
            })
        segments.sort(key=lambda s: s['base_level_date'])

        out = []
        prev_df = None

        for i, seg in enumerate(segments):
            df = seg['df']
            meta = seg['meta']
            base_level = meta.get('level')
            base_lvl_date = seg["base_level_date"]
            print(base_lvl_date)

            # 1: Always calibrate the new segment to its own base level FIRST
            if base_level is not None and base_lvl_date is not None:
                wl_at_base = df.set_index('datetime').reindex([base_lvl_date], method='nearest')['wl_final'].iloc[0]
                base_correction = base_level - wl_at_base
                df['wl_final'] = df['wl_final'] + base_correction
                # debug_baselevel_value(df, "2024-08-20T10:00:00+06:00", label=f"1: {station}")
            if i == 0:
                # Use df as previous one (corrected for base level reading if it was available)
                prev_df = df
                continue

            prev_end = prev_df['datetime'].max()
            new_start = df['datetime'].min()
            gap_hours = (new_start - prev_end).total_seconds() / 3600

            overlap_times = df['datetime'].isin(prev_df['datetime'])
            has_overlap = overlap_times.any()

            if has_overlap:
                # 2: Calculate average offset from overlap; use for tail of prev_df (not new df!)
                print("step 2: overlap")
                overlap_df_new = df[overlap_times].set_index('datetime')
                overlap_df_prev = prev_df[prev_df['datetime'].isin(overlap_df_new.index)].set_index('datetime')
                overlap_df_new = overlap_df_new.loc[overlap_df_prev.index]
                avg_offset = (overlap_df_prev['wl_final'] - overlap_df_new['wl_final']).mean()
                # Tail adjustment: adjust only prev_df for datetimes with same year as new df
                start_new_year = df['datetime'].dt.year.iloc[0]
                mask_new_year = prev_df['datetime'].dt.year == start_new_year
                prev_df.loc[mask_new_year, 'wl_final'] = prev_df.loc[mask_new_year, 'wl_final'] - avg_offset
                print(
                    f"new and adjusted value in 'df_prev' is: {df.set_index('datetime').reindex([base_lvl_date], method='nearest')['wl_final'].iloc[0]}")
                # Remove overlapping times from prev_df before concatenation
                mask_not_overlap = ~prev_df['datetime'].isin(df['datetime'])
                cropped_prev_df = prev_df[mask_not_overlap]
                # debug_baselevel_value(cropped_prev_df, "2024-08-20T10:00:00+06:00", label=f"2: {station}")
                out.append(cropped_prev_df)
            else:
                # 3: No overlap: append previous segment as is
                print("step 3: no overlap")
                # debug_baselevel_value(prev_df, "2024-08-20T10:00:00+06:00", label=f"3: {station}")
                out.append(prev_df)

            # 4: Medium gap handling (1h ≤ gap ≤ 5d)
            if 1 <= gap_hours <= 120:
                print("step 4: no overlap, max dt 1<= gap <=120 hours.")
                prev_24h_mask = (prev_df['datetime'] >= prev_end - pd.Timedelta(hours=24)) & (
                            prev_df['datetime'] <= prev_end)
                prev_24h_mean = prev_df.loc[prev_24h_mask, 'wl_final'].mean()
                new_24h_mask = (df['datetime'] >= new_start) & (df['datetime'] < new_start + pd.Timedelta(hours=24))
                new_24h_mean = df.loc[new_24h_mask, 'wl_final'].mean()
                shift = prev_24h_mean - new_24h_mean
                df['wl_final'] = df['wl_final'] + shift
                # debug_baselevel_value(prev_df, "2024-08-20T10:00:00+06:00", label=f"4: {station}")
            elif gap_hours > 120:
                print(
                    f"[Concat] Long gap ({gap_hours:.1f} h) between {prev_end} and {new_start} for station {station}: segment treated independently.")
                # No shift

            elif 0 < gap_hours < 1: #5
                print("step 5: no overlap, gap < 1hour.")
                wl_prev_end = prev_df['wl_final'].iloc[-1]
                wl_new_start = df['wl_final'].iloc[0]
                shift = wl_prev_end - wl_new_start
                df['wl_final'] = df['wl_final'] + shift
                # debug_baselevel_value(prev_df, "2024-08-20T10:00:00+06:00", label=f"5: {station}")

            # Prepare for the next iteration
            prev_df = df

        # Always append the last segment
        out.append(prev_df)
        full = pd.concat(out, ignore_index=True)

        # ------------------------------------------------------------------
        # FINAL STEP: adjust last calendar year of the last segment
        # to match the last available base_level from the parameter file.
        # This is specifically to enforce e.g. Arabel 2025 final base level.
        # ------------------------------------------------------------------
        # 1) identify last segment and its station
        #    (we already know `station` argument of this method)
        station_key = station

        # 2) get all years present in the concatenated series
        years_in_full = sorted(full['datetime'].dt.year.unique())
        if years_in_full:
            last_year = years_in_full[-1]

            # 3) get base levels and dates from parameters for that last year
            base_levels_last = self.get_base_levels(station_key, last_year)
            base_dates_last = self.get_base_level_dates(station_key, last_year)

            if base_levels_last and base_dates_last:
                # Use all base-level points for that last year
                # (they are already stored as UTC ISO strings)
                bases = pd.Series(base_levels_last, dtype="float64")
                dates = pd.to_datetime(base_dates_last, utc=True)

                # Restrict to last_year part of the full series
                mask_last_year = full['datetime'].dt.year == last_year
                df_last_year = full.loc[mask_last_year].copy()
                if not df_last_year.empty:
                    # For each base date, find the nearest wl_final in that last year
                    df_last_idx = df_last_year.set_index('datetime')
                    nearest_idx = df_last_idx.index.get_indexer(dates, method='nearest')
                    measured = df_last_idx['wl_final'].iloc[nearest_idx].reset_index(drop=True)

                    # mean deviation: base - measured
                    deviations = bases.reset_index(drop=True) - measured
                    final_offset = deviations.mean()

                    # apply correction ONLY to last_year rows in the concatenated full series
                    full.loc[mask_last_year, 'wl_final'] = (
                            full.loc[mask_last_year, 'wl_final'] + final_offset
                    )

        self.full = full
        return full

    @staticmethod
    def ensure_hourly_continuous(df, datetime_col="datetime"):
        """
        Fills a DataFrame so the 'datetime' column is a continuous hourly series from the min to max date.
        All other columns will be set to NaN for missing times.
        Returns the fixed DataFrame.
        """
        # Ensure datetime column is datetime type
        dt = pd.to_datetime(df[datetime_col])

        # Create full hourly DatetimeIndex
        full_range = pd.date_range(start=dt.min(), end=dt.max(), freq="h", tz=dt.dt.tz)

        # Set index, reindex to full hourly range, reset index
        df = df.set_index(datetime_col)
        df_full = df.reindex(full_range)
        df_full.index.name = datetime_col

        # Restore DataFrame format
        return df_full.reset_index()



def debug_baselevel_value(df, base_lvl_date, label=""):
    dt_check = pd.to_datetime(base_lvl_date)
    # Find closest matching datetime in df
    nearest_idx = df['datetime'].sub(dt_check).abs().idxmin()
    v_at_base = df.loc[nearest_idx, 'wl_final']
    print(f"{label} | Calibration timestamp: {dt_check} | Nearest df timestamp: {df.loc[nearest_idx, 'datetime']} | wl_final: {v_at_base}")


h = WaterLevelHomogenizer("params_simple_2025-12-01T1936.yml")

stations = ["KG_G354-new","KG_G354-old", "KG_BS-old","KG_BS-new1","KG_BS-new2", "KG_Arabel"]
# years = ["2019","2021","2022","2023", "2024"]
for station in stations:
    # station = "KG_BS-old"
    print(f"################### {station} ###################")
    years = list(h.get_station_info(station)['base_levels'].keys())
    for year in years:
        df_final = h.process_year(station, year)
        # if df_final is not None:
        #     h.write_to_csv(df_final,f"test_{station}_{year}.csv")
        #     # h.plot_ts_single_year(station,year)


station = "KG_G354-new"
cat_df = h.concatenate_series_with_baselevel(station=station)
# cat_df = h.ensure_hourly_continuous(cat_df) # Not needed
h.write_to_csv(cat_df, f"/Users/pohle/Dropbox/Central_Asia/DISCHARGE/L3/{station}_cat_df.csv")
h.plot_ts_multi_years(h,cat_df, station)
plt.close()

station = "KG_G354-old"
cat_df = h.concatenate_series_with_baselevel(station=station)
cat_df = h.ensure_hourly_continuous(cat_df) # Not needed
h.write_to_csv(cat_df, f"/Users/pohle/Dropbox/Central_Asia/DISCHARGE/L3/{station}_cat_df.csv")
h.plot_ts_multi_years(h,cat_df, station)
# plt.close()

station = "KG_BS-old"
cat_df = h.concatenate_series_with_baselevel(station=station)
# cat_df = h.ensure_hourly_continuous(cat_df)  # not needed
h.write_to_csv(cat_df, f"/Users/pohle/Dropbox/Central_Asia/DISCHARGE/L3/{station}_cat_df.csv")
h.plot_ts_multi_years(h,cat_df, station)
plt.close()

station = "KG_BS-new1"
cat_df = h.concatenate_series_with_baselevel(station=station)
# cat_df = h.ensure_hourly_continuous(cat_df)  # not needed
h.write_to_csv(cat_df, f"/Users/pohle/Dropbox/Central_Asia/DISCHARGE/L3/{station}_cat_df.csv")
h.write_to_csv(cat_df, f"/Users/pohle/Dropbox/Central_Asia/DISCHARGE/L3/KG_BS-new_cat_df.csv")
h.plot_ts_multi_years(h,cat_df, station)
plt.close()

station = "KG_Arabel"
cat_df = h.concatenate_series_with_baselevel(station=station)
# cat_df = h.ensure_hourly_continuous(cat_df)  # not needed
h.write_to_csv(cat_df, f"/Users/pohle/Dropbox/Central_Asia/DISCHARGE/L3/{station}_cat_df.csv")
h.plot_ts_multi_years(h,cat_df, station,legend_location="best")
plt.close()
#
#
# # Suppose df_arabel is the df passed into adjust_base_level for that year
# df_year = cat_df[cat_df['datetime'].dt.year == 2024]  # or the relevant year
# tmin, tmax = df_year['datetime'].min(), df_year['datetime'].max()
# print("tmin, tmax:", tmin, tmax)
#
# info = h.get_station_info('KG_Arabel')
# print(info['base_level_dates'][2024])  # list of strings
#
# for d in info['base_level_dates'][2024]:
#     ts = pd.Timestamp(d)
#     print(d, "->", ts, "inside?", tmin <= ts <= tmax)

#
#asdasd
# print(cat_df["datetime"].dt.tz)
#
# start = pd.Timestamp('2021-08-01', tz='UTC')
# end = pd.Timestamp('2022-08-30', tz='UTC')
#
# subset = cat_df[(cat_df['datetime'] >= start) & (cat_df['datetime'] <= end)]
# subset.to_csv("test.csv")
#
#
# h.plot_ts_multi_years(h,cat_df, station)


# # Concatenate and adjust series to have smooth WL transition:
# cat_df = h.concatenate_series_with_baselevel(station="KG_Arabel")
# # cat_df.plot(x='datetime', y='wl_final')
# cat_df = h.concatenate_series_with_baselevel(station="KG_BS-old")
# # cat_df.plot(x='datetime', y='wl_final')
#
# cat_df = h.concatenate_series_with_baselevel(station="KG_G354-old")
# # cat_df.plot(x='datetime', y='wl_final')
# #
# # h.results.keys()
# #
# # full.plot(x='datetime', y='wl_final')
