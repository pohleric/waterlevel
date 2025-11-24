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


class WaterLevelHomogenizer:
    def __init__(self, param_file):
        with open(param_file, 'r') as f:
            self.params = yaml.safe_load(f)
        self.main_data_path = Path(self.params['main_data_path'])
        self.stations = self.params['stations']
        self.results = {}
        self.pressure_to_cmH2O = {
            "kPa": 1 / 0.0980665,  # kPa to cmH2O
            "hPa": 0.1 / 0.0980665,  # hPa to cmH2O
            "cmHg": 1.33322 / 0.0980665,  # cmHg to cmH2O
            "cmH2O": 1,  # already in cmH2O
        }
        self.base_level_metadata = {}  # stores info for each station/year

    @staticmethod
    def write_to_csv(dataset, output_filename):
        import copy
        df_out = copy.copy(dataset)
        df_out['datetime'] = df_out['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S%z')
        df_out['datetime'] = df_out['datetime'].str.replace(r'([+-]\d{2})(\d{2})$', r'\1:\2', regex=True)
        df_out.to_csv(output_filename, index=False)

    @staticmethod
    def extract_years(filename, only_start=False):
        # match = re.search(r'(\d{4})(?:[-_–]+(\d{4}))?', filename)
        match = re.search(r'(\d{4})-(\d{4})', filename)
        if match:
            if only_start:
                return match.group(1)  # Return just the start year
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
        # Only match files where the year pattern is 'year-XXXX'
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

    def get_base_level(self, station_key, year):
        info = self.get_station_info(station_key)
        return info['base_levels'].get(int(year))

    def get_base_level_uncertainty(self, station_key, year):
        info = self.get_station_info(station_key)
        return info.get('base_level_uncertainties', {}).get(int(year))

    def get_base_level_date(self, station_key, year):
        info = self.get_station_info(station_key)
        return info.get('base_level_dates', {}).get(int(year))

    def detect_file_type(self, filepath):
        # Use a robust fallback encoding
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

        # Detect pressure unit from header2:
        conversion = None
        unit_match = re.search(r'Abs Pres, *([a-zA-Z0-9]+)', header2)
        if unit_match:
            unit = unit_match.group(1)
            conversion = self.pressure_to_cmH2O.get(unit, None)
        else:
            raise ValueError(f"No unit information found for pressure. Please check input file {filepath}")

        df = pd.read_csv(filepath, header=1)
        df.drop(columns="#", inplace=True)
        df.rename(columns={df.columns[0]: "datetime", df.columns[1]: "pressure", df.columns[2]: "temp"}, inplace=True)
        df = df.iloc[:, 0:3]  # remove decoupling information
        df["datetime"] = pd.to_datetime(df["datetime"], format="%m/%d/%Y %I:%M:%S %p")
        df["datetime"] = df["datetime"] - pd.to_timedelta(tz_offset, unit="h")
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
        # --- Convert pressure column to cmH2O ---
        df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce") * conversion

        # clear invalid datetime stamps
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

        # --- UNIT DETECTION ---
        header_line = lines[start].strip()
        col_names = [c.strip() for c in header_line.split(',')]
        pressure_col = next((col for col in col_names if "Pressure" in col), None)
        unit_match = re.search(r'Pressure\[(.*?)\]', pressure_col) if pressure_col else None
        unit = unit_match.group(1) if unit_match else "cmH2O"
        conversion = self.pressure_to_cmH2O.get(unit, None)
        if conversion is None:
            raise ValueError(f"Unknown pressure unit '{unit}' in file header: {header_line}")

        df = pd.read_csv(filepath, skiprows=start + 1, names=col_names, encoding="latin1")

        # Clean up and parse datetime robustly (handles both formats)
        df.rename(columns={col_names[0]: "datetime", col_names[1]: "pressure", col_names[2]: "temp"}, inplace=True)
        # Automatic conversion
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        # Remove rows where datetime could not be parsed (NaT)
        df = df.dropna(subset=['datetime'])
        # Apply timezone correction
        df["datetime"] = df["datetime"] - pd.to_timedelta(tz_offset, unit="h")
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")

        # Convert pressure to cmH2O
        df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce") * conversion
        df = df[["datetime", "pressure", "temp"]]

        # clear invalid datetime stamps
        df = df.dropna(subset=['datetime'])

        return df

    def correct_water_level(self, wl_df, atmo_df):
        atmo_df = atmo_df.sort_values('datetime')
        wl_df = wl_df.sort_values('datetime')
        merged = pd.merge_asof(wl_df, atmo_df, on="datetime", direction="nearest", suffixes=('', '_atmo'))
        merged['wl_corr'] = merged['pressure'] - merged['pressure_atmo']
        return merged

    def adjust_base_level(self, df, station_key, year):
        base_levels = self.get_base_level(station_key, year)
        base_level_dates = self.get_base_level_date(station_key, year)
        base_level_uncertainties = self.get_base_level_uncertainty(station_key, year)
        # Store in the class
        self.base_level_metadata[(station_key, year)] = {
            'level': base_levels,
            'uncertainty': base_level_uncertainties,
            'date': base_level_dates
        }
        print(f"base_levels: {base_levels}")
        print(f"base_level_dates: {base_level_dates}")
        print(f"base_level_uncertainties: {base_level_uncertainties}")

        if base_levels is not None and base_level_dates is not None:
            target_time = pd.Timestamp(base_level_dates)
            # Find the closest timestamp in the df
            time_diffs = abs(df['datetime'] - target_time)
            if (time_diffs < pd.to_timedelta(1 ,'h')).any():
                idx = time_diffs.idxmin()
                wl_corr_at_base = df.loc[idx, 'wl_corr']
                correction = base_levels - wl_corr_at_base
                df['wl_final'] = df['wl_corr'] + correction
                print(f"wl_final has been adjusted by {correction} cm.")
        else:
            df['wl_final'] = df['wl_corr']
        return df

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
                                  label="Temperature [°C]")
        ax2.set_ylabel("Temperature [°C]", color=colax2)
        ax2.set_ylim(-1, 10)
        ax2.axhline(0, color=colax2, linestyle=':', label="")

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
        plt.savefig(f'{year}_{station_key}_water-level-plot.pdf')
        plt.close()

    @staticmethod
    def plot_ts_multi_years(h, df, station_key, close=False):
        # station_key = "KG_G354-new"
        # df = cat_df
        # station = "KG_G354-old"
        # cat_df = h.concatenate_series_with_baselevel(station=station)
        # cat_df = h.ensure_hourly_continuous(cat_df)
        # h.plot_ts_multi_years(h, cat_df, station)
        # h
        # df = cat_df
        # station_key = station
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, ax = plt.subplots(figsize=(12, 6))
        LW = 0.4
        station_meta = {k: v for k, v in h.base_level_metadata.items() if k[0] == station}
        # Prepare rows as a list of dicts that include the year
        rows = []
        for (name, year), meta in station_meta.items():
            meta_row = meta.copy()
            meta_row['year'] = year
            # Optionally convert 'date' to pandas Timestamp
            if meta_row.get('date') is not None:
                meta_row['date'] = pd.to_datetime(meta_row['date'])
            rows.append(meta_row)

        # Make DataFrame sorted by year
        meta_df = pd.DataFrame(rows).sort_values("year")

        # Primary axis: water level
        x = df['datetime']
        y = df["wl_final"]
        y2 = df["temp"]
        # df.plot(x="datetime", y="wl_final", legend=False, ax=ax, linewidth=LW, color='b',
        #                         label="Water level [cm]")
        ax.plot(x.values,y.values, linewidth=LW, color='b', label="Water level [cm]")
        ax.set_ylabel("Water level [cm]", color='b')

        # Secondary axis: temperature
        ax2 = ax.twinx()
        colax2 = "#33333377"
        ax2.plot(x, y2,  linewidth=LW, color=colax2, label="Temperature [°C]")

        ax2.set_ylabel("Temperature [°C]", color=colax2)
        ax2.set_ylim(-1, 10)
        ax2.axhline(0, color=colax2, linestyle=':', label="")

        # Axis tick colors
        ax.tick_params(axis='y', colors='b')
        ax2.tick_params(axis='y', colors=colax2)

        # Monthly ticks
        months = pd.date_range(df['datetime'].min().normalize(),
                               df['datetime'].max().normalize(),
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

        for j, lvl in enumerate(meta_df['level']):
            dt = meta_df['date'].iloc[j]
            # Strip timezone from dt (if any)
            dt = dt.tz_convert('UTC').tz_localize(None)
            dt = pd.Timestamp(dt)
            ax.plot(dt, lvl, marker='o', color='r', markersize=6, mfc='none')
            # Convert to matplotlib float date

            if j ==0 and not pd.isna(dt):
                l_date = ax.axvline(dt, color='r', linestyle='--', label='$in situ$ water level', linewidth=.5)
            elif not pd.isna(dt):
                l_date = ax.axvline(dt, color='r', linestyle='--', linewidth=.5)
            extra_handles.append(l_date)

        # Combine handles from both y-axes for legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, facecolor='white', framealpha=1) #, loc='upper left'
        ax.set_title(f"{station_key}")
        plt.tight_layout()
        plt.show()
        plt.savefig(f'{station_key}_water-level-plot.pdf')
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
        # station = "KG_G354-new"
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

            # 1: Always calibrate the new segment to its own base level FIRST
            if base_level is not None and base_lvl_date is not None:
                wl_at_base = df.set_index('datetime').reindex([base_lvl_date], method='nearest')['wl_final'].iloc[0]
                base_correction = base_level - wl_at_base
                df['wl_final'] = df['wl_final'] + base_correction
                debug_baselevel_value(df, "2024-08-20T10:00:00+06:00", label=f"1: {station}")
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
                debug_baselevel_value(cropped_prev_df, "2024-08-20T10:00:00+06:00", label=f"2: {station}")
                out.append(cropped_prev_df)
            else:
                # 3: No overlap: append previous segment as is
                debug_baselevel_value(prev_df, "2024-08-20T10:00:00+06:00", label=f"3: {station}")
                out.append(prev_df)

            # 4: Medium gap handling (1h ≤ gap ≤ 5d)
            if 1 <= gap_hours <= 120:
                prev_24h_mask = (prev_df['datetime'] >= prev_end - pd.Timedelta(hours=24)) & (
                            prev_df['datetime'] <= prev_end)
                prev_24h_mean = prev_df.loc[prev_24h_mask, 'wl_final'].mean()
                new_24h_mask = (df['datetime'] >= new_start) & (df['datetime'] < new_start + pd.Timedelta(hours=24))
                new_24h_mean = df.loc[new_24h_mask, 'wl_final'].mean()
                shift = prev_24h_mean - new_24h_mean
                df['wl_final'] = df['wl_final'] + shift
                debug_baselevel_value(prev_df, "2024-08-20T10:00:00+06:00", label=f"4: {station}")
            elif gap_hours > 120:
                print(
                    f"[Concat] Long gap ({gap_hours:.1f} h) between {prev_end} and {new_start} for station {station}: segment treated independently.")
                # No shift

            elif 0 < gap_hours < 1: #5
                wl_prev_end = prev_df['wl_final'].iloc[-1]
                wl_new_start = df['wl_final'].iloc[0]
                shift = wl_prev_end - wl_new_start
                df['wl_final'] = df['wl_final'] + shift
                debug_baselevel_value(prev_df, "2024-08-20T10:00:00+06:00", label=f"5: {station}")

            # Prepare for the next iteration
            prev_df = df

        # Always append the last segment
        out.append(prev_df)
        full = pd.concat(out, ignore_index=True)
        # debug_baselevel_value(full, "2024-08-20T10:00:00+06:00", label=f"5: {station}")

        # debug_baselevel_value(full, "2024-08-20T10:00:00+06:00", label=f" {station}")
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


h = WaterLevelHomogenizer("params_simple_2025-11-25T0040.yml")

stations = ["KG_G354-new","KG_G354-old", "KG_BS-old","KG_BS-new1","KG_BS-new2", "KG_Arabel"]
# years = ["2019","2021","2022","2023", "2024"]
for station in stations:
    # station = "KG_BS-old"
    print(f"################### {station} ###################")
    years = list(h.get_station_info(station)['base_levels'].keys())
    for year in years:
        df_final = h.process_year(station, year)
        if df_final is not None:
            h.write_to_csv(df_final,f"test_{station}_{year}.csv")
            # h.plot_ts_single_year(station,year)
#
station = "KG_G354-old"
cat_df = h.concatenate_series_with_baselevel(station=station)
cat_df = h.ensure_hourly_continuous(cat_df)  # adds empty time steps if interrupted time-series
h.write_to_csv(cat_df, f'{station}_cat_df.csv')
h.plot_ts_multi_years(h,cat_df, station)
# #

station = "KG_G354-new"
cat_df = h.concatenate_series_with_baselevel(station=station)
# cat_df = h.ensure_hourly_continuous(cat_df) # Not needed
h.write_to_csv(cat_df, f'{station}_cat_df.csv')
h.plot_ts_multi_years(h,cat_df, station)

station = "KG_BS-old"
cat_df = h.concatenate_series_with_baselevel(station=station)
# cat_df = h.ensure_hourly_continuous(cat_df)  # not needed
h.write_to_csv(cat_df, f'{station}_cat_df.csv')
h.plot_ts_multi_years(h,cat_df, station)

station = "KG_BS-new1"
cat_df = h.concatenate_series_with_baselevel(station=station)
# cat_df = h.ensure_hourly_continuous(cat_df)  # not needed
h.write_to_csv(cat_df, f'{station}_cat_df.csv')
h.plot_ts_multi_years(h,cat_df, station)

station = "KG_Arabel"
cat_df = h.concatenate_series_with_baselevel(station=station)
# cat_df = h.ensure_hourly_continuous(cat_df)  # not needed
h.write_to_csv(cat_df, f'{station}_cat_df.csv')
h.plot_ts_multi_years(h,cat_df, station)


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
