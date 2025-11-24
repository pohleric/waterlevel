import os

import pandas as pd
import yaml
import re
from pathlib import Path

class WaterLevelHomogenizer:
    def __init__(self, param_file):
        with open(param_file, 'r') as f:
            self.params = yaml.safe_load(f)
        # Store main root data path only once
        self.main_data_path = Path(self.params['base_paths'].get('MAIN', ''))

    # --- File Identification Utilities ---
    @staticmethod
    def extract_years(filename):
        match = re.search(r'(20\d{2})(?:[-_–]+(20\d{2}))?', filename)
        if match:
            if match.group(2):
                return f"{match.group(1)}-{match.group(2)}"
            return match.group(1)
        return None

    @staticmethod
    def extract_type(filename):
        m = re.search(r'(atmo|water|cond)', filename, re.IGNORECASE)
        if m:
            return m.group(1).lower()
        return None

    @staticmethod
    def extract_level(filename):
        m = re.search(r'L([1234])', filename, re.IGNORECASE)
        if m:
            return f"L{m.group(1)}"
        return None

    @staticmethod
    def select_files(file_list, year_range, file_type):
        candidates = [f for f in file_list
                      if WaterLevelHomogenizer.extract_years(f) == year_range and
                         WaterLevelHomogenizer.extract_type(f) == file_type]
        levels = ['L2', 'L1', None]
        for lvl in levels:
            for f in candidates:
                if lvl is None:
                    if WaterLevelHomogenizer.extract_level(f) is None:
                        return f
                elif WaterLevelHomogenizer.extract_level(f) == lvl:
                    return f
        return None

    def get_file_path(self, subfolder_key, filename):
        # subfolder_key example: "KG_Arabel", "KG_BS", etc.
        subfolder = self.params['base_paths'].get(subfolder_key, '')
        return self.main_data_path / subfolder / filename

    def detect_file_type(self, filepath):
        # Read first lines to detect type
        with open(filepath, 'r', encoding="utf-8") as f:
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
        # Timezone extraction
        with open(filepath, 'r', encoding="utf-8") as f:
            header = f.readline()  # Plot Title
            header2 = f.readline() # Contains "Date Time, GMT+hh"
        tz_match = re.search(r'GMT([+-]\d+):', header2)
        tz_offset = int(tz_match.group(1)) if tz_match else 0

        df = pd.read_csv(filepath, skiprows=2)
        df.rename(columns={df.columns[1]: "datetime", df.columns[2]: "pressure", df.columns[3]: "temp"}, inplace=True)
        df["datetime"] = pd.to_datetime(df["datetime"], format="%m/%d/%Y %I:%M:%S %p")
        df["datetime"] = df["datetime"] - pd.to_timedelta(tz_offset, unit="h")
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
        return df

    def _read_vanessen(self, filepath):
        # Find which line contains column headers
        start = 0
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "Date/time" in line:
                start = i
                break

        # Extract timezone offset from instrument settings, fallback to 0
        match = re.search(r'UTC([+-]\d+)', "".join(lines[:start]))
        tz_offset = int(match.group(1)) if match else 0

        df = pd.read_csv(filepath, skiprows=start)
        df.rename(columns={df.columns[0]: "datetime", df.columns[1]: "pressure", df.columns[2]: "temp"}, inplace=True)
        df["datetime"] = pd.to_datetime(df["datetime"], format="%Y/%m/%d %H:%M:%S")
        df["datetime"] = df["datetime"] - pd.to_timedelta(tz_offset, unit="h")
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
        return df

    def correct_water_level(self, wl_df, atmo_df):
        atmo_df = atmo_df.sort_values('datetime')
        wl_df = wl_df.sort_values('datetime')
        merged = pd.merge_asof(wl_df, atmo_df, on="datetime", direction="nearest", suffixes=('', '_atmo'))
        merged['wl_corr'] = merged['pressure'] - merged['pressure_atmo']
        return merged

    def adjust_base_level(self, df, key):
        # key example: KG_Arabel_BL_2023
        base_val = self.params.get('base_levels', {}).get(key)
        if base_val is not None:
            # Assume first valid wl_corr is used unless you specify otherwise
            first_valid = df['wl_corr'].dropna().iloc[0] if not df['wl_corr'].dropna().empty else 0
            correction = base_val - first_valid
            df['wl_final'] = df['wl_corr'] + correction
        else:
            df['wl_final'] = df['wl_corr']
        return df

    def adjust_atmo(self, wl_file, atmo_file, base_key):
        df_wl = self.read_with_timezone(wl_file)
        df_atmo = self.read_with_timezone(atmo_file)
        df_corr = self.correct_water_level(df_wl, df_atmo)
        df_final = self.adjust_base_level(df_corr, base_key)
        return df_final

    def get_base_level_uncertainty(self, key):
        return self.params.get('base_level_uncertainties', {}).get(key, None)

    def get_base_level_date(self, key):
        return self.params.get('base_levels_dates', {}).get(key, None)

    def get_station_path(self, main_key, sub_key):
        base = self.params.get('base_paths', {}).get(main_key, "")
        sub = self.params.get('base_paths', {}).get(sub_key, "")
        return Path(base) / sub

    # Extend here for other features (uncertainty, gap-filling, etc.)

# Example YAML structure referenced.
# Usage (minimal):
# h = WaterLevelHomogenizer("params.yml")
# df = h.homogenize("hobo_sample.csv", "hobo_atmo.csv", "KG_Arabel_BL_2024")
h = WaterLevelHomogenizer("params_simple_2025-11-22T1650.yml")

h.get_file_path("")
print(h.params.get("base_paths"))
h.params['base_paths']["KG_Arabel"]

df = h.homogenize("hobo_sample.csv", "hobo_atmo.csv", "KG_Arabel_2024")


