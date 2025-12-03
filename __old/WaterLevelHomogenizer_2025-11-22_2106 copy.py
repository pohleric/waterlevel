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

    @staticmethod
    def write_to_csv(dataset, output_filename):
        dataset['datetime'] = dataset['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S%z')
        dataset['datetime'] = dataset['datetime'].str.replace(r'([+-]\d{2})(\d{2})$', r'\1:\2', regex=True)
        dataset.to_csv(output_filename, index=False)

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
        return info['base_levels'].get(year)

    def get_base_level_uncertainty(self, station_key, year):
        info = self.get_station_info(station_key)
        return info.get('base_level_uncertainties', {}).get(year)

    def get_base_level_date(self, station_key, year):
        info = self.get_station_info(station_key)
        return info.get('base_level_dates', {}).get(year)

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
        return df

    def correct_water_level(self, wl_df, atmo_df):
        atmo_df = atmo_df.sort_values('datetime')
        wl_df = wl_df.sort_values('datetime')
        merged = pd.merge_asof(wl_df, atmo_df, on="datetime", direction="nearest", suffixes=('', '_atmo'))
        merged['wl_corr'] = merged['pressure'] - merged['pressure_atmo']
        return merged

    def adjust_base_level(self, df, station_key, year):
        base_val = self.get_base_level(station_key, year)
        if base_val is not None:
            first_valid = df['wl_corr'].dropna().iloc[0] if not df['wl_corr'].dropna().empty else 0
            correction = base_val - first_valid
            df['wl_final'] = df['wl_corr'] + correction
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
            raise FileNotFoundError(
                f"No water file found for station '{station_key}', year '{year_str}'. Files checked: {water_files}")
        if atmo_file is None:
            raise FileNotFoundError(
                f"No atmospheric file found for station '{station_key}', year '{year_str}'. Files checked: {atmo_files}")

        info = self.get_station_info(station_key)
        water_fullpath = self.main_data_path / info['wl_folder'] / water_file
        atmo_fullpath = self.main_data_path / info['atmo_folder'] / atmo_file

        df_wl = self.read_with_timezone(water_fullpath)
        df_atmo = self.read_with_timezone(atmo_fullpath)
        self.df_corr = self.correct_water_level(df_wl, df_atmo)
        df_final = self.adjust_base_level(self.df_corr, station_key, year)
        return df_final



# Conversion UNITS
# kPa to cmH2O
#

h = WaterLevelHomogenizer("params_simple_2025-11-22T1812.yml")
station = "KG_G354-new"
# station = "KG_G354-old"
# year = "2019"
year = "2024"


# h.get_file_list(station,'wl')
# h.select_files(h.get_file_list(station,'wl'),"2024",'wl')
# h.extract_years('10662647_-_G354_water_2019-2021.csv')
# h.extract_level('10662647_-_G354_water_2019-2021.csv') is None

df_final = h.process_year(station, year)

h.adjust_base_level(h.df_corr,station,year)

h.write_to_csv(df_final,"test_g354-5.csv")

