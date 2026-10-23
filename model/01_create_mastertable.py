"""Create a BTC master table for the thesis.

Creates the ten BTC metrics and the input columns required by
BTC_metric_plots_area_normalized.py and 13_model_new_corrected.py.

"""

# libraries
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd


# 1. SETTINGS ---------------------------------------------------------------


# defne the base folder
BASE_DIR = Path(__file__).resolve().parent

# define the input data
INPUT_DIR = BASE_DIR / "btc_background_corrected"

# define where the output should be stored 
# this can be changed
OUTPUT_DIR = Path.home() / "BTC_output"

# output stored as a .csv for further analysis steps
OUTPUT_CSV = OUTPUT_DIR / "BTC_mastertable.csv"

# output stored as a .xlsx for a better overview
OUTPUT_XLSX = OUTPUT_DIR / "BTC_mastertable.xlsx"


# definition of the uncalibrated fluorometer-values
# the "LA_mV_clean_manal" is for the self-adjusted BTCs where the semi-automatic approach cut-off was to early.
SIGNAL_COLUMNS = ("LA_mV_clean_manual", "LA_mV_clean")

# Tail starts at 10% of peak on the falling limb
TAIL_THRESHOLD = 0.10  

# Calibration: concentration = clean LA_mV * station-specific factor.
CALIBRATION = {901: 7.35e-11, 902: 4.90e-11, 904: 6.60e-11, 907: 7.20e-11}

# dictionary with the correct station names for each device
STATIONS = {901: "MS1", 902: "MS2", 904: "MS3", 907: "MS4"}


STREAMS = {
    "starzeln_s": (1, "Starzeln"),
    "leitschachbach_l": (2, "Leitschachbach"),
    "gangbach_g": (3, "Gangbach"),
}

# dictionary with the stream, station and the corresponding distance from the injection point
# the data for the distance was derived from geo.admin.ch
DISTANCES_M = {
    ("Starzeln", 901): 107.0, ("Starzeln", 902): 724.0,
    ("Starzeln", 904): 3944.0, ("Starzeln", 907): 4521.0,
    ("Gangbach", 901): 380.0, ("Gangbach", 902): 796.0,
    ("Gangbach", 904): 1626.0, ("Gangbach", 907): 2224.0,
    ("Leitschachbach", 901): 323.0, ("Leitschachbach", 902): 1007.0,
    ("Leitschachbach", 904): 1400.0, ("Leitschachbach", 907): 1986.0,
}
# these timestamps must have the same timezone convention as CSV's 'time'
INJECTION_TIMES_UTC = {
    (1, 1): "2026-04-16 11:10:00",
    (3, 1): "2026-05-13 12:45:00",
    (3, 2): "2026-05-15 09:21:00",
    (3, 3): "2026-05-17 11:45:00",
    (2, 1): "2026-05-21 12:00:00",
    (2, 2): "2026-05-23 10:57:00",
    (2, 3): "2026-05-25 12:52:00",
}
FILENAME = re.compile(
    r"^(starzeln_s|leitschachbach_l|gangbach_g)"
    r"_(901|902|904|907)_measurement_(\d+)_LA_mV_recleaned\.csv$",
    re.IGNORECASE,
)


# 2. BTC CALCULATIONS
def integral(t, y):
    """Trapezoidal integration, supporting older NumPy versions."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, t))
    return float(np.trapz(y, t))

# this function calculates how much tracer has already passed the monitoring station
# this is the fundation for metrics like t10 or t90
def quantile_time(t, c, q):
   
    segments = 0.5 * (c[:-1] + c[1:]) * np.diff(t)
    cumulative = np.r_[0.0, np.cumsum(segments)]
    if cumulative[-1] <= 0:
        return np.nan
    return float(np.interp(q * cumulative[-1], cumulative, t))

# this function calculates the FWHM, it gets interpolated between measurement points
def fwhm(t, c):
    
    peak = int(np.argmax(c))
    half = 0.5 * c[peak]
    if half <= 0:
        return np.nan

    left = right = np.nan
    for i in range(peak - 1, -1, -1):
        if c[i] <= half <= c[i + 1]:
            left = t[i] + (half - c[i]) * (t[i + 1] - t[i]) / (c[i + 1] - c[i])
            break
    for i in range(peak, len(c) - 1):
        if c[i] >= half >= c[i + 1]:
            right = t[i] + (half - c[i]) * (t[i + 1] - t[i]) / (c[i + 1] - c[i])
            break
    return float(right - left) if np.isfinite(left) and np.isfinite(right) else np.nan


# calculates the temporal variance and skewness of the BTC
def btc_moments(t, c, area):
    """Temporal variance and skewness, weighted by concentration."""
    mean = integral(t, t * c) / area
    variance = max(integral(t, (t - mean) ** 2 * c) / area, 0.0)
    sigma = np.sqrt(variance)
    skewness = integral(t, (t - mean) ** 3 * c) / area / sigma**3 if sigma > 0 else np.nan
    return variance, float(skewness)

# Calculate the fraction of the BTC area after the concentration first falls below 10% of the peak concentration
def tail_fraction(t, c, area):
    """Area after the FIRST fall below 10% of peak, divided by total area."""
    peak = int(np.argmax(c))
    threshold = TAIL_THRESHOLD * c[peak]
    start = next((i for i in range(peak + 1, len(c)) if c[i] <= threshold), None)
    return integral(t[start:], c[start:]) / area if start is not None else np.nan


# 3. ONE MEASURED CURVE

# this function extracts stream, device and injection information from the filename
def analyse_file(path):
    match = FILENAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected filename: {path.name}")
    prefix, device_text, injection_text = match.groups()
    stream_id, stream = STREAMS[prefix.lower()]
    device, injection = int(device_text), int(injection_text)
    injection_key = (stream_id, injection)

    # Load the cleaned BTC data and remove invalid measurements.
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)
    clean_col = next((col for col in SIGNAL_COLUMNS if col in df.columns), None)
    if clean_col is None or "time" not in df.columns:
        raise ValueError(f"Missing 'time' or cleaned signal in {path.name}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df[clean_col] = pd.to_numeric(df[clean_col], errors="coerce")
    df = df.dropna(subset=["time", clean_col]).sort_values("time").reset_index(drop=True)
    if len(df) < 2:
        raise ValueError(f"Too few valid measurements in {path.name}")

    # calculates time since injection and convert the cleaned signal to concentration
    injection_time = pd.to_datetime(INJECTION_TIMES_UTC[injection_key])
    t = (df["time"] - injection_time).dt.total_seconds().to_numpy(float)
    c = np.maximum(df[clean_col].to_numpy(float), 0.0) * CALIBRATION[device]
    area = integral(t, c)
    if area <= 0 or not np.isfinite(area):
        raise ValueError(f"No positive BTC area in {path.name}")

    # calculates arrival times, BTC quantiles and statistical moments
    positive = np.flatnonzero(c > 0)
    peak = int(np.argmax(c))
    distance = DISTANCES_M[(stream, device)]
    first_time = float(t[positive[0]])
    peak_time = float(t[peak])
    t10 = quantile_time(t, c, 0.10)
    t50 = quantile_time(t, c, 0.50)
    t90 = quantile_time(t, c, 0.90)
    variance, skewness = btc_moments(t, c, area)

    # returns the calculated BTC metrics for the master table

    return {
        "Stream_ID": stream_id,
        "Stream": stream,
        "Injection": injection,
        "Injection_time_UTC": injection_time,
        "Device_number": device,
        "Monitoring_station": STATIONS[device],
        "Distance_m": distance,
        "Source_file": path.name,
        "Cleaning_column_used": clean_col,
        "First_arrival_time_s": first_time,
        "First_arrival_velocity_m_s": distance / first_time if first_time > 0 else np.nan,
        "Peak_time_s": peak_time,
        "t10_s": t10,
        "t50_s": t50,
        "t90_s": t90,
        "Interquantile_duration_t90_t10_s": t90 - t10,
        "FWHM_s": fwhm(t, c),
        "Variance_s2": variance,
        "Skewness": skewness,
        "Quantile_asymmetry_ratio": (t90 - t50) / (t50 - t10) if t50 > t10 else np.nan,
        "Peak_to_median_lag_s": t50 - peak_time,
        "tail_fraction": tail_fraction(t, c, area),
        "Peak_concentration": float(c[peak]),
        "Area_concentration_time": area,
        "Peak_concentration_area_normalized": float(c[peak] / area),
    }

# this function calculates t50-based velocities between consecutive monitoring stations
def add_sectional_velocities(master):
 
    master["Section_from"] = ""
    master["Sectional_velocity_m_s"] = np.nan

    # calculates velocities separately for each stream and injection
    for _, group in master.groupby(["Stream", "Injection"], sort=False):
        previous = None
        for idx in group.index:
            if previous is not None:
                dx = master.at[idx, "Distance_m"] - master.at[previous, "Distance_m"]
                dt = master.at[idx, "t50_s"] - master.at[previous, "t50_s"]
                master.at[idx, "Section_from"] = master.at[previous, "Monitoring_station"]

                # calculates velocity only for positive distance and time differences
                if dx > 0 and dt > 0:
                    master.at[idx, "Sectional_velocity_m_s"] = dx / dt
            previous = idx
    return master


# 4. PROCESS BTC FILES AND SAVE MASTER TABLE
def main():
    files = sorted(INPUT_DIR.glob("*_LA_mV_recleaned.csv"))
    if not files:
        raise FileNotFoundError(f"No cleaned BTC files found in {INPUT_DIR}")

    rows, errors = [], []
    for path in files:
        try:
            rows.append(analyse_file(path))
        except Exception as exc:
            errors.append({"Source_file": path.name, "Error": str(exc)})
            warnings.warn(f"Skipped {path.name}: {exc}")

    if not rows:
        examples = "\n".join(
            f"  {item['Source_file']}: {item['Error']}"
            for item in errors[:5]
        )
        raise RuntimeError(
            f"No BTC files could be processed. {len(files)} files found.\n"
            f"First processing errors:\n{examples}"
        )

    master = pd.DataFrame(rows).sort_values(
        ["Stream_ID", "Injection", "Distance_m", "Device_number"]
    ).reset_index(drop=True)
    master = add_sectional_velocities(master)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUTPUT_CSV, sep=";", index=False)
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        master.to_excel(writer, sheet_name="BTC_mastertable", index=False)
        if errors:
            pd.DataFrame(errors).to_excel(writer, sheet_name="Processing_errors", index=False)

    print(f"Analyzed BTCs: {len(master)}; errors: {len(errors)}")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"Excel: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
