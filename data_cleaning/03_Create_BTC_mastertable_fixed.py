from __future__ import annotations

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================
INPUT_DIR = Path("btc_background_corrected")
OUTPUT_CSV = Path("BTC_mastertable_corrected.csv")
OUTPUT_XLSX = Path("BTC_mastertable_corrected.xlsx")

TIME_COLUMN = "time"
CLEAN_COLUMN_CANDIDATES = ["LA_mV_clean_manual", "LA_mV_clean"]

# concentration = LA_mV * factor + offset
CALIBRATION = {
    901: (7.35e-11, 0.0),
    902: (4.90e-11, 0.0),
    904: (6.60e-11, 0.0),
    907: (7.20e-11, 0.0),
}

STATION_NAMES = {901: "MS1", 902: "MS2", 904: "MS3", 907: "MS4"}

STREAM_INFO = {
    "starzeln_s": (1, "Starzeln"),
    "leitschachbach_l": (2, "Leitschachbach"),
    "gangbach_g": (3, "Gangbach"),
}

DISTANCE_M = {
    ("Starzeln", 901): 102.0,
    ("Starzeln", 902): 710.0,
    ("Starzeln", 904): 3859.0,
    ("Starzeln", 907): 4433.0,
    ("Gangbach", 901): 362.0,
    ("Gangbach", 902): 755.0,
    ("Gangbach", 904): 1538.0,
    ("Gangbach", 907): 2108.0,
    ("Leitschachbach", 901): 299.0,
    ("Leitschachbach", 902): 906.0,
    ("Leitschachbach", 904): 1168.0,
    ("Leitschachbach", 907): 1758.0,
}

INJECTION_TIMES_UTC = {
    (1, 1): "2026-04-16 11:10:00",
    (3, 1): "2026-05-13 12:45:00",
    (3, 2): "2026-05-15 09:21:00",
    (3, 3): "2026-05-17 11:45:00",
    (2, 1): "2026-05-21 12:00:00",
    (2, 2): "2026-05-23 10:57:00",
    (2, 3): "2026-05-25 12:52:00",
}

# Optional: fill these dictionaries to calculate discharge/recovery.
INJECTED_MASS_MG: dict[tuple[int, int], float] = {}
REFERENCE_DISCHARGE_M3_S: dict[tuple[int, int], float] = {}

MIN_SNR = 3.0
MIN_POSITIVE_POINTS = 5
MIN_RECOVERY_PERCENT = 0.1
TAIL_START_FRACTION_OF_PEAK = 0.10
TAIL_FIT_MIN_POINTS = 5
TAIL_FIT_MIN_FRACTION = 1e-4

FILENAME_PATTERN = re.compile(
    r"^(?P<prefix>starzeln_s|leitschachbach_l|gangbach_g)"
    r"_(?P<device>901|902|904|907)"
    r"_measurement_(?P<injection>\d+)"
    r"_LA_mV_recleaned\.csv$",
    re.IGNORECASE,
)


def read_csv_auto(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = (
        df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)
    )
    return df


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.std(values, ddof=1)
    return float(sigma) if np.isfinite(sigma) else np.nan


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))


def parse_filename(path: Path) -> dict:
    m = FILENAME_PATTERN.match(path.name)
    if m is None:
        raise ValueError(f"Unexpected filename: {path.name}")
    prefix = m.group("prefix").lower()
    stream_id, stream = STREAM_INFO[prefix]
    device = int(m.group("device"))
    injection = int(m.group("injection"))
    return {
        "stream_id": stream_id,
        "stream": stream,
        "device": device,
        "station": STATION_NAMES[device],
        "injection": injection,
    }


def choose_clean_column(df: pd.DataFrame, path: Path) -> str:
    for col in CLEAN_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError(
        f"No cleaned column in {path.name}; expected {CLEAN_COLUMN_CANDIDATES}"
    )


def cumulative_quantile_time(t: np.ndarray, c: np.ndarray, q: float) -> float:
    if len(t) < 2:
        return np.nan
    seg = 0.5 * (c[:-1] + c[1:]) * np.diff(t)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    total = cum[-1]
    if not np.isfinite(total) or total <= 0:
        return np.nan
    return float(np.interp(q * total, cum, t))


def crossing(x1, y1, x2, y2, target) -> float:
    if y2 == y1:
        return float(x1)
    return float(x1 + (target - y1) * (x2 - x1) / (y2 - y1))


def calculate_fwhm(t: np.ndarray, c: np.ndarray) -> tuple[float, float, float]:
    peak = int(np.argmax(c))
    cmax = float(c[peak])
    if cmax <= 0:
        return np.nan, np.nan, np.nan
    half = 0.5 * cmax
    left = np.nan
    right = np.nan
    for i in range(peak - 1, -1, -1):
        if c[i] <= half <= c[i + 1]:
            left = crossing(t[i], c[i], t[i + 1], c[i + 1], half)
            break
    for i in range(peak, len(c) - 1):
        if c[i] >= half >= c[i + 1]:
            right = crossing(t[i], c[i], t[i + 1], c[i + 1], half)
            break
    width = right - left if np.isfinite(left) and np.isfinite(right) else np.nan
    return left, right, float(width)


def calculate_moments(t: np.ndarray, c: np.ndarray) -> dict:
    area = trapz(c, t)
    if not np.isfinite(area) or area <= 0:
        return dict(area=np.nan, mean=np.nan, variance=np.nan, std=np.nan,
                    skewness=np.nan, kurtosis=np.nan)
    mean = trapz(t * c, t) / area
    d = t - mean
    var = max(trapz(d**2 * c, t) / area, 0.0)
    std = np.sqrt(var)
    if std > 0:
        skew = trapz(d**3 * c, t) / area / std**3
        kurt = trapz(d**4 * c, t) / area / std**4 - 3.0
    else:
        skew = np.nan
        kurt = np.nan
    return dict(area=float(area), mean=float(mean), variance=float(var),
                std=float(std), skewness=float(skew), kurtosis=float(kurt))


def calculate_tail_metrics(t: np.ndarray, c: np.ndarray) -> dict:
    peak = int(np.argmax(c))
    cmax = float(c[peak])
    total_area = trapz(c, t)
    if cmax <= 0 or total_area <= 0:
        return dict(tail_start_time_s=np.nan, tail_fraction=np.nan,
                    tail_decay_lambda_1_s=np.nan, tail_timescale_s=np.nan,
                    tail_fit_r2=np.nan)
    threshold = TAIL_START_FRACTION_OF_PEAK * cmax
    start = next((i for i in range(peak + 1, len(c)) if c[i] <= threshold), None)
    if start is None:
        return dict(tail_start_time_s=np.nan, tail_fraction=np.nan,
                    tail_decay_lambda_1_s=np.nan, tail_timescale_s=np.nan,
                    tail_fit_r2=np.nan)
    tt = t[start:]
    cc = c[start:]
    tail_fraction = trapz(cc, tt) / total_area
    mask = np.isfinite(tt) & np.isfinite(cc) & (cc > cmax * TAIL_FIT_MIN_FRACTION)
    if mask.sum() < TAIL_FIT_MIN_POINTS:
        lam = tau = r2 = np.nan
    else:
        x = tt[mask] - tt[mask][0]
        y = np.log(cc[mask])
        slope, intercept = np.polyfit(x, y, 1)
        pred = intercept + slope * x
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        if slope < 0:
            lam = -float(slope)
            tau = 1.0 / lam
        else:
            lam = tau = np.nan
    return dict(tail_start_time_s=float(tt[0]), tail_fraction=float(tail_fraction),
                tail_decay_lambda_1_s=lam, tail_timescale_s=tau, tail_fit_r2=r2)


def background_metrics(df: pd.DataFrame, signal: np.ndarray) -> dict:
    out = dict(background_before_mV=np.nan, background_after_mV=np.nan,
               background_drift_mV=np.nan, noise_before_mV=np.nan,
               noise_after_mV=np.nan)
    if "LA_baseline_manual" in df.columns:
        b = pd.to_numeric(df["LA_baseline_manual"], errors="coerce").to_numpy(float)
        b = b[np.isfinite(b)]
        if len(b):
            out["background_before_mV"] = float(b[0])
            out["background_after_mV"] = float(b[-1])
            out["background_drift_mV"] = float(b[-1] - b[0])
    pos = np.flatnonzero(signal > 0)
    if len(pos):
        if pos[0] >= 3:
            out["noise_before_mV"] = robust_sigma(signal[:pos[0]])
        if pos[-1] + 3 < len(signal):
            out["noise_after_mV"] = robust_sigma(signal[pos[-1] + 1:])
    return out


def make_uiid(stream_id: int, injection: int, device: int) -> int:
    return stream_id * 10000 + injection * 100 + device % 100


def analyse_file(path: Path) -> dict:
    meta = parse_filename(path)
    stream_id = meta["stream_id"]
    stream = meta["stream"]
    device = meta["device"]
    injection = meta["injection"]

    df = read_csv_auto(path)
    clean_col = choose_clean_column(df, path)
    if TIME_COLUMN not in df.columns:
        raise KeyError(f"Missing '{TIME_COLUMN}' in {path.name}")

    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors="coerce")
    df[clean_col] = pd.to_numeric(df[clean_col], errors="coerce")
    df = df.dropna(subset=[TIME_COLUMN, clean_col]).sort_values(TIME_COLUMN).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No valid data in {path.name}")

    signal = df[clean_col].to_numpy(dtype=float, copy=True)
    signal[signal < 0] = 0.0
    factor, offset = CALIBRATION[device]
    concentration = signal * factor + offset
    concentration[concentration < 0] = 0.0

    key = (stream_id, injection)
    if key not in INJECTION_TIMES_UTC:
        raise KeyError(f"No injection time configured for {key}")
    injection_time = pd.to_datetime(INJECTION_TIMES_UTC[key])
    t = (df[TIME_COLUMN] - injection_time).dt.total_seconds().to_numpy(float)

    positive = np.flatnonzero(concentration > 0)
    if len(positive) == 0:
        raise ValueError(f"No positive BTC values in {path.name}")
    first, last = int(positive[0]), int(positive[-1])
    peak = int(np.argmax(concentration))

    moments = calculate_moments(t, concentration)
    t10 = cumulative_quantile_time(t, concentration, 0.10)
    t50 = cumulative_quantile_time(t, concentration, 0.50)
    t90 = cumulative_quantile_time(t, concentration, 0.90)
    fwhm_left, fwhm_right, fwhm = calculate_fwhm(t, concentration)
    tail = calculate_tail_metrics(t, concentration)
    bg = background_metrics(df, signal)

    noise_candidates = [bg["noise_before_mV"], bg["noise_after_mV"]]
    noise_candidates = [x for x in noise_candidates if np.isfinite(x) and x > 0]
    noise = float(np.median(noise_candidates)) if noise_candidates else np.nan
    peak_mV = float(signal[peak])
    snr = peak_mV / noise if np.isfinite(noise) and noise > 0 else np.nan

    distance = DISTANCE_M.get((stream, device), np.nan)
    peak_velocity = distance / t[peak] if np.isfinite(distance) and t[peak] > 0 else np.nan
    mean_velocity = distance / moments["mean"] if np.isfinite(distance) and moments["mean"] > 0 else np.nan
    t50_velocity = distance / t50 if np.isfinite(distance) and t50 > 0 else np.nan

    injected_mass = INJECTED_MASS_MG.get(key, np.nan)
    known_q = REFERENCE_DISCHARGE_M3_S.get(key, np.nan)
    q_est = injected_mass / moments["area"] / 1000.0 if np.isfinite(injected_mass) and moments["area"] > 0 else np.nan
    recovered_mass = known_q * 1000.0 * moments["area"] if np.isfinite(known_q) and moments["area"] > 0 else np.nan
    recovery = 100.0 * recovered_mass / injected_mass if np.isfinite(recovered_mass) and np.isfinite(injected_mass) and injected_mass > 0 else np.nan

    reasons = []
    if len(positive) < MIN_POSITIVE_POINTS:
        reasons.append(f"fewer than {MIN_POSITIVE_POINTS} positive points")
    if np.isfinite(snr) and snr < MIN_SNR:
        reasons.append(f"SNR below {MIN_SNR:g}")
    if np.isfinite(recovery) and recovery < MIN_RECOVERY_PERCENT:
        reasons.append(f"recovery below {MIN_RECOVERY_PERCENT:g}%")
    if not np.isfinite(fwhm):
        reasons.append("FWHM not available")

    return {
        "uiid": make_uiid(stream_id, injection, device),
        "Stream_ID": stream_id,
        "Stream": stream,
        "Injection": injection,
        "Injection_time_UTC": injection_time,
        "Device_number": device,
        "Monitoring_station": meta["station"],
        "Distance_m": distance,
        "Source_file": path.name,
        "Cleaning_column_used": clean_col,
        "Measurement_start": df[TIME_COLUMN].iloc[0],
        "Measurement_end": df[TIME_COLUMN].iloc[-1],
        "BTC_start_time": df[TIME_COLUMN].iloc[first],
        "BTC_end_time": df[TIME_COLUMN].iloc[last],
        "First_arrival_time_s": float(t[first]),
        "BTC_end_time_s": float(t[last]),
        "BTC_duration_s": float(t[last] - t[first]),
        "Positive_points": int(len(positive)),
        **bg,
        "Noise_used_mV": noise,
        "Signal_to_noise": snr,
        "Peak_mV": peak_mV,
        "Peak_concentration": float(concentration[peak]),
        "Peak_time_s": float(t[peak]),
        "Mean_arrival_time_s": moments["mean"],
        "t10_s": t10,
        "t50_s": t50,
        "t90_s": t90,
        "Interquantile_duration_t90_t10_s": t90 - t10 if np.isfinite(t90) and np.isfinite(t10) else np.nan,
        "Area_concentration_time": moments["area"],
        "Variance_s2": moments["variance"],
        "Std_s": moments["std"],
        "Skewness": moments["skewness"],
        "Kurtosis_excess": moments["kurtosis"],
        "FWHM_left_time_s": fwhm_left,
        "FWHM_right_time_s": fwhm_right,
        "FWHM_s": fwhm,
        "Peak_sharpness": float(concentration[peak]) / fwhm if np.isfinite(fwhm) and fwhm > 0 else np.nan,
        **tail,
        "Peak_velocity_m_s": peak_velocity,
        "Mean_velocity_m_s": mean_velocity,
        "t50_velocity_m_s": t50_velocity,
        "Injected_mass_mg": injected_mass,
        "Reference_discharge_m3_s": known_q,
        "Estimated_discharge_m3_s": q_est,
        "Recovered_mass_mg": recovered_mass,
        "Recovery_percent": recovery,
        "Auto_quality_pass": len(reasons) == 0,
        "Auto_quality_reason": "; ".join(reasons),
        "Selected_manual": "",
        "Exclusion_reason_manual": "",
        "Comments": "",
    }


if not INPUT_DIR.exists():
    raise FileNotFoundError(f"Input folder not found: {INPUT_DIR.resolve()}")

files = sorted(INPUT_DIR.glob("*_LA_mV_recleaned.csv"))
if not files:
    raise FileNotFoundError(f"No recleaned CSV files found in {INPUT_DIR.resolve()}")

rows, errors = [], []
for path in files:
    try:
        rows.append(analyse_file(path))
        print(f"Loaded: {path.name}")
    except Exception as exc:
        errors.append({"Source_file": path.name, "Error": str(exc)})
        warnings.warn(f"Could not process {path.name}: {exc}")

if not rows:
    raise RuntimeError("No BTC files could be processed.")

master = pd.DataFrame(rows).sort_values(
    ["Stream_ID", "Injection", "Distance_m", "Device_number"],
    na_position="last",
).reset_index(drop=True)

master.to_csv(OUTPUT_CSV, sep=";", index=False)

try:
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        master.to_excel(writer, sheet_name="BTC_mastertable", index=False)
        if errors:
            pd.DataFrame(errors).to_excel(writer, sheet_name="Processing_errors", index=False)
except ImportError:
    warnings.warn("openpyxl missing: XLSX output skipped; CSV was created.")

if errors:
    pd.DataFrame(errors).to_csv("BTC_mastertable_processing_errors.csv", sep=";", index=False)

print("\nMaster table completed.")
print(f"Successfully processed BTCs: {len(master)}")
print(f"Processing errors:            {len(errors)}")
print(f"CSV output:                   {OUTPUT_CSV.resolve()}")
if OUTPUT_XLSX.exists():
    print(f"Excel output:                 {OUTPUT_XLSX.resolve()}")
print("\nFill INJECTED_MASS_MG and REFERENCE_DISCHARGE_M3_S if recovery/discharge should be calculated.")
