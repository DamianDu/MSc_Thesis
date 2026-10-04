# important libraries for model
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# selection of the stream ("Gangbach", "Leitschachbach" and "Starzeln" are possible)
# and injection number
STREAM = "Starzeln"
INJECTION = 1

# path to the data derived from the measurements
MASTERFILE = Path("BTC_mastertable.xlsx")

#  path to the BTCs (corrected)
BTC_FOLDER = Path("btc_background_corrected")

# output folder of the result
# a new run of the model deletes previous results, has to be adjusted for each new run
# the following folder names are used for the analysis
# multiple simulations for the corresponding version (vX) at the end
# "Gangbach_inj2_model_output_v1"
# "Gangbach_inj3_model_output_v1"
# "Leitschachbach_inj2_model_output_v1"
# "Leitschachbach_inj3_model_output_v1"
# "Starzeln_inj1_model_output_v1"
OUTPUT_DIR = Path("Starzeln_inj1_model_output_v2")

# core information about the simulated BTC from the model
RANDOM_SEED = 42
N_PARTICLES = 10000
DT = 0.2
T_MAX = 50000.0
BIN_WIDTH = 20.0

#Gangbach = 1.66
#Leitschach_normal = 1.39
#Leitschach_high = 4.5
#starzeln = 3.26

# the relative submergence is derived from the field data, it should be changed and the corresponding value of each stream/injection should be taken.
RELATIVE_SUBMERGE = 1.5

# global scaling factor of the velocity
VELOCITY_SCALE = 0.87

# adjusted velocity based on the transport state
VELOCITY_FACTORS = (
    VELOCITY_SCALE
    * np.array([1.5, 1.0, 0.2])
)

# initial division of the particles in their transport zone-> format (fast, intermediate, storage)
MODEL_FRACTIONS = np.array([0.0, 1.0, 0.0])

# Dispersion calibration:
# D_SCALE = 1.0 means that the FWHM-derived empirical dispersion trend is
# transferred without an additional global scaling of the random walk.
D_SCALE = 1.0

# Relative dispersion between transport states.
# The intermediate zone is the empirical reference zone.
D_ZONE_FACTORS = np.array([0.7, 1.0, 0.3], dtype=float)

# Distance-dependent dispersion:
# A single increase coefficient beta_D [m/s] is estimated from the observed
# station-wise FWHM-derived dispersion values. The first observed station
# provides D0. Upstream of MS1, D = D0. Downstream, D grows linearly with
# distance and is capped at the largest observed D_FWHM to avoid extrapolation.
USE_DISTANCE_DEPENDENT_DISPERSION = True

# Storage-exchange calibration:
# intermediate -> storage: controls mainly how much tracer enters storage / tail
STORAGE_IN_SCALE = 1.0

# storage -> intermediate: controls mainly how quickly tracer returns
# smaller value = longer residence time / longer tail
STORAGE_OUT_SCALE = 0.05


ZONES = ["fast core", "intermediate", "storage"]
CLEAN_COLUMNS = ["LA_mV_clean_manual", "LA_mV_clean"]

STREAM_PREFIX = {
    "Starzeln": "starzeln_s",
    "Gangbach": "gangbach_g",
    "Leitschachbach": "leitschachbach_l",
}

def exchange_matrix(relative_submerge):
    """
    k[i, j] = transition rate from SOURCE zone i to TARGET zone j [1/s]

    Zone order:
        0 = fast core
        1 = intermediate
        2 = storage

    Only the two storage-related transitions are calibrated here:
        k[1, 2]: intermediate -> storage
        k[2, 1]: storage -> intermediate
    """
    # this matrix is based on the mean value of the injections with normal/compareable water level
    k0 = np.array([
        [0.0, 0.003072461, 0.0],
        [0.001536231, 0.0, 0.000140479],
        [0.0, 0.008727661, 0.0],
    ], dtype=float)

    # Calibrate only storage entry and storage return.
    k0[1, 2] *= STORAGE_IN_SCALE
    k0[2, 1] *= STORAGE_OUT_SCALE

    F = np.array([
        [0.0, 1.0 + 0.8*relative_submerge, np.exp(-0.8*relative_submerge)],
        [1.0 + 0.5*relative_submerge, 0.0, np.exp(-0.6*relative_submerge)],
        [1.0 + 0.3*relative_submerge, 1.0 + 0.7*relative_submerge, 0.0],
    ])

    return k0 * F

def load_master_rows():
    if not MASTERFILE.exists():
        raise FileNotFoundError(f"Masterfile not found: {MASTERFILE.resolve()}")

    df = pd.read_excel(MASTERFILE)

    required = [
        "Stream", "Injection", "Device_number", "Monitoring_station",
        "Distance_m", "Injection_time_UTC", "Peak_time_s", "t50_s",
        "FWHM_s", "Variance_s2"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required masterfile columns: {missing}")

    for col in ["Injection", "Device_number", "Distance_m", "Peak_time_s", "t50_s", "FWHM_s", "Variance_s2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    subset = df.loc[
        (df["Stream"].astype(str).str.strip() == STREAM)
        & (df["Injection"] == INJECTION)
    ].copy()

    subset = subset.dropna(
        subset=["Device_number", "Distance_m", "Peak_time_s", "t50_s", "FWHM_s", "Variance_s2"]
    ).sort_values("Distance_m").reset_index(drop=True)

    if subset.empty:
        raise RuntimeError(f"No usable rows found for {STREAM}, injection {INJECTION}.")
    return subset

def estimate_empirical_dispersion(df):
    out = df.copy()

    out["Peak_velocity_from_injection_m_s"] = np.where(
        (out["Distance_m"] > 0) & (out["Peak_time_s"] > 0),
        out["Distance_m"] / out["Peak_time_s"],
        np.nan,
    )

    out["t50_velocity_from_injection_m_s"] = np.where(
        (out["Distance_m"] > 0) & (out["t50_s"] > 0),
        out["Distance_m"] / out["t50_s"],
        np.nan,
    )

    out["sigma_from_FWHM_s"] = out["FWHM_s"] / 2.355

    out["D_from_FWHM_m2_s"] = (
        out["sigma_from_FWHM_s"]**2
        * out["Peak_velocity_from_injection_m_s"]**3
        / (2.0 * out["Distance_m"])
    )

    out["D_from_variance_m2_s"] = (
        out["Variance_s2"]
        * out["Peak_velocity_from_injection_m_s"]**3
        / (2.0 * out["Distance_m"])
    )

    out["D_variance_to_FWHM_ratio"] = np.where(
        out["D_from_FWHM_m2_s"] > 0,
        out["D_from_variance_m2_s"] / out["D_from_FWHM_m2_s"],
        np.nan,
    )
    return out

def estimate_dispersion_distance_trend(df):
    """
    Estimate a simple distance-dependent reference dispersion:

        D_ref(x) = D0 + beta_D * max(x - x0, 0)

    where:
        x0 = distance to the first available monitoring station
        D0 = FWHM-derived dispersion at that station
        beta_D = least-squares increase coefficient derived from all
                 available station-wise D_FWHM values

    The trend is anchored at the first station so that the very sharp first
    BTC is not artificially broadened by a downstream median D value.

    This is an empirical parameterization of downstream spreading, not a
    direct measurement of a local physical dispersion coefficient.
    """
    work = df[["Distance_m", "D_from_FWHM_m2_s"]].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    work = work.loc[
        (work["Distance_m"] >= 0)
        & (work["D_from_FWHM_m2_s"] > 0)
    ].sort_values("Distance_m")

    if work.empty:
        raise RuntimeError("No valid FWHM-based dispersion estimates.")

    x = work["Distance_m"].to_numpy(dtype=float)
    D = work["D_from_FWHM_m2_s"].to_numpy(dtype=float)

    x0 = float(x[0])
    D0 = float(D[0])

    dx = x - x0
    dD = D - D0

    if len(work) >= 2 and np.sum(dx**2) > 0:
        # Least-squares slope with the trend constrained through (x0, D0).
        beta_D = float(np.sum(dx * dD) / np.sum(dx**2))
    else:
        beta_D = 0.0

    # For the present model we only allow a downstream increase.
    beta_D = max(beta_D, 0.0)

    D_cap = float(np.nanmax(D))

    diagnostics = work.copy()
    diagnostics["D_distance_trend_m2_s"] = (
        D0 + beta_D * np.maximum(diagnostics["Distance_m"] - x0, 0.0)
    )
    diagnostics["D_distance_trend_m2_s"] = np.minimum(
        diagnostics["D_distance_trend_m2_s"],
        D_cap,
    )

    return {
        "x0_m": x0,
        "D0_m2_s": D0,
        "beta_D_m_s": beta_D,
        "D_cap_m2_s": D_cap,
        "diagnostics": diagnostics,
    }


def reference_dispersion_at_distance(x, dispersion_trend):
    """Return the empirical reference D at particle position x."""
    x = np.asarray(x, dtype=float)

    if not USE_DISTANCE_DEPENDENT_DISPERSION:
        return np.full_like(x, dispersion_trend["D0_m2_s"], dtype=float)

    D_ref = (
        dispersion_trend["D0_m2_s"]
        + dispersion_trend["beta_D_m_s"]
        * np.maximum(x - dispersion_trend["x0_m"], 0.0)
    )

    D_ref = np.clip(
        D_ref,
        dispersion_trend["D0_m2_s"],
        dispersion_trend["D_cap_m2_s"],
    )
    return D_ref


def calculate_section_velocities(df):
    distances = df["Distance_m"].to_numpy(dtype=float, copy=True)
    peak_time = df["Peak_time_s"].to_numpy(dtype=float, copy=True)
    t50 = df["t50_s"].to_numpy(dtype=float, copy=True)

    start_x = np.concatenate(([0.0], distances[:-1]))
    end_x = distances.copy()
    start_peak = np.concatenate(([0.0], peak_time[:-1]))
    end_peak = peak_time.copy()
    start_t50 = np.concatenate(([0.0], t50[:-1]))
    end_t50 = t50.copy()

    dx = end_x - start_x
    dpeak = end_peak - start_peak
    dt50 = end_t50 - start_t50

    invalid = (~np.isfinite(dx)) | (~np.isfinite(dpeak)) | (dx <= 0) | (dpeak <= 0)
    if np.any(invalid):
        raise RuntimeError(
            "Invalid peak-based sectional velocity calculation. "
            "Distance and Peak_time_s must increase downstream. "
            f"Bad section indices: {np.where(invalid)[0].tolist()}"
        )

    v_peak_section = dx / dpeak
    v_t50_section = np.where((np.isfinite(dt50)) & (dt50 > 0), dx / dt50, np.nan)
    u_section_zones = v_peak_section[:, None] * VELOCITY_FACTORS[None, :]

    names = df["Monitoring_station"].astype(str).tolist()

    section_table = pd.DataFrame({
        "Section_number": np.arange(1, len(distances) + 1),
        "Section_from": ["Injection"] + names[:-1],
        "Section_to": names,
        "Start_distance_m": start_x,
        "End_distance_m": end_x,
        "Section_length_m": dx,
        "Start_peak_time_s": start_peak,
        "End_peak_time_s": end_peak,
        "Delta_peak_time_s": dpeak,
        "Section_peak_velocity_m_s": v_peak_section,
        "Start_t50_s": start_t50,
        "End_t50_s": end_t50,
        "Delta_t50_s": dt50,
        "Section_t50_velocity_m_s": v_t50_section,
        "Fast_velocity_m_s": u_section_zones[:, 0],
        "Intermediate_velocity_m_s": u_section_zones[:, 1],
        "Storage_velocity_m_s": u_section_zones[:, 2],
    })
    return section_table, u_section_zones

def find_btc_file(device):
    device = int(device)
    prefix = STREAM_PREFIX[STREAM]
    filename = f"{prefix}_{device}_measurement_{INJECTION}_LA_mV_recleaned.csv"
    path = BTC_FOLDER / filename
    if not path.exists():
        raise FileNotFoundError(f"BTC file not found: {path.resolve()}")
    return path

def detect_time_column(df):
    for col in ["Time", "time", "datetime", "Datetime", "DateTime", "timestamp", "Timestamp"]:
        if col in df.columns:
            return col
    for col in df.columns:
        low = str(col).lower()
        if "time" in low or "date" in low:
            return col
    raise KeyError("No time column found.")

def load_measured_btc(path, injection_time):
    df = pd.read_csv(path)
    clean_col = next((c for c in CLEAN_COLUMNS if c in df.columns), None)
    if clean_col is None:
        raise KeyError(f"No cleaned signal column in {path.name}.")

    time_col = detect_time_column(df)
    timestamp = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    injection_time = pd.to_datetime(injection_time, utc=True)

    t_s = (timestamp - injection_time).dt.total_seconds().to_numpy(dtype=float, copy=True)
    signal = pd.to_numeric(df[clean_col], errors="coerce").to_numpy(dtype=float, copy=True)

    valid = np.isfinite(t_s) & np.isfinite(signal) & (t_s >= 0)
    t_s = t_s[valid]
    signal = np.maximum(signal[valid], 0.0)

    order = np.argsort(t_s)
    t_s = t_s[order]
    signal = signal[order]

    if len(t_s) < 2:
        raise RuntimeError(f"Too few valid measured points in {path.name}.")

    area = np.trapezoid(signal, t_s)
    if not np.isfinite(area) or area <= 0:
        raise RuntimeError(f"Measured BTC has zero/invalid area: {path.name}")

    return t_s, signal / area

def initialize_particle_zones(n_particles, fractions, rng):
    fractions = np.asarray(fractions, dtype=float)
    if len(fractions) != 3:
        raise ValueError("MODEL_FRACTIONS must contain exactly 3 values.")
    if np.any(fractions < 0):
        raise ValueError("MODEL_FRACTIONS cannot contain negative values.")
    if not np.isclose(fractions.sum(), 1.0):
        raise ValueError("MODEL_FRACTIONS must sum to 1.0.")
    return rng.choice(len(ZONES), size=n_particles, p=fractions)

def run_model(distances, u_section_zones, dispersion_trend):
    rng = np.random.default_rng(RANDOM_SEED)
    distances = np.asarray(distances, dtype=float)
    L_max = float(distances.max())
    k = exchange_matrix(RELATIVE_SUBMERGE)

    x = np.zeros(N_PARTICLES, dtype=float)
    zone = initialize_particle_zones(N_PARTICLES, MODEL_FRACTIONS, rng)

    arrival_times = {float(dist): [] for dist in distances}
    reached = {float(dist): np.zeros(N_PARTICLES, dtype=bool) for dist in distances}

    n_steps = int(T_MAX / DT)

    for step in range(n_steps):
        t = step * DT
        active = x < L_max
        if not np.any(active):
            break

        idx = np.where(active)[0]
        # Freeze the source states for movement and exchange in this time step.
        zi = zone[idx].copy()

        section_idx = np.searchsorted(distances, x[idx], side="right")
        section_idx = np.clip(section_idx, 0, len(distances) - 1)

        particle_velocity = u_section_zones[section_idx, zi]

        advective_step = particle_velocity * DT

        # Reference dispersion increases with downstream distance according
        # to the empirical FWHM-derived trend. Each transport zone receives
        # its fixed relative share of that local reference value.
        D_ref_particle = reference_dispersion_at_distance(
            x[idx],
            dispersion_trend,
        )
        zone_scale = D_SCALE * D_ZONE_FACTORS[zi]
        D_particle = zone_scale * D_ref_particle

        # Fickian drift correction for spatially variable dispersion:
        # dx = (u_z + dD_z/dx) * dt + sqrt(2 * D_z * dt) * xi.
        # The reference gradient is zero upstream of x0 and at/above the cap.
        dD_ref_dx = np.zeros_like(D_ref_particle)
        if USE_DISTANCE_DEPENDENT_DISPERSION:
            increasing = (
                (x[idx] > dispersion_trend["x0_m"])
                & (D_ref_particle < dispersion_trend["D_cap_m2_s"])
            )
            dD_ref_dx[increasing] = dispersion_trend["beta_D_m_s"]

        # Apply the same scaling to the gradient as to D itself.
        dD_particle_dx = zone_scale * dD_ref_dx
        dispersion_drift_step = dD_particle_dx * DT

        random_step = (
            np.sqrt(2.0 * D_particle * DT)
            * rng.standard_normal(len(idx))
        )

        x[idx] += advective_step + dispersion_drift_step + random_step
        x[idx] = np.maximum(x[idx], 0.0)

        for i in range(len(ZONES)):
            # Use the frozen source states, not states changed by earlier
            # iterations of this loop. Each particle gets one exchange trial.
            particles_i = idx[zi == i]
            if len(particles_i) == 0:
                continue

            probs = k[i] * DT
            total_prob = probs.sum()

            if total_prob <= 0:
                continue
            if total_prob > 1:
                raise RuntimeError(f"Switch probability > 1 for zone {i}. Reduce DT or exchange rates.")

            switching = rng.random(len(particles_i)) < total_prob
            switching_particles = particles_i[switching]

            if len(switching_particles) > 0:
                target_probs = probs / total_prob
                new_zones = rng.choice(
                    len(ZONES),
                    size=len(switching_particles),
                    p=target_probs,
                )
                zone[switching_particles] = new_zones

        for dist in distances:
            dist = float(dist)
            newly_arrived = (x >= dist) & (~reached[dist])

            if np.any(newly_arrived):
                arrival_times[dist].extend([t] * int(np.sum(newly_arrived)))
                reached[dist][newly_arrived] = True

    return {dist: np.asarray(times, dtype=float) for dist, times in arrival_times.items()}

def model_pdf_from_arrivals(arrivals):
    bins = np.arange(0, T_MAX + BIN_WIDTH, BIN_WIDTH)
    time_mid = (bins[:-1] + bins[1:]) / 2.0

    counts, _ = np.histogram(arrivals, bins=bins)
    model_pdf = counts / (N_PARTICLES * BIN_WIDTH)

    area = np.sum(model_pdf * BIN_WIDTH)
    if area > 0:
        model_pdf = model_pdf / area

    return time_mid, model_pdf

def comparison_metrics(model_t, model_pdf, obs_t, obs_pdf):
    model_on_obs = np.interp(obs_t, model_t, model_pdf, left=0.0, right=0.0)

    valid = np.isfinite(obs_pdf) & np.isfinite(model_on_obs)
    if valid.sum() == 0:
        return {"RMSE": np.nan, "MAE": np.nan}

    residual = model_on_obs[valid] - obs_pdf[valid]

    return {
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "MAE": float(np.mean(np.abs(residual))),
    }

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    master_rows = load_master_rows()
    master_rows = estimate_empirical_dispersion(master_rows)

    dispersion_trend = estimate_dispersion_distance_trend(master_rows)
    section_table, u_section_zones = calculate_section_velocities(master_rows)

    print()
    print("=" * 90)
    print(f"{STREAM}, injection {INJECTION}")
    print("=" * 90)
    print("DISTANCE-DEPENDENT DISPERSION")
    print(f"D0 at first station = {dispersion_trend['D0_m2_s']:.6f} m2/s")
    print(f"x0 = {dispersion_trend['x0_m']:.1f} m")
    print(f"beta_D = {dispersion_trend['beta_D_m_s']:.6e} m/s")
    print(f"D cap = {dispersion_trend['D_cap_m2_s']:.6f} m2/s")
    print(f"D_SCALE = {D_SCALE:.4f}")
    print(f"D zone factors = {D_ZONE_FACTORS}")
    print()

    k_effective = exchange_matrix(RELATIVE_SUBMERGE)
    k_in = k_effective[1, 2]
    k_out = k_effective[2, 1]
    tau_storage = np.inf if k_out <= 0 else 1.0 / k_out

    print("STORAGE EXCHANGE")
    print(f"STORAGE_IN_SCALE  = {STORAGE_IN_SCALE:.4f}")
    print(f"STORAGE_OUT_SCALE = {STORAGE_OUT_SCALE:.4f}")
    print(f"effective k intermediate -> storage = {k_in:.6e} 1/s")
    print(f"effective k storage -> intermediate = {k_out:.6e} 1/s")
    print(f"approx. storage residence time 1/k_out = {tau_storage:.1f} s")
    print()
    print("PEAK-BASED SECTION VELOCITIES")
    print(
        section_table[
            [
                "Section_from",
                "Section_to",
                "Section_length_m",
                "Delta_peak_time_s",
                "Section_peak_velocity_m_s",
                "Section_t50_velocity_m_s",
                "Fast_velocity_m_s",
                "Intermediate_velocity_m_s",
                "Storage_velocity_m_s",
            ]
        ].to_string(index=False)
    )
    print()

    distances = master_rows["Distance_m"].to_numpy(dtype=float, copy=True)
    arrival_times = run_model(distances, u_section_zones, dispersion_trend)

    n_stations = len(master_rows)

    fig, axes = plt.subplots(
        n_stations,
        1,
        figsize=(8.0, max(3.2 * n_stations, 5.0)),
        sharex=True,
    )

    if n_stations == 1:
        axes = [axes]

    result_rows = []

    for ax, (_, row) in zip(axes, master_rows.iterrows()):
        device = int(row["Device_number"])
        station = str(row["Monitoring_station"])
        dist = float(row["Distance_m"])

        btc_path = find_btc_file(device)
        obs_t, obs_pdf = load_measured_btc(btc_path, row["Injection_time_UTC"])

        model_t, model_pdf = model_pdf_from_arrivals(arrival_times[dist])
        metrics = comparison_metrics(model_t, model_pdf, obs_t, obs_pdf)

        ax.plot(obs_t, obs_pdf, linewidth=1.8, label="Measured BTC")
        ax.plot(model_t, model_pdf, linewidth=1.5, linestyle="--", label="Model BTC")

        ax.set_ylabel("Normalized BTC [1/s]")
        ax.set_title(
            f"{station}, {dist:.0f} m | "
            f"D_FWHM(obs)={row['D_from_FWHM_m2_s']:.3g} m2/s | "
            f"RMSE={metrics['RMSE']:.3e}"
        )
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

        result_rows.append({
            "Stream": STREAM,
            "Injection": INJECTION,
            "Monitoring_station": station,
            "Device_number": device,
            "Distance_m": dist,
            "Peak_time_s": row["Peak_time_s"],
            "t50_s": row["t50_s"],
            "FWHM_s": row["FWHM_s"],
            "Variance_s2": row["Variance_s2"],
            "Peak_velocity_from_injection_m_s": row["Peak_velocity_from_injection_m_s"],
            "t50_velocity_from_injection_m_s": row["t50_velocity_from_injection_m_s"],
            "sigma_from_FWHM_s": row["sigma_from_FWHM_s"],
            "D_from_FWHM_station_m2_s": row["D_from_FWHM_m2_s"],
            "D_from_variance_station_m2_s": row["D_from_variance_m2_s"],
            "D_variance_to_FWHM_ratio": row["D_variance_to_FWHM_ratio"],
            "D_scale": D_SCALE,
            "D0_distance_trend_m2_s": dispersion_trend["D0_m2_s"],
            "D_growth_beta_m_s": dispersion_trend["beta_D_m_s"],
            "D_cap_m2_s": dispersion_trend["D_cap_m2_s"],
            "D_reference_at_station_m2_s": float(
                reference_dispersion_at_distance(
                    np.array([dist]),
                    dispersion_trend,
                )[0]
            ),
            "D_fast_at_station_m2_s": float(
                D_SCALE * D_ZONE_FACTORS[0]
                * reference_dispersion_at_distance(
                    np.array([dist]),
                    dispersion_trend,
                )[0]
            ),
            "D_intermediate_at_station_m2_s": float(
                D_SCALE * D_ZONE_FACTORS[1]
                * reference_dispersion_at_distance(
                    np.array([dist]),
                    dispersion_trend,
                )[0]
            ),
            "D_storage_at_station_m2_s": float(
                D_SCALE * D_ZONE_FACTORS[2]
                * reference_dispersion_at_distance(
                    np.array([dist]),
                    dispersion_trend,
                )[0]
            ),
            "storage_in_scale": STORAGE_IN_SCALE,
            "storage_out_scale": STORAGE_OUT_SCALE,
            "effective_k_intermediate_to_storage_1_s": exchange_matrix(RELATIVE_SUBMERGE)[1, 2],
            "effective_k_storage_to_intermediate_1_s": exchange_matrix(RELATIVE_SUBMERGE)[2, 1],
            "fraction_fast_initial": MODEL_FRACTIONS[0],
            "fraction_intermediate_initial": MODEL_FRACTIONS[1],
            "fraction_storage_initial": MODEL_FRACTIONS[2],
            "RMSE_shape": metrics["RMSE"],
            "MAE_shape": metrics["MAE"],
        })

    axes[-1].set_xlabel("Time since injection [s]")
    fig.suptitle(
        f"{STREAM}, injection {INJECTION}: distance-dependent dispersion model vs measured BTC",
        y=1.002,
    )
    fig.tight_layout()

    stem = f"{STREAM}_injection_{INJECTION}_distance_dependent_dispersion_model_vs_tracer"

    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")

    pd.DataFrame(result_rows).to_excel(
        OUTPUT_DIR / f"{STREAM}_injection_{INJECTION}_peak_sectional_velocity_metrics.xlsx",
        index=False,
    )

    section_table.to_excel(
        OUTPUT_DIR / f"{STREAM}_injection_{INJECTION}_peak_section_velocities.xlsx",
        index=False,
    )

    master_rows.to_excel(
        OUTPUT_DIR / f"{STREAM}_injection_{INJECTION}_peak_dispersion_diagnostics.xlsx",
        index=False,
    )

    dispersion_trend["diagnostics"].to_excel(
        OUTPUT_DIR / f"{STREAM}_injection_{INJECTION}_distance_dispersion_trend.xlsx",
        index=False,
    )

    print(f"Results saved in: {OUTPUT_DIR.resolve()}")
    plt.show()

if __name__ == "__main__":
    main()
