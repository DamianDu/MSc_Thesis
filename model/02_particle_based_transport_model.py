# important libraries for model
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# selection of the stream ("Gangbach", "Leitschachbach" and "Starzeln" are possible)
# and injection number
STREAM = "Gangbach"
INJECTION = 2

# path to the data derived from the measurements (mastertable)
MASTERFILE = Path("BTC_mastertable.csv")

# path to the BTCs (corrected)
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
OUTPUT_DIR = Path("Gangbach_inj2_model_output_v2")

# core information about the simulated BTC from the model
# amout of particles
RANDOM_SEED = 42
N_PARTICLES = 10000

# timestep
DT = 0.2

# maximum time the model runs
T_MAX = 50000.0
BIN_WIDTH = 20.0


# values for the relative submergence
#Gangbach = 1.66
#Leitschach_normal = 1.39
#Leitschach_high = 4.5
#starzeln = 3.26

# the relative submergence is derived from the field data, it should be changed and the corresponding value of each stream/injection should be taken.
RELATIVE_SUBMERGE = 1.5

# global scaling factor of the velocity (see methods )
VELOCITY_SCALE = 0.87

# adjusted velocity based on the transport state (see chapter 4.6.1)
VELOCITY_FACTORS = (
    VELOCITY_SCALE
    * np.array([1.5, 1.0, 0.2])
)

# initial division of the particles in their transport zone-> format (fast, intermediate, storage) (see chapter 4.6.2)
MODEL_FRACTIONS = np.array([0.0, 1.0, 0.0])

# Dispersion calibration:
# D_SCALE = 1.0 means that the FWHM-derived empirical dispersion trend is
# transferred without an additional global scaling of the random walk.
D_SCALE = 1.0

# Relative dispersion between transport states (see chapter 4.6.2)
# The intermediate zone is the empirical reference zone.
D_ZONE_FACTORS = np.array([0.7, 1.0, 0.3], dtype=float)

# Distance-dependent dispersion:
# A single increase coefficient beta_D [m/s] is estimated from the observed
# station-wise FWHM-derived dispersion values. The first observed station
# provides D0. Upstream of MS1, D = D0. Downstream, D grows linearly with
# distance and is capped at the largest observed D_FWHM to avoid extrapolation.
# chapter 4.6.2 explains this step in further details.
USE_DISTANCE_DEPENDENT_DISPERSION = True

# Storage-exchange calibration:
# intermediate -> storage: controls mainly how much tracer enters storage / tail
# this was only added for a first test of the model, it should not be changed
# the storage_in_scale remains 1.0
# this step would only add more uncertainy and unclearity for the model
STORAGE_IN_SCALE = 1.0

# storage -> intermediate: controls mainly how quickly tracer returns
# smaller value = longer residence time / longer tail
# this values can be changed, it can also attribute to recirculation zones in streams
# this scale can be dependent on water level and the morphology of the stream
STORAGE_OUT_SCALE = 0.04


# definition of the transport states
ZONES = ["fast core", "intermediate", "storage"]

# definition of the raw and uncalibrated columns
CLEAN_COLUMNS = ["LA_mV_clean_manual", "LA_mV_clean"]

# dictionary with the prefix of the streams
STREAM_PREFIX = {
    "Starzeln": "starzeln_s",
    "Gangbach": "gangbach_g",
    "Leitschachbach": "leitschachbach_l",
}

def exchange_matrix(relative_submerge):
    """
    k[i, j] = probability of a state change from source state i to target zone j [1/s]

    State order:
        0 = fast state
        1 = intermediate state
        2 = storage state

    Only the two storage-related transitions are calibrated here:
        k[1, 2]: intermediate -> storage
        k[2, 1]: storage -> intermediate
    """
    # this matrix is based on the mean value of the injections with normal/compareable water level
    # therefore the third injection of the gangbach got excluded
    k0 = np.array([
        [0.0, 0.003742812309, 0.0],
        [0.001871406155, 0.0, 0.000156274940],
        [0.0, 0.011391571758, 0.0],
    ], dtype=float)

    # Calibrate only the probability of a change into the storage state and the probability of entering the intermediate state from the storage state.
    k0[1, 2] *= STORAGE_IN_SCALE
    k0[2, 1] *= STORAGE_OUT_SCALE

    # matrix with the addition of the relative_submergence
    # this leads to a matrix with the possibiliy of a more dynamic and flow height driven behaviour
    F = np.array([
        [0.0, 1.0 + 0.8*relative_submerge, np.exp(-0.8*relative_submerge)],
        [1.0 + 0.5*relative_submerge, 0.0, np.exp(-0.6*relative_submerge)],
        [1.0 + 0.3*relative_submerge, 1.0 + 0.7*relative_submerge, 0.0],
    ])

    return k0 * F


# the function load_master_rows loads the masterfile in
# this file contains valuable information for the model
# it also checks whethever the needed parameters are in the file and if the file exists
def load_master_rows():

    # checks if the file is available
    if not MASTERFILE.exists():
        raise FileNotFoundError(f"Masterfile not found: {MASTERFILE.resolve()}")

    # reads in the masterfile
    # seperator is only needed on for certain .csv file (depends on country and settings of the laptop)
    df = pd.read_csv(MASTERFILE, sep=';')

    # list of the required parameters/columns
    required = [
        "Stream", "Injection", "Device_number", "Monitoring_station",
        "Distance_m", "Injection_time_UTC", "Peak_time_s", "t50_s",
        "FWHM_s", "Variance_s2"
    ]

    # checks for missing columns and raises error if the file does not contain all the columns in the above list
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required masterfile columns: {missing}")

    # transforms the data needed for calculation into numeric
    # this step is important, otherwise, if certain columns are stored in a different format (e.g. string), calculations are not possible
    for col in ["Injection", "Device_number", "Distance_m", "Peak_time_s", "t50_s", "FWHM_s", "Variance_s2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")


    # Filter the data by stream and injection
    # Convert stream values to strings and remove leading/trailing whitespace
    # Check whether the values match the specified stream and injection
    subset = df.loc[
        (df["Stream"].astype(str).str.strip() == STREAM)
        & (df["Injection"] == INJECTION)
    ].copy()

    # delets the rows if there is no value in one of the columns in subset
    subset = subset.dropna(
        subset=["Device_number", "Distance_m", "Peak_time_s", "t50_s", "FWHM_s", "Variance_s2"]
    ).sort_values("Distance_m").reset_index(drop=True)

    # raises en error if no rows are available
    if subset.empty:
        raise RuntimeError(f"No usable rows found for {STREAM}, injection {INJECTION}.")
    return subset



# this function estimates the empirical dispersion (see chapter 4.6.2)
# is is a FWHM based approach for the dispersion estimation
# it is done for each stream and injection seperately
# the reason for that is the larger variabiliy in disperion
# Based on the results of the observed BTCs, there seems to be a stream and discharge specific response (see results)
def estimate_empirical_dispersion(df):

    out = df.copy()

    # Peak velocity from injection to monitoring station

    out["Peak_velocity_from_injection_m_s"] = np.where(
        (out["Distance_m"] > 0) & (out["Peak_time_s"] > 0),
        out["Distance_m"] / out["Peak_time_s"],
        np.nan,
    )

    # temporal standard deviation for the FWHM
    out["sigma_from_FWHM_s"] = out["FWHM_s"] / 2.355

    # calculation of the longitudinal disperion
    out["D_from_FWHM_m2_s"] = (
        out["sigma_from_FWHM_s"]**2
        * out["Peak_velocity_from_injection_m_s"]**3
        / (2.0 * out["Distance_m"])
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

    # creates a new variable named "work"
    # the needed values (FWHM-based dispersion and distance from injection point) values are copied from the dataframe 
    work = df[["Distance_m", "D_from_FWHM_m2_s"]].copy()
    # drops NAN values
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    # filters and sorts the values by distance 
    work = work.loc[
        (work["Distance_m"] >= 0)
        & (work["D_from_FWHM_m2_s"] > 0)
    ].sort_values("Distance_m")

    # raises error if there is no value for the estimation
    if work.empty:
        raise RuntimeError("No valid FWHM-based dispersion estimates.")

    # converts distances and FWHM-derived dispersion values to numpy arrays. 
    x = work["Distance_m"].to_numpy(dtype=float)
    D = work["D_from_FWHM_m2_s"].to_numpy(dtype=float)

    # define the first monitoring station as reference for distance and dispersion
    x0 = float(x[0])
    D0 = float(D[0])

    # calculates differences in distance and dispersion relative to the first station.
    dx = x - x0
    dD = D - D0


    # calculate the least-squares slope of D with distance, using the first monitoring station as a fixed reference.
    if len(work) >= 2 and np.sum(dx**2) > 0:
        # Least-squares slope with the trend constrained through (x0, D0).
        beta_D = float(np.sum(dx * dD) / np.sum(dx**2))
    else:
        beta_D = 0.0

    # for this model we only allow a downstream increase.
    # "beta_D" prevents decreasing dispersion along the stream.
    beta_D = max(beta_D, 0.0)

    # "D_cap" limits dispersion to the maximum observed FWHM-derived value.
    D_cap = float(np.nanmax(D))


    # calculates the slope
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

# the following function determines the dispersion of each particle at their current position
# the dispersion is based on the estimated distance dependent trend of the function "estimate_dispersion_distance_trend"
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


# calculates section velocities from the differences in peak arrival times
# velocities are calculated separately for each stream and injection
# the resulting velocities are used to define the three transport zones
def calculate_section_velocities(df):

    # converts the distance_m, peak_time_s and t50_s into numpy array in float format
    distances = df["Distance_m"].to_numpy(dtype=float, copy=True)
    peak_time = df["Peak_time_s"].to_numpy(dtype=float, copy=True)
    t50 = df["t50_s"].to_numpy(dtype=float, copy=True)


    # defines the start and end positions and arrival times of each section
    # the first section starts at the injection point (x = 0, t = 0)
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

    # Calculate peak-based and t50-based section velocities.
    v_peak_section = dx / dpeak
    v_t50_section = np.where((np.isfinite(dt50)) & (dt50 > 0), dx / dt50, np.nan)

    # Scale peak-based velocities for the three transport zones.
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


# finds the corresponding BTC file for the selected stream, injection and device, and check whether the file exists.
def find_btc_file(device):
    device = int(device)
    prefix = STREAM_PREFIX[STREAM]
    filename = f"{prefix}_{device}_measurement_{INJECTION}_LA_mV_recleaned.csv"
    path = BTC_FOLDER / filename
    if not path.exists():
        raise FileNotFoundError(f"BTC file not found: {path.resolve()}")
    return path


# checks whether a time column exists and return its name
def detect_time_column(df):
    for col in ["Time", "time", "datetime", "Datetime", "DateTime", "timestamp", "Timestamp"]:
        if col in df.columns:
            return col
    for col in df.columns:
        low = str(col).lower()
        if "time" in low or "date" in low:
            return col
    raise KeyError("No time column found.")



# this function loads and prepares the measured BTC for comparison with the model.
def load_measured_btc(path, injection_time):
    df = pd.read_csv(path)

    # selects the cleaned signal column
    clean_col = next((c for c in CLEAN_COLUMNS if c in df.columns), None)
    if clean_col is None:
        raise KeyError(f"No cleaned signal column in {path.name}.")


    # converts timestamps to time since injection in seconds
    time_col = detect_time_column(df)
    timestamp = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    injection_time = pd.to_datetime(injection_time, utc=True)

    t_s = (timestamp - injection_time).dt.total_seconds().to_numpy(dtype=float, copy=True)
    signal = pd.to_numeric(df[clean_col], errors="coerce").to_numpy(dtype=float, copy=True)


    # removes invalid and pre-injection values and sort by time
    valid = np.isfinite(t_s) & np.isfinite(signal) & (t_s >= 0)
    t_s = t_s[valid]
    signal = np.maximum(signal[valid], 0.0)

    order = np.argsort(t_s)
    t_s = t_s[order]
    signal = signal[order]

    if len(t_s) < 2:
        raise RuntimeError(f"Too few valid measured points in {path.name}.")

    # normalizes the BTC so that its integrated area equals 1
    area = np.trapezoid(signal, t_s)
    if not np.isfinite(area) or area <= 0:
        raise RuntimeError(f"Measured BTC has zero/invalid area: {path.name}")

    return t_s, signal / area



# this function initialize particles in the three transport zones according to the specified initial fractions.
def initialize_particle_zones(n_particles, fractions, rng):
    fractions = np.asarray(fractions, dtype=float)

    # checks if the fractions has exactely three values
    if len(fractions) != 3:
        raise ValueError("MODEL_FRACTIONS must contain exactly 3 values.")

    # checks if all the values are positive, negative values are not possible
    if np.any(fractions < 0):
        raise ValueError("MODEL_FRACTIONS cannot contain negative values.")

    # the sum of the fractions must be 1.0
    # otherwise it raises a value error, because it is possible
    if not np.isclose(fractions.sum(), 1.0):
        raise ValueError("MODEL_FRACTIONS must sum to 1.0.")
    return rng.choice(len(ZONES), size=n_particles, p=fractions)


# simulates particle transport and record arrival times at each monitoring station
# this is the main part of the model, after obtaining all the important values/parameters, the model can be run with them
def run_model(distances, u_section_zones, dispersion_trend):
    rng = np.random.default_rng(RANDOM_SEED)
    distances = np.asarray(distances, dtype=float)
    L_max = float(distances.max())
    k = exchange_matrix(RELATIVE_SUBMERGE)

    # initializes particle positions and their transport zones
    x = np.zeros(N_PARTICLES, dtype=float)
    zone = initialize_particle_zones(N_PARTICLES, MODEL_FRACTIONS, rng)

    # stores the first arrival time of each particle at each monitoring station
    arrival_times = {float(dist): [] for dist in distances}
    # makes sure that the particle only counts for the first arrival time
    reached = {float(dist): np.zeros(N_PARTICLES, dtype=bool) for dist in distances}

    # particle movement in descrete steps
    # depends on the duration of the BTC and chosen timestep
    n_steps = int(T_MAX / DT)


    # runs the simulation in time steps until all particles reach the last monitoring station or the maximum time is reached
    for step in range(n_steps):
        t = step * DT
        active = x < L_max
        if not np.any(active):
            break

        idx = np.where(active)[0]
        # Freeze the source states for movement and exchange in this time step.
        zi = zone[idx].copy()

        # determines the current section 
        section_idx = np.searchsorted(distances, x[idx], side="right")
        section_idx = np.clip(section_idx, 0, len(distances) - 1)

        # determines the transport-state velocity of each particle
        particle_velocity = u_section_zones[section_idx, zi]

        # calculates the advective displacement of each particle during one time step.
        advective_step = particle_velocity * DT

        # Reference dispersion increases with downstream distance according to the empirical FWHM-derived trend. 
        # Each transport zone receives its fixed relative share of that local reference value.
        D_ref_particle = reference_dispersion_at_distance(
            x[idx],
            dispersion_trend,
        )
        zone_scale = D_SCALE * D_ZONE_FACTORS[zi]
        D_particle = zone_scale * D_ref_particle

        # fickian drift correction for spatially variable dispersion:
        # dx = (u_z + dD_z/dx) * dt + sqrt(2 * D_z * dt) * xi.
        # the reference gradient is zero upstream of x0 and at/above the cap.
        dD_ref_dx = np.zeros_like(D_ref_particle)
        if USE_DISTANCE_DEPENDENT_DISPERSION:
            increasing = (
                (x[idx] > dispersion_trend["x0_m"])
                & (D_ref_particle < dispersion_trend["D_cap_m2_s"])
            )
            dD_ref_dx[increasing] = dispersion_trend["beta_D_m_s"]

        # apply the same scaling to the gradient as to D itself.
        dD_particle_dx = zone_scale * dD_ref_dx
        dispersion_drift_step = dD_particle_dx * DT

        # calculates the random particle displacement caused by dispersion.
        random_step = (
            np.sqrt(2.0 * D_particle * DT)
            * rng.standard_normal(len(idx))
        )
        # new position of the particle with all the processes at each timestep
        x[idx] += advective_step + dispersion_drift_step + random_step

        # makes sure that a particle cannot be moved above the injection point
        x[idx] = np.maximum(x[idx], 0.0)

        for i in range(len(ZONES)):
            # Use the frozen source states, not states changed by earlier
            # iterations of this loop. Each particle gets one exchange trial.
            # "particles_i" contains all particles that are in the current state
            particles_i = idx[zi == i]

            # when there is no particle in a certain state, it will directely continue
            if len(particles_i) == 0:
                continue

            probs = k[i] * DT
            total_prob = probs.sum()

            # if the probability is zero, the particle stays in their current transport state
            if total_prob <= 0:
                continue

            # if the probability is larger then one, it will raise en error (impossible to be larger than one)
            if total_prob > 1:
                raise RuntimeError(f"Switch probability > 1 for zone {i}. Reduce DT or exchange rates.")

            # randomly determines which particles change their transport zone.
            switching = rng.random(len(particles_i)) < total_prob
            switching_particles = particles_i[switching]

            if len(switching_particles) > 0:
                # selects the new state according to the relative transition rates
                target_probs = probs / total_prob
                new_zones = rng.choice(
                    len(ZONES),
                    size=len(switching_particles),
                    p=target_probs,
                )
                zone[switching_particles] = new_zones

        # records the first arrival time of each particle at each monitoring station.
        for dist in distances:
            dist = float(dist)
            newly_arrived = (x >= dist) & (~reached[dist])

            if np.any(newly_arrived):
                arrival_times[dist].extend([t] * int(np.sum(newly_arrived)))
                reached[dist][newly_arrived] = True

    # returns the recorded arrival times for each monitoring station.
    return {dist: np.asarray(times, dtype=float) for dist, times in arrival_times.items()}

def model_pdf_from_arrivals(arrivals):

    # the time of simulation is divided into bins
    bins = np.arange(0, T_MAX + BIN_WIDTH, BIN_WIDTH)

    # calculates the mean otf the intervals
    time_mid = (bins[:-1] + bins[1:]) / 2.0

    # counts how many particles have crossed/reached the monitoring station each time interval
    counts, _ = np.histogram(arrivals, bins=bins)

    # the particle numbers is divided by the total amount of particles and the bin width
    model_pdf = counts / (N_PARTICLES * BIN_WIDTH)

    # AUC gets calculated and noramlized to one
    area = np.sum(model_pdf * BIN_WIDTH)
    if area > 0:
        model_pdf = model_pdf / area

    return time_mid, model_pdf



# Load measurement data and estimate dispersion and section velocities.
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    master_rows = load_master_rows()
    master_rows = estimate_empirical_dispersion(master_rows)

    dispersion_trend = estimate_dispersion_distance_trend(master_rows)
    section_table, u_section_zones = calculate_section_velocities(master_rows)

    # prints the calculated dispersion, storage exchange and velocity parameters.
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

    # runs the particle tracer model and records the arrival times
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

    # comparison of the observed and simulated BTC data 
    for ax, (_, row) in zip(axes, master_rows.iterrows()):
        device = int(row["Device_number"])
        station = str(row["Monitoring_station"])
        dist = float(row["Distance_m"])

        btc_path = find_btc_file(device)
        obs_t, obs_pdf = load_measured_btc(btc_path, row["Injection_time_UTC"])

        model_t, model_pdf = model_pdf_from_arrivals(arrival_times[dist])

        ax.plot(obs_t, obs_pdf, linewidth=1.8, label="Measured BTC")
        ax.plot(model_t, model_pdf, linewidth=1.5, linestyle="--", label="Model BTC")

        ax.set_ylabel("Normalized BTC [1/s]")
        ax.set_title(
            f"{station}, {dist:.0f} m | "
            f"D_FWHM(obs)={row['D_from_FWHM_m2_s']:.3g} m2/s | "
        )
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

        # collects the model parameters and BTC characteristics for each station.
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
            "sigma_from_FWHM_s": row["sigma_from_FWHM_s"],
            "D_from_FWHM_station_m2_s": row["D_from_FWHM_m2_s"],
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
