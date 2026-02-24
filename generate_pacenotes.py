import pandas as pd
import numpy as np
import json
import sys

# ==============================
# CONFIGURATION (Deterministic)
# ==============================

BRAKE_THRESHOLD = 0.08
LIFT_DROP_THRESHOLD = 0.40
MIN_BRAKE_SPEED = 60          # km/h
MIN_LIFT_SPEED = 100          # km/h
MIN_EVENT_DURATION = 0.30     # seconds
MERGE_DISTANCE_PCT = 0.002    # merge events close in lap pct
SMOOTHING_WINDOW = 5          # samples

WORDS_PER_SECOND = 2.6
WORD_TIME = 1 / WORDS_PER_SECOND
SPEECH_BUFFER = 0.40          # seconds


# ==============================
# Helper Functions
# ==============================

def smooth(series, window=5):
    return series.rolling(window=window, center=True).mean().bfill().ffill()


def classify_corner(min_speed):
    if min_speed < 70:
        return 1
    elif min_speed < 90:
        return 2
    elif min_speed < 110:
        return 3
    elif min_speed < 130:
        return 4
    elif min_speed < 160:
        return 5
    else:
        return 6


def classify_brake_prefix(peak_brake, throttle_drop):
    if peak_brake > 0.5:
        return "Big brake"
    elif peak_brake > 0.2:
        return "Brake"
    elif throttle_drop > 0.4:
        return "Lift"
    else:
        return ""


def direction_from_steering(steering_peak):
    return "Right" if steering_peak > 0 else "Left"


def calculate_speech_duration(note):
    word_count = len(note.split())
    return (word_count * WORD_TIME) + SPEECH_BUFFER


def normalize_speed(speed_series):
    raw = pd.to_numeric(speed_series, errors="coerce")
    max_speed = float(raw.max())

    # Garage61 exports can be m/s; convert to km/h for human-facing values and thresholds.
    if max_speed <= 120:
        speed_mps = raw
        speed_kmh = raw * 3.6
    else:
        speed_kmh = raw
        speed_mps = raw / 3.6

    return speed_kmh, speed_mps


def estimate_time_delta(df, track_length_m, fallback=0.01):
    lap_pct = pd.to_numeric(df["LapDistPct"], errors="coerce")
    speed_mps = pd.to_numeric(df["SpeedMps"], errors="coerce").replace(0, np.nan)

    # Handle lap wrap so diff is always forward progress in [0,1).
    lap_diff = lap_pct.diff()
    lap_diff = (lap_diff + 1.0) % 1.0

    dist_m = lap_diff * track_length_m
    dt = dist_m / speed_mps
    dt = dt.where((dt > 0) & np.isfinite(dt))
    if dt.notna().any():
        fill_value = float(dt.dropna().median())
        return dt.fillna(fill_value).clip(lower=1e-4)

    return pd.Series(fallback, index=df.index)


# ==============================
# Event Detection
# ==============================

def detect_events(df):

    events = []

    brake = df["Brake"]
    throttle = df["Throttle"]
    speed = df["SpeedKmh"]
    lap_pct = df["LapDistPct"]

    brake_active = False
    lift_active = False
    brake_start_idx = None
    lift_start_idx = None

    for i in range(1, len(df)):

        # Brake event start
        if not brake_active:
            if brake.iloc[i] > BRAKE_THRESHOLD and speed.iloc[i] > MIN_BRAKE_SPEED:
                brake_active = True
                brake_start_idx = i

        # Brake event end
        elif brake_active:
            if brake.iloc[i] < BRAKE_THRESHOLD:
                end_idx = i
                duration = float(df["TimeDelta"].iloc[brake_start_idx:end_idx + 1].sum())
                if duration >= MIN_EVENT_DURATION:
                    events.append(("brake", brake_start_idx, end_idx))
                brake_active = False
                brake_start_idx = None

        # Lift event start
        if not lift_active:
            throttle_drop = throttle.iloc[i-1] - throttle.iloc[i]
            if throttle_drop > LIFT_DROP_THRESHOLD and speed.iloc[i] > MIN_LIFT_SPEED and brake.iloc[i] < 0.03:
                lift_active = True
                lift_start_idx = i

        elif lift_active:
            # End lift when throttle is back on or braking starts.
            if throttle.iloc[i] > 0.9 or brake.iloc[i] > BRAKE_THRESHOLD:
                end_idx = i
                duration = float(df["TimeDelta"].iloc[lift_start_idx:end_idx + 1].sum())
                if duration >= MIN_EVENT_DURATION:
                    events.append(("lift", lift_start_idx, end_idx))
                lift_active = False
                lift_start_idx = None

    # Close open events at end of lap.
    last_idx = len(df) - 1
    if brake_active and brake_start_idx is not None:
        duration = float(df["TimeDelta"].iloc[brake_start_idx:last_idx + 1].sum())
        if duration >= MIN_EVENT_DURATION:
            events.append(("brake", brake_start_idx, last_idx))

    if lift_active and lift_start_idx is not None:
        duration = float(df["TimeDelta"].iloc[lift_start_idx:last_idx + 1].sum())
        if duration >= MIN_EVENT_DURATION:
            events.append(("lift", lift_start_idx, last_idx))

    if not events:
        return events

    # Merge events that are effectively the same corner approach.
    events = sorted(events, key=lambda e: e[1])
    merged = []
    cur_type, cur_start, cur_end = events[0]

    for next_type, next_start, next_end in events[1:]:
        gap_pct = float(lap_pct.iloc[next_start] - lap_pct.iloc[cur_start])
        if gap_pct <= MERGE_DISTANCE_PCT:
            cur_start = min(cur_start, next_start)
            cur_end = max(cur_end, next_end)
            if cur_type != "brake" and next_type == "brake":
                cur_type = "brake"
        else:
            merged.append((cur_type, cur_start, cur_end))
            cur_type, cur_start, cur_end = next_type, next_start, next_end

    merged.append((cur_type, cur_start, cur_end))

    return merged


# ==============================
# Main Pace Note Generator
# ==============================

def generate_notes(csv_path):

    df = pd.read_csv(csv_path)
    track_length_m = 24358  # Nürburgring VLN default; replace if known
    df["SpeedKmh"], df["SpeedMps"] = normalize_speed(df["Speed"])

    # Garage61 schema has no absolute time; estimate dt from distance delta and speed.
    df["TimeDelta"] = estimate_time_delta(df, track_length_m)

    # Smooth signals
    df["Brake"] = smooth(df["Brake"], SMOOTHING_WINDOW)
    df["Throttle"] = smooth(df["Throttle"], SMOOTHING_WINDOW)
    df["SteeringWheelAngle"] = smooth(df["SteeringWheelAngle"], SMOOTHING_WINDOW)

    events = detect_events(df)

    pace_notes = []
    zone_id = 1

    for event_type, start, end in events:

        entry_speed = df["SpeedKmh"].iloc[start]
        min_speed = df["SpeedKmh"].iloc[start:end].min()
        peak_brake = df["Brake"].iloc[start:end].max()
        steering_peak = df["SteeringWheelAngle"].iloc[start:end].mean()
        lap_trigger_pct = df["LapDistPct"].iloc[start]

        throttle_drop = df["Throttle"].iloc[start-1] - df["Throttle"].iloc[start]

        grade = classify_corner(min_speed)
        prefix = classify_brake_prefix(peak_brake, throttle_drop)
        direction = direction_from_steering(steering_peak)

        if prefix:
            note = f"{prefix}. {direction} {grade}."
        else:
            note = f"{direction} {grade}."

        duration = calculate_speech_duration(note)

        speed_mps = entry_speed / 3.6
        lead_distance = speed_mps * duration
        lead_pct = lead_distance / track_length_m

        speak_pct = lap_trigger_pct - lead_pct
        if speak_pct < 0:
            speak_pct += 1.0

        pace_notes.append({
            "zone_id": zone_id,
            "note": note,
            "entry_speed_kmh": round(float(entry_speed), 1),
            "min_speed_kmh": round(float(min_speed), 1),
            "spoken_duration_sec": round(duration, 2),
            "lead_distance_m": round(float(lead_distance), 1),
            "speak_at_lap_pct": round(float(speak_pct), 6),
            "action_trigger_pct": round(float(lap_trigger_pct), 6)
        })

        zone_id += 1

    return pace_notes


# ==============================
# CLI
# ==============================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python generate_pacenotes.py yourfile.csv")
        sys.exit(1)

    csv_path = sys.argv[1]

    notes = generate_notes(csv_path)

    output_file = "data/pace_notes_output.json"
    with open(output_file, "w") as f:
        json.dump(notes, f, indent=2)

    print(f"Generated {len(notes)} pace notes.")
    print(f"Saved to {output_file}")
