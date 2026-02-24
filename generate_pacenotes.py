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
    return series.rolling(window=window, center=True).mean().fillna(method="bfill").fillna(method="ffill")


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


# ==============================
# Event Detection
# ==============================

def detect_events(df):

    events = []

    brake = df["Brake"]
    throttle = df["Throttle"]
    speed = df["Speed"]
    steering = df["SteeringWheelAngle"]
    lap_pct = df["LapDistPct"]

    brake_active = False
    lift_active = False
    start_idx = None

    for i in range(1, len(df)):

        # Brake event start
        if not brake_active:
            if brake.iloc[i] > BRAKE_THRESHOLD and speed.iloc[i] > MIN_BRAKE_SPEED:
                brake_active = True
                start_idx = i

        # Brake event end
        elif brake_active:
            if brake.iloc[i] < BRAKE_THRESHOLD:
                end_idx = i
                duration = (end_idx - start_idx) * df["TimeDelta"].mean()
                if duration >= MIN_EVENT_DURATION:
                    events.append(("brake", start_idx, end_idx))
                brake_active = False

        # Lift event start
        if not lift_active:
            throttle_drop = throttle.iloc[i-1] - throttle.iloc[i]
            if throttle_drop > LIFT_DROP_THRESHOLD and speed.iloc[i] > MIN_LIFT_SPEED and brake.iloc[i] < 0.03:
                lift_active = True
                start_idx = i

        elif lift_active:
            if throttle.iloc[i] > 0.9:
                end_idx = i
                duration = (end_idx - start_idx) * df["TimeDelta"].mean()
                if duration >= MIN_EVENT_DURATION:
                    events.append(("lift", start_idx, end_idx))
                lift_active = False

    return events


# ==============================
# Main Pace Note Generator
# ==============================

def generate_notes(csv_path):

    df = pd.read_csv(csv_path)

    # Ensure TimeDelta exists
    if "TimeDelta" not in df.columns:
        df["TimeDelta"] = df["Time"].diff().fillna(0.01)

    # Smooth signals
    df["Brake"] = smooth(df["Brake"], SMOOTHING_WINDOW)
    df["Throttle"] = smooth(df["Throttle"], SMOOTHING_WINDOW)
    df["SteeringWheelAngle"] = smooth(df["SteeringWheelAngle"], SMOOTHING_WINDOW)

    track_length_m = 24358  # Nürburgring VLN default; replace if known

    events = detect_events(df)

    pace_notes = []
    zone_id = 1

    for event_type, start, end in events:

        entry_speed = df["Speed"].iloc[start]
        min_speed = df["Speed"].iloc[start:end].min()
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

    output_file = "pace_notes_output.json"
    with open(output_file, "w") as f:
        json.dump(notes, f, indent=2)

    print(f"Generated {len(notes)} pace notes.")
    print(f"Saved to {output_file}")
