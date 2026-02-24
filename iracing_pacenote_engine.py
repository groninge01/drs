import irsdk
import time
import json
import pyttsx3

# ==============================
# CONFIG
# ==============================

NOTES_FILE = "pace_notes_output.json"
UPDATE_RATE = 0.02  # 50Hz check


# ==============================
# INIT
# ==============================

ir = irsdk.IRSDK()
ir.startup()

engine = pyttsx3.init()
engine.setProperty("rate", 165)  # calm engineer tone


with open(NOTES_FILE, "r") as f:
    notes = sorted(json.load(f), key=lambda x: x["speak_at_lap_pct"])


current_note_index = 0
last_lap = -1
spoken_this_lap = set()


# ==============================
# MAIN LOOP
# ==============================

print("Pace note engine running...")

while True:

    if not ir.is_initialized:
        ir.startup()
        time.sleep(1)
        continue

    if ir["IsOnTrack"] != 1:
        time.sleep(UPDATE_RATE)
        continue

    lap = ir["Lap"]
    lap_pct = ir["LapDistPct"]

    # Detect new lap
    if lap != last_lap:
        current_note_index = 0
        spoken_this_lap.clear()
        last_lap = lap

    # Check next note
    if current_note_index < len(notes):

        note = notes[current_note_index]

        if lap_pct >= note["speak_at_lap_pct"]:
            if current_note_index not in spoken_this_lap:

                engine.say(note["note"])
                engine.runAndWait()

                spoken_this_lap.add(current_note_index)
                current_note_index += 1

    time.sleep(UPDATE_RATE)
