# iRacing Telemetry Voice Coach

This app reads live iRacing telemetry, compares your current lap position to a reference Garage61 CSV, and speaks upcoming corner cues:

- stage 1: prepare call
- stage 2: action call with minimum corner speed (km/h) + gear guidance
- optional brake pressure mode with tolerance band

## Files

- `telemetry_coach.py`: main app
- `requirements.txt`: Python dependencies
- `data/...csv`: your reference lap

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 telemetry_coach.py -c "data/your-reference-lap.csv"
```

Optional flags:

```bash
python3 telemetry_coach.py \
  --lookahead-seconds 2.4 \
  --action-lead-seconds 0.85 \
  --brake-tolerance-band 8 \
  --lift-cutoff 0.18 \
  --voice-contains english \
  -c "data/your-reference-lap.csv"
```

## How It Works

1. Loads the Garage61 CSV and extracts braking zones.
2. Connects to iRacing via `irsdk`.
3. Tracks live `LapDistPct`, speed, and gear.
4. Looks ahead to the next zone and speaks two-stage cues before turn-in.

Example cue:

- `Prepare for corner in 190 meters.`
- `In 72 meters. Minimum corner speed about 121 kilometers per hour, downshift to gear 3.`

## Notes

- iRacing must be running and on track for live telemetry.
- If `pyttsx3` is unavailable, cues are printed to terminal instead of spoken.
- The app assumes your reference lap and current session are the same layout/car style.

## Tuning

- `--lookahead-seconds`: higher means earlier warnings.
- `--action-lead-seconds`: timing of the second-stage action call.
- `--action-target`: `min-speed` (default) or `brake`.
- `--lift-cutoff`: brake peak below this value becomes a lift call.
- `--brake-tolerance-band`: spoken +/- range around brake target.
- `--min-lookahead-pct` / `--max-lookahead-pct`: clamp cue lead distance.
- `--cue-cooldown-seconds`: prevents repeated rapid callouts.

Brake-pressure action cue mode:

```bash
python3 telemetry_coach.py -c "data/your-reference-lap.csv" --action-target brake
```

## Build Windows Executable

Install minimal Python from PowerShell:

```powershell
winget install -e --id Python.Python.3.12
py -V
```

Build `telemetry-coach.exe`:

```powershell
cd C:\path\to\drs
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller
pyinstaller --onefile --name telemetry-coach telemetry_coach.py
```

If install fails with `No matching distribution found for irsdk`, update deps and retry:

```powershell
python -m pip install --upgrade pip
pip uninstall -y irsdk
pip install -r requirements.txt pyinstaller
```

Executable output:

- `dist\telemetry-coach.exe`

Run example:

```powershell
.\dist\telemetry-coach.exe -c "data\your-reference-lap.csv"
```

If script activation is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Next Improvements

- map zones to named corners
- separate cues into two-stage calls ("brake soon" then "brake now")
- calibrate brake/gear targets by pace delta to reference
