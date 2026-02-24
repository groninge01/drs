# iRacing Pacenote Tools

This repo contains two scripts:

- `generate_pacenotes.py`: builds pace notes from a Garage61 telemetry CSV and writes `pace_notes_output.json`.
- `iracing_pacenote_engine.py`: reads `pace_notes_output.json`, watches live iRacing lap position, and speaks notes at the configured trigger points.

## Requirements

- Python 3.10+
- iRacing running (for live engine playback)

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Generate Pace Notes JSON

```bash
python3 generate_pacenotes.py data/your-reference-lap.csv
```

Output:

- `pace_notes_output.json` in the project root

## 2) Run Live Pacenote Engine

```bash
python3 iracing_pacenote_engine.py
```

Notes:

- `iracing_pacenote_engine.py` expects `pace_notes_output.json` in the same folder.
- If iRacing is not on track, the engine waits and keeps polling.

## Build Windows EXE (Engine)

Build a standalone `iracing-pacenote-engine.exe` with PyInstaller:

```powershell
cd C:\path\to\drs
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pyinstaller --onefile --name iracing-pacenote-engine iracing_pacenote_engine.py
```

Output executable:

- `dist\iracing-pacenote-engine.exe`

Run it:

```powershell
.\dist\iracing-pacenote-engine.exe
```

If PowerShell script execution is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
