#!/usr/bin/env python3
"""Speak driving cues from live iRacing telemetry using a reference Garage61 CSV."""

from __future__ import annotations

import argparse
import csv
import platform
import queue
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


try:
    import irsdk  # type: ignore
except ImportError:  # pragma: no cover
    irsdk = None

try:
    import pyttsx3
except ImportError:  # pragma: no cover
    pyttsx3 = None

try:
    import pythoncom  # type: ignore
except ImportError:  # pragma: no cover
    pythoncom = None


@dataclass
class RefSample:
    lap_pct: float
    speed_mps: float
    brake: float
    throttle: float
    gear: int


@dataclass
class BrakeZone:
    index: int
    start_pct: float
    end_pct: float
    peak_brake: float
    min_speed_mps: float
    min_gear: int
    approach_speed_mps: float


@dataclass
class ZoneAnnouncementState:
    prepare_done: bool = False
    action_done: bool = False


class SpeechWorker:
    def __init__(self, rate: int = 185, voice_contains: Optional[str] = None) -> None:
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._rate = rate
        self._voice_contains = voice_contains
        self._thread.start()

    def say(self, text: str) -> None:
        self._queue.put(text)

    def _run(self) -> None:
        if pyttsx3 is None:
            print("[WARN] pyttsx3 is not installed. Printing cues only.")
            while True:
                text = self._queue.get()
                print(f"[CUE] {text}")
            return

        com_initialized = False
        if platform.system().lower().startswith("win") and pythoncom is not None:
            pythoncom.CoInitialize()
            com_initialized = True
        engine = self._init_engine()

        try:
            while True:
                text = self._queue.get()
                print(f"[CUE] {text}")
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:  # pragma: no cover
                    print(f"[WARN] TTS playback failed ({e}); retrying once.")
                    try:
                        engine = self._init_engine()
                        engine.say(text)
                        engine.runAndWait()
                    except Exception as retry_error:
                        print(f"[WARN] TTS retry failed ({retry_error}).")
        finally:
            if com_initialized:
                pythoncom.CoUninitialize()

    def _init_engine(self) -> "pyttsx3.Engine":
        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        if self._voice_contains:
            for voice in engine.getProperty("voices"):
                name = (voice.name or "").lower()
                if self._voice_contains.lower() in name:
                    engine.setProperty("voice", voice.id)
                    break
        return engine


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def load_reference_csv(path: Path) -> List[RefSample]:
    samples: List[RefSample] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"LapDistPct", "Speed", "Brake", "Throttle", "Gear"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")

        for row in reader:
            samples.append(
                RefSample(
                    lap_pct=float(row["LapDistPct"]),
                    speed_mps=float(row["Speed"]),
                    brake=float(row["Brake"]),
                    throttle=float(row["Throttle"]),
                    gear=int(float(row["Gear"])),
                )
            )

    if len(samples) < 100:
        raise ValueError("CSV has too few rows to build cue zones")

    samples.sort(key=lambda s: s.lap_pct)
    return samples


def extract_brake_zones(
    samples: List[RefSample],
    start_threshold: float,
    end_threshold: float,
    min_zone_len: int,
) -> List[BrakeZone]:
    zones: List[BrakeZone] = []
    in_zone = False
    zone_start = 0

    for i, s in enumerate(samples):
        if not in_zone and s.brake >= start_threshold:
            in_zone = True
            zone_start = i
            continue

        if in_zone and (s.brake <= end_threshold or i == len(samples) - 1):
            zone_end = i
            if zone_end - zone_start + 1 >= min_zone_len:
                chunk = samples[zone_start : zone_end + 1]
                peak_brake = max(x.brake for x in chunk)
                min_speed_mps = min(x.speed_mps for x in chunk)
                driving_gears = [x.gear for x in chunk if x.gear > 0]
                min_gear = min(driving_gears) if driving_gears else max(chunk[0].gear, 1)
                zones.append(
                    BrakeZone(
                        index=len(zones),
                        start_pct=chunk[0].lap_pct,
                        end_pct=chunk[-1].lap_pct,
                        peak_brake=peak_brake,
                        min_speed_mps=min_speed_mps,
                        min_gear=min_gear,
                        approach_speed_mps=chunk[0].speed_mps,
                    )
                )
            in_zone = False

    return zones


def circular_delta(from_pct: float, to_pct: float) -> float:
    d = to_pct - from_pct
    return d if d >= 0 else d + 1.0


def next_zone(
    zones: List[BrakeZone], current_pct: float, lookahead_pct: float
) -> Optional[BrakeZone]:
    best: Optional[BrakeZone] = None
    best_delta = 2.0
    for zone in zones:
        d = circular_delta(current_pct, zone.start_pct)
        if d <= lookahead_pct and d < best_delta:
            best = zone
            best_delta = d
    return best


def pct_to_meters(track_length_m: Optional[float], pct_delta: float) -> Optional[float]:
    if not track_length_m:
        return None
    return pct_delta * track_length_m


def format_distance_callout(
    distance_m: Optional[float], speed_mps: float, unit: str
) -> Optional[str]:
    if distance_m is None:
        return None
    if unit == "seconds":
        if speed_mps <= 0.5:
            return None
        seconds = max(1, int(round(distance_m / speed_mps)))
        return f"in {seconds} seconds"
    meters = max(1, int(round(distance_m)))
    return f"in {meters} meters"


def build_action_cue(
    zone: BrakeZone,
    current_gear: int,
    lift_cutoff: float,
    brake_band: int,
    action_target: str,
) -> str:
    if action_target == "min-speed":
        min_kph = int(round(zone.min_speed_mps * 3.6))
        if zone.peak_brake < lift_cutoff:
            cue = f"Lift, minimum speed about {min_kph} kilometers per hour"
        else:
            cue = f"Minimum speed about {min_kph} kilometers per hour"
    else:
        if zone.peak_brake < lift_cutoff:
            cue = "Lift"
        else:
            target = int(round(zone.peak_brake * 100, 0))
            cue = f"Brake about {target} percent, plus minus {brake_band}"

    if zone.min_gear < current_gear:
        downshift_count = current_gear - zone.min_gear
        if downshift_count == 1:
            cue += f", downshift to gear {zone.min_gear}"
        else:
            cue += (
                f", downshift {downshift_count} gears to gear {zone.min_gear}"
            )
    elif zone.min_gear > current_gear:
        upshift_count = zone.min_gear - current_gear
        if upshift_count == 1:
            cue += f", upshift to gear {zone.min_gear}"
        else:
            cue += f", upshift {upshift_count} gears to gear {zone.min_gear}"
    else:
        cue += f", hold gear {current_gear}"

    return cue


def estimate_track_length_m(values: List[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) < 5:
        return None
    return statistics.median(values)


def run_live(
    csv_path: Path,
    lookahead_seconds: float,
    min_lookahead_pct: float,
    max_lookahead_pct: float,
    action_lead_seconds: float,
    lift_cutoff: float,
    brake_tolerance_band: int,
    action_target: str,
    distance_callout_unit: str,
    cue_cooldown_seconds: float,
    voice_contains: Optional[str],
) -> None:
    if irsdk is None:
        raise RuntimeError("irsdk is not installed. Run: pip install -r requirements.txt")

    samples = load_reference_csv(csv_path)
    zones = extract_brake_zones(
        samples=samples,
        start_threshold=0.08,
        end_threshold=0.03,
        min_zone_len=8,
    )
    if not zones:
        raise RuntimeError("No braking zones found in the reference CSV")

    print(f"Loaded {len(samples)} reference samples")
    print(f"Extracted {len(zones)} braking zones")

    ir = irsdk.IRSDK()
    while not ir.startup():
        print("Waiting for iRacing SDK... start iRacing and load into a car.")
        time.sleep(1.0)

    speaker = SpeechWorker(rate=185, voice_contains=voice_contains)

    last_pct = 0.0
    lap_count = 0
    stage_state: dict[tuple[int, int], ZoneAnnouncementState] = {}
    last_spoken_time = 0.0
    track_length_estimates: List[float] = []
    last_wait_log = 0.0

    print("Connected to iRacing SDK.")
    print("Listening to live telemetry. Press Ctrl+C to stop.")

    while True:
        ir.freeze_var_buffer_latest()
        if not ir.is_initialized or not ir.is_connected:
            now = time.time()
            if now - last_wait_log >= 3.0:
                print("Waiting for live telemetry... enter the car and drive out of the garage.")
                last_wait_log = now
            time.sleep(0.2)
            continue

        lap_pct = float(ir["LapDistPct"] or 0.0)
        speed_mps = float(ir["Speed"] or 0.0)
        gear = int(ir["Gear"] or 0)

        if lap_pct < last_pct - 0.5:
            lap_count += 1
            stage_state = {k: v for k, v in stage_state.items() if k[0] >= lap_count - 1}
        last_pct = lap_pct

        lap_dist = ir["LapDist"]
        if lap_dist is not None and lap_pct > 0.05:
            track_length_estimates.append(float(lap_dist) / lap_pct)
            if len(track_length_estimates) > 200:
                track_length_estimates.pop(0)

        track_len = estimate_track_length_m(track_length_estimates)
        if track_len:
            lookahead_pct = _clamp((speed_mps * lookahead_seconds) / track_len, min_lookahead_pct, max_lookahead_pct)
        else:
            lookahead_pct = 0.006

        zone = next_zone(zones, lap_pct, lookahead_pct)
        now = time.time()
        if zone is not None:
            key = (lap_count, zone.index)
            state = stage_state.setdefault(key, ZoneAnnouncementState())
            d_pct = circular_delta(lap_pct, zone.start_pct)
            d_m = pct_to_meters(track_len, d_pct)
            lead_callout = format_distance_callout(d_m, speed_mps, distance_callout_unit)

            if track_len:
                action_pct = _clamp(
                    (speed_mps * action_lead_seconds) / track_len,
                    min_lookahead_pct * 0.25,
                    max_lookahead_pct,
                )
            else:
                action_pct = min_lookahead_pct

            if (
                not state.prepare_done
                and d_pct <= lookahead_pct
                and d_pct > action_pct
                and (now - last_spoken_time) >= cue_cooldown_seconds
            ):
                preview = build_action_cue(
                    zone,
                    gear,
                    lift_cutoff,
                    brake_tolerance_band,
                    action_target,
                )
                if lead_callout is not None:
                    speaker.say(f"Corner {lead_callout}. {preview}.")
                else:
                    speaker.say(f"Corner coming up. {preview}.")
                state.prepare_done = True
                last_spoken_time = now

            if (
                not state.action_done
                and d_pct <= action_pct
                and (now - last_spoken_time) >= cue_cooldown_seconds
            ):
                speaker.say("Brake now.")
                state.action_done = True
                last_spoken_time = now

        time.sleep(0.03)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live iRacing telemetry coach with spoken lift/brake/gear cues"
    )
    p.add_argument(
        "-c",
        "--csv",
        type=Path,
        default=None,
        help="Reference lap CSV exported from Garage61",
    )
    p.add_argument("--lookahead-seconds", type=float, default=2.2)
    p.add_argument("--min-lookahead-pct", type=float, default=0.003)
    p.add_argument("--max-lookahead-pct", type=float, default=0.015)
    p.add_argument(
        "--action-lead-seconds",
        type=float,
        default=0.85,
        help="Second-stage call timing before brake point",
    )
    p.add_argument("--lift-cutoff", type=float, default=0.16)
    p.add_argument(
        "--brake-tolerance-band",
        type=int,
        default=8,
        help="Spoken brake tolerance band in percent",
    )
    p.add_argument(
        "--action-target",
        choices=["min-speed", "brake"],
        default="min-speed",
        help="Select action cue mode: minimum corner speed (default) or brake pressure",
    )
    p.add_argument(
        "--distance-callout-unit",
        choices=["meters", "seconds"],
        default="seconds",
        help="Distance lead-in unit for prepare/action calls",
    )
    p.add_argument("--cue-cooldown-seconds", type=float, default=1.0)
    p.add_argument(
        "--voice-contains",
        type=str,
        default=None,
        help="Optional substring match for TTS voice name",
    )
    args = p.parse_args()
    if args.csv is None:
        print("[WARN] No CSV supplied. Use -c/--csv with a Garage61 reference file.")
        raise SystemExit(1)
    if not args.csv.exists():
        p.error(f"CSV file not found: {args.csv}")
    if args.csv.suffix.lower() != ".csv":
        p.error(f"Expected a .csv file: {args.csv}")
    return args


def main() -> None:
    args = parse_args()
    run_live(
        csv_path=args.csv,
        lookahead_seconds=args.lookahead_seconds,
        min_lookahead_pct=args.min_lookahead_pct,
        max_lookahead_pct=args.max_lookahead_pct,
        action_lead_seconds=args.action_lead_seconds,
        lift_cutoff=args.lift_cutoff,
        brake_tolerance_band=args.brake_tolerance_band,
        action_target=args.action_target,
        distance_callout_unit=args.distance_callout_unit,
        cue_cooldown_seconds=args.cue_cooldown_seconds,
        voice_contains=args.voice_contains,
    )


if __name__ == "__main__":
    main()
