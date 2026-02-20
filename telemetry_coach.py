#!/usr/bin/env python3
"""CSV-led live race engineer for iRacing."""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import platform
import queue
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
    lat: float
    lon: float
    steering_angle: float
    brake: float
    throttle: float
    gear: int


@dataclass
class EngineerEvent:
    index: int
    start_pct: float
    apex_pct: float
    end_pct: float
    action_type: str  # brake | lift
    target_min_speed_kph: Optional[int]
    target_throttle_pct: Optional[int]
    target_gear: int
    peak_brake_pct: int


@dataclass
class EventAnnouncementState:
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

        try:
            while True:
                text = self._queue.get()
                print(f"[CUE] {text}")
                self._speak_with_retry(text)
        finally:
            if com_initialized:
                pythoncom.CoUninitialize()

    def _speak_with_retry(self, text: str) -> None:
        try:
            self._speak_once(text)
        except Exception as e:  # pragma: no cover
            print(f"[WARN] TTS playback failed ({e}); retrying once.")
            try:
                self._speak_once(text)
            except Exception as retry_error:  # pragma: no cover
                print(f"[WARN] TTS retry failed ({retry_error}).")

    def _speak_once(self, text: str) -> None:
        engine = self._init_engine()
        engine.say(text)
        engine.runAndWait()
        try:
            engine.stop()
        except Exception:
            pass

    def _init_engine(self) -> "pyttsx3.Engine":
        if platform.system().lower().startswith("win"):
            engine = pyttsx3.init(driverName="sapi5")
        else:
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


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = p2 - p1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))


def load_reference_csv(path: Path) -> List[RefSample]:
    samples: List[RefSample] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "LapDistPct",
            "Speed",
            "Lat",
            "Lon",
            "SteeringWheelAngle",
            "Brake",
            "Throttle",
            "Gear",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")

        for row in reader:
            samples.append(
                RefSample(
                    lap_pct=float(row["LapDistPct"]),
                    speed_mps=float(row["Speed"]),
                    lat=float(row["Lat"]),
                    lon=float(row["Lon"]),
                    steering_angle=float(row["SteeringWheelAngle"]),
                    brake=float(row["Brake"]),
                    throttle=float(row["Throttle"]),
                    gear=int(float(row["Gear"])),
                )
            )

    if len(samples) < 100:
        raise ValueError("CSV has too few rows to build events")

    samples.sort(key=lambda s: s.lap_pct)
    return samples


def build_cumulative_distance(samples: List[RefSample]) -> List[float]:
    cum = [0.0] * len(samples)
    for i in range(1, len(samples)):
        cum[i] = cum[i - 1] + _haversine_m(
            samples[i - 1].lat,
            samples[i - 1].lon,
            samples[i].lat,
            samples[i].lon,
        )
    return cum


def circular_delta(from_pct: float, to_pct: float) -> float:
    d = to_pct - from_pct
    return d if d >= 0 else d + 1.0


def _forward_indices(start_idx: int, end_idx: int, n: int) -> List[int]:
    if start_idx <= end_idx:
        return list(range(start_idx, end_idx + 1))
    return list(range(start_idx, n)) + list(range(0, end_idx + 1))


def _lookahead_indices(cum_dist: List[float], distance_m: float) -> List[int]:
    n = len(cum_dist)
    if n == 0:
        return []
    total_len = cum_dist[-1]
    if total_len <= 0.0:
        return [0] * n

    extended = cum_dist + [x + total_len for x in cum_dist]
    out = [0] * n
    for i in range(n):
        target = cum_dist[i] + distance_m
        j = bisect.bisect_left(extended, target, lo=i, hi=i + n)
        out[i] = j % n
    return out


def _distance_between(cum_dist: List[float], i: int, j: int) -> float:
    total_len = cum_dist[-1]
    if total_len <= 0.0:
        return 0.0
    d = abs(cum_dist[j] - cum_dist[i])
    return min(d, total_len - d)


def extract_engineer_events(samples: List[RefSample]) -> List[EngineerEvent]:
    n = len(samples)
    cum = build_cumulative_distance(samples)
    if n < 50 or cum[-1] <= 100.0:
        return []

    ahead_120 = _lookahead_indices(cum, 120.0)
    ahead_250 = _lookahead_indices(cum, 250.0)

    brake_onset = 0.08
    brake_release = 0.04
    lift_drop = 0.12
    lift_throttle = 0.90

    candidates: List[tuple[int, str, float]] = []  # idx, type, strength

    for i in range(1, n):
        prev = samples[i - 1]
        cur = samples[i]

        if prev.brake < brake_onset <= cur.brake:
            strength = cur.brake - prev.brake
            candidates.append((i, "brake", strength + 1.0))
            continue

        drop = prev.throttle - cur.throttle
        if drop >= lift_drop and cur.throttle <= lift_throttle and cur.brake < 0.03:
            candidates.append((i, "lift", drop))

    if not candidates:
        return []

    # Keep close events when action changes; dedupe very similar same-action triggers.
    candidates.sort(key=lambda x: x[0])
    filtered: List[tuple[int, str, float]] = []
    min_same_action_sep_m = 35.0
    for idx, action, strength in candidates:
        if not filtered:
            filtered.append((idx, action, strength))
            continue

        prev_idx, prev_action, prev_strength = filtered[-1]
        gap_m = _distance_between(cum, prev_idx, idx)
        if gap_m < min_same_action_sep_m and action == prev_action:
            if strength > prev_strength:
                filtered[-1] = (idx, action, strength)
            continue
        if gap_m < 10.0:
            continue
        filtered.append((idx, action, strength))

    indexed_events: List[tuple[int, int, EngineerEvent]] = []
    for idx, action, _strength in filtered:
        end_idx = ahead_120[idx]
        eval_idx = ahead_250[idx]
        window = _forward_indices(idx, eval_idx, n)
        if len(window) < 3:
            continue

        min_speed_idx = min(window, key=lambda j: samples[j].speed_mps)
        min_speed_kph = int(round(samples[min_speed_idx].speed_mps * 3.6))
        min_gear = min((samples[j].gear for j in window if samples[j].gear > 0), default=max(samples[idx].gear, 1))
        peak_brake = max(samples[j].brake for j in window)
        min_throttle = min(samples[j].throttle for j in window)
        approach_speed_kph = int(round(samples[idx].speed_mps * 3.6))
        speed_drop_kph = max(0, approach_speed_kph - min_speed_kph)

        # User rule: lift = throttle goes down with no/very little braking (<10%).
        if peak_brake >= 0.10:
            final_action = "brake"
            target_speed = min_speed_kph
            target_throttle = None
        else:
            # Must show a real lift signature, not tiny noise.
            if min_throttle > 95 and speed_drop_kph < 5:
                continue
            final_action = "lift"
            target_speed = None
            target_throttle = int(round(min_throttle * 100.0))

        # Skip trivial lift calls like "lift to 0%" unless they have major speed impact.
        if final_action == "lift" and (target_throttle or 0) <= 8 and speed_drop_kph < 25:
            continue

        indexed_events.append(
            (
                idx,
                min_speed_idx,
                EngineerEvent(
                    index=0,
                    start_pct=samples[idx].lap_pct,
                    apex_pct=samples[min_speed_idx].lap_pct,
                    end_pct=samples[end_idx].lap_pct,
                    action_type=final_action,
                    target_min_speed_kph=target_speed,
                    target_throttle_pct=target_throttle,
                    target_gear=min_gear,
                    peak_brake_pct=int(round(peak_brake * 100.0)),
                ),
            )
        )

    if not indexed_events:
        return []

    # Merge very close events; prefer brake when clustered.
    indexed_events.sort(key=lambda x: x[0])
    merged: List[tuple[int, int, EngineerEvent]] = []
    min_event_sep_m = 45.0
    for idx, apex_idx, ev in indexed_events:
        if not merged:
            merged.append((idx, apex_idx, ev))
            continue
        prev_idx, prev_apex_idx, prev_ev = merged[-1]
        gap_m = cum[idx] - cum[prev_idx]
        apex_gap_m = _distance_between(cum, apex_idx, prev_apex_idx)

        # If apexes are very close and starts are close, treat as same corner.
        if apex_gap_m < 70.0 and gap_m < 120.0:
            if ev.action_type == "brake" and prev_ev.action_type != "brake":
                merged[-1] = (idx, apex_idx, ev)
                continue
            if ev.action_type == prev_ev.action_type == "brake":
                prev_speed = prev_ev.target_min_speed_kph or 999
                cur_speed = ev.target_min_speed_kph or 999
                prev_gear = prev_ev.target_gear
                cur_gear = ev.target_gear
                if abs(cur_speed - prev_speed) <= 12 and abs(cur_gear - prev_gear) <= 1:
                    # Very similar target profile: keep only one.
                    if cur_speed < prev_speed:
                        merged[-1] = (idx, apex_idx, ev)
                    continue
            if ev.action_type == prev_ev.action_type == "lift":
                prev_throttle = prev_ev.target_throttle_pct or 100
                cur_throttle = ev.target_throttle_pct or 100
                if abs(cur_throttle - prev_throttle) <= 10:
                    if cur_throttle < prev_throttle:
                        merged[-1] = (idx, apex_idx, ev)
                    continue
            if ev.action_type != prev_ev.action_type:
                if gap_m >= 90.0:
                    merged.append((idx, apex_idx, ev))
                    continue
                if ev.action_type == "brake":
                    merged[-1] = (idx, apex_idx, ev)
                continue

        if gap_m < min_event_sep_m:
            if ev.action_type == "brake" and prev_ev.action_type != "brake":
                merged[-1] = (idx, apex_idx, ev)
                continue
            if ev.action_type == prev_ev.action_type == "brake":
                prev_speed = prev_ev.target_min_speed_kph or 999
                cur_speed = ev.target_min_speed_kph or 999
                if cur_speed < prev_speed:
                    merged[-1] = (idx, apex_idx, ev)
                continue
            if ev.action_type == prev_ev.action_type == "lift":
                prev_throttle = prev_ev.target_throttle_pct or 100
                cur_throttle = ev.target_throttle_pct or 100
                if cur_throttle < prev_throttle:
                    merged[-1] = (idx, apex_idx, ev)
                continue
            continue
        merged.append((idx, apex_idx, ev))

    # Long-gap duplicate cleanup for brake events with near-identical targets.
    compact: List[tuple[int, int, EngineerEvent]] = []
    for idx, apex_idx, ev in merged:
        if not compact:
            compact.append((idx, apex_idx, ev))
            continue
        prev_idx, prev_apex_idx, prev_ev = compact[-1]
        gap_m = cum[idx] - cum[prev_idx]
        if ev.action_type == prev_ev.action_type == "brake" and gap_m < 260.0:
            prev_speed = prev_ev.target_min_speed_kph or 999
            cur_speed = ev.target_min_speed_kph or 999
            if abs(cur_speed - prev_speed) <= 8 and abs(ev.target_gear - prev_ev.target_gear) <= 1:
                if cur_speed < prev_speed:
                    compact[-1] = (idx, apex_idx, ev)
                continue
        compact.append((idx, apex_idx, ev))

    # Final dedupe by start position.
    events = [ev for _, _, ev in compact]
    events.sort(key=lambda e: e.start_pct)
    deduped: List[EngineerEvent] = []
    min_start_sep_pct = 0.0012
    for ev in events:
        if not deduped:
            deduped.append(ev)
            continue
        prev = deduped[-1]
        if circular_delta(prev.start_pct, ev.start_pct) < min_start_sep_pct:
            if ev.action_type == "brake" and prev.action_type != "brake":
                deduped[-1] = ev
            continue
        deduped.append(ev)

    for i, ev in enumerate(deduped):
        ev.index = i
    return deduped


def next_event(events: List[EngineerEvent], current_pct: float, lookahead_pct: float) -> Optional[EngineerEvent]:
    best: Optional[EngineerEvent] = None
    best_delta = 2.0
    for ev in events:
        d = circular_delta(current_pct, ev.start_pct)
        if d <= lookahead_pct and d < best_delta:
            best = ev
            best_delta = d
    return best


def estimate_track_length_m(values: List[float]) -> Optional[float]:
    if len(values) < 5:
        return None
    return sorted(values)[len(values) // 2]


def gear_instruction(current_gear: int, target_gear: int) -> str:
    if target_gear < current_gear:
        n = current_gear - target_gear
        if n == 1:
            return f"downshift to {target_gear}"
        return f"downshift {n} gears to {target_gear}"
    if target_gear > current_gear:
        n = target_gear - current_gear
        if n == 1:
            return f"upshift to {target_gear}"
        return f"upshift {n} gears to {target_gear}"
    return f"hold gear {target_gear}"


def build_prepare_cue(event: EngineerEvent, current_gear: int, seconds_to_event: int) -> str:
    shift = gear_instruction(current_gear, event.target_gear)
    if event.action_type == "brake":
        target_speed = event.target_min_speed_kph if event.target_min_speed_kph is not None else 0
        return f"In {seconds_to_event} seconds: brake to about {target_speed} kilometers per hour, {shift}."

    target_throttle = event.target_throttle_pct if event.target_throttle_pct is not None else 0
    return f"In {seconds_to_event} seconds: lift to about {target_throttle} percent, {shift}."


def run_live(
    csv_path: Path,
    lookahead_seconds: float,
    min_lookahead_meters: float,
    max_lookahead_meters: float,
    action_lead_seconds: float,
    voice_contains: Optional[str],
) -> None:
    if irsdk is None:
        raise RuntimeError("irsdk is not installed. Run: pip install -r requirements.txt")

    samples = load_reference_csv(csv_path)
    events = extract_engineer_events(samples)
    if not events:
        raise RuntimeError("No actionable events found in the reference CSV")

    n_brake = sum(1 for e in events if e.action_type == "brake")
    n_lift = sum(1 for e in events if e.action_type == "lift")

    print(f"Loaded {len(samples)} reference samples")
    print(f"Extracted {len(events)} engineer events ({n_brake} brake, {n_lift} lift)")

    ir = irsdk.IRSDK()
    while not ir.startup():
        print("Waiting for iRacing SDK... start iRacing and load into a car.")
        time.sleep(1.0)

    speaker = SpeechWorker(rate=185, voice_contains=voice_contains)

    last_pct = 0.0
    lap_count = 0
    state_map: dict[tuple[int, int], EventAnnouncementState] = {}
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
        on_track = bool(ir["IsOnTrackCar"]) if ir["IsOnTrackCar"] is not None else True

        if not on_track or speed_mps < 2.5:
            now = time.time()
            if now - last_wait_log >= 3.0:
                print("Waiting to drive... cues start once you're on track and moving.")
                last_wait_log = now
            last_pct = lap_pct
            time.sleep(0.1)
            continue

        if lap_pct < last_pct - 0.5:
            lap_count += 1
            state_map = {k: v for k, v in state_map.items() if k[0] >= lap_count - 1}
        last_pct = lap_pct

        lap_dist = ir["LapDist"]
        if lap_dist is not None and lap_pct > 0.05:
            track_length_estimates.append(float(lap_dist) / lap_pct)
            if len(track_length_estimates) > 200:
                track_length_estimates.pop(0)

        track_len = estimate_track_length_m(track_length_estimates)
        if track_len is None:
            time.sleep(0.03)
            continue

        adaptive_lookahead_m = _clamp(
            speed_mps * lookahead_seconds,
            min_lookahead_meters,
            max_lookahead_meters,
        )
        lookahead_pct = _clamp(adaptive_lookahead_m / track_len, 0.001, 0.04)
        event = next_event(events, lap_pct, lookahead_pct)
        if event is None:
            time.sleep(0.03)
            continue

        key = (lap_count, event.index)
        state = state_map.setdefault(key, EventAnnouncementState())

        d_pct = circular_delta(lap_pct, event.start_pct)
        d_m = d_pct * track_len
        seconds_to_event = max(1, int(round(d_m / max(0.5, speed_mps))))

        if not state.prepare_done and d_pct <= lookahead_pct:
            speaker.say(build_prepare_cue(event, gear, seconds_to_event))
            state.prepare_done = True

        action_pct = _clamp((speed_mps * action_lead_seconds) / track_len, 0.0005, 0.01)
        if not state.action_done and d_pct <= action_pct:
            speaker.say("Now.")
            state.action_done = True

        time.sleep(0.03)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CSV-led live iRacing digital race engineer")
    p.add_argument("-c", "--csv", type=Path, default=None, help="Reference lap CSV exported from Garage61")
    p.add_argument(
        "--lookahead-seconds",
        type=float,
        default=5.0,
        help="Dynamic lookahead based on speed (distance = speed * seconds)",
    )
    p.add_argument(
        "--min-lookahead-meters",
        type=float,
        default=110.0,
        help="Minimum dynamic lookahead distance in meters (low-speed floor)",
    )
    p.add_argument(
        "--max-lookahead-meters",
        type=float,
        default=420.0,
        help="Maximum dynamic lookahead distance in meters (high-speed cap)",
    )
    p.add_argument(
        "--action-lead-seconds",
        type=float,
        default=0.40,
        help="How early to speak 'Now.' before the reference action point",
    )
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
    if args.lookahead_seconds <= 0.3:
        p.error("--lookahead-seconds must be > 0.3")
    if args.min_lookahead_meters <= 30.0:
        p.error("--min-lookahead-meters must be > 30")
    if args.max_lookahead_meters <= args.min_lookahead_meters:
        p.error("--max-lookahead-meters must be greater than --min-lookahead-meters")
    return args


def main() -> None:
    args = parse_args()
    run_live(
        csv_path=args.csv,
        lookahead_seconds=args.lookahead_seconds,
        min_lookahead_meters=args.min_lookahead_meters,
        max_lookahead_meters=args.max_lookahead_meters,
        action_lead_seconds=args.action_lead_seconds,
        voice_contains=args.voice_contains,
    )


if __name__ == "__main__":
    main()
