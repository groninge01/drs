#!/usr/bin/env python3
"""Speak driving cues from live iRacing telemetry using a reference Garage61 CSV."""

from __future__ import annotations

import argparse
import csv
import math
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
    lat: float
    lon: float
    brake: float
    throttle: float
    gear: int


@dataclass
class BrakeZone:
    index: int
    start_pct: float
    apex_pct: float
    end_pct: float
    action_type: str
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


def load_reference_csv(path: Path) -> List[RefSample]:
    samples: List[RefSample] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"LapDistPct", "Speed", "Lat", "Lon", "Brake", "Throttle", "Gear"}
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
                    brake=float(row["Brake"]),
                    throttle=float(row["Throttle"]),
                    gear=int(float(row["Gear"])),
                )
            )

    if len(samples) < 100:
        raise ValueError("CSV has too few rows to build cue zones")

    samples.sort(key=lambda s: s.lap_pct)
    return samples


def _is_local_minimum(values: List[float], i: int) -> bool:
    return values[i] <= values[i - 1] and values[i] <= values[i + 1]


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = p2 - p1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))


def _bearing_rad(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlambda)
    return math.atan2(y, x)


def _angle_diff_rad(a: float, b: float) -> float:
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


def _smooth(values: List[float], half_window: int) -> List[float]:
    if half_window <= 0 or len(values) < 3:
        return values[:]
    out: List[float] = [0.0] * len(values)
    n = len(values)
    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        out[i] = statistics.mean(values[lo:hi])
    return out


def _lookahead_indices(cum_dist: List[float], distance_m: float) -> List[int]:
    n = len(cum_dist)
    out = [n - 1] * n
    j = 0
    for i in range(n):
        if j < i:
            j = i
        target = cum_dist[i] + distance_m
        while j < n - 1 and cum_dist[j] < target:
            j += 1
        out[i] = j
    return out


def _lookback_indices(cum_dist: List[float], distance_m: float) -> List[int]:
    n = len(cum_dist)
    out = [0] * n
    j = n - 1
    for i in range(n - 1, -1, -1):
        if j > i:
            j = i
        target = cum_dist[i] - distance_m
        while j > 0 and cum_dist[j] > target:
            j -= 1
        out[i] = j
    return out


def _has_brake_reset(samples: List[RefSample], start_idx: int, end_idx: int, release_threshold: float = 0.05) -> bool:
    if end_idx - start_idx < 4:
        return False
    valley = min(samples[j].brake for j in range(start_idx + 1, end_idx))
    return valley <= release_threshold


def extract_corners(samples: List[RefSample]) -> List[BrakeZone]:
    n = len(samples)
    if n < 200:
        return []

    seg_dist: List[float] = [0.0] * n
    cum_dist: List[float] = [0.0] * n
    heading: List[float] = [0.0] * n
    for i in range(1, n):
        d = _haversine_m(samples[i - 1].lat, samples[i - 1].lon, samples[i].lat, samples[i].lon)
        seg_dist[i] = max(0.05, d)
        cum_dist[i] = cum_dist[i - 1] + seg_dist[i]
        heading[i] = _bearing_rad(samples[i - 1].lat, samples[i - 1].lon, samples[i].lat, samples[i].lon)

    look_ahead_120 = _lookahead_indices(cum_dist, 120.0)
    look_ahead_250 = _lookahead_indices(cum_dist, 250.0)
    look_ahead_500 = _lookahead_indices(cum_dist, 500.0)
    look_back_120 = _lookback_indices(cum_dist, 120.0)
    look_back_250 = _lookback_indices(cum_dist, 250.0)

    turn_score_deg = [0.0] * n
    for i in range(1, n - 1):
        j250 = look_ahead_250[i]
        j500 = look_ahead_500[i]
        turn250 = abs(math.degrees(_angle_diff_rad(heading[j250], heading[i])))
        turn500 = abs(math.degrees(_angle_diff_rad(heading[j500], heading[i])))
        # 250 m is responsive for close corners; 500 m catches long-radius bends early.
        turn_score_deg[i] = max(turn250, turn500 * 0.72)

    turn_score_deg = _smooth(turn_score_deg, half_window=3)
    nonzero_scores = sorted(x for x in turn_score_deg if x > 0.0)
    if not nonzero_scores:
        return []

    min_peak_score = max(8.0, nonzero_scores[int(len(nonzero_scores) * 0.70)])
    speed = [s.speed_mps for s in samples]

    raw_candidates: List[tuple[int, float]] = []
    for i in range(2, n - 2):
        if turn_score_deg[i] < min_peak_score:
            continue
        if turn_score_deg[i] >= turn_score_deg[i - 1] and turn_score_deg[i] > turn_score_deg[i + 1]:
            raw_candidates.append((i, turn_score_deg[i]))

    if not raw_candidates:
        return []

    # Keep strongest peaks first, then enforce spatial spacing to avoid duplicates.
    raw_candidates.sort(key=lambda item: item[1], reverse=True)
    selected_centers: List[int] = []
    min_center_sep_m = 75.0
    for center_idx, _score in raw_candidates:
        if any(abs(cum_dist[center_idx] - cum_dist[other]) < min_center_sep_m for other in selected_centers):
            continue
        selected_centers.append(center_idx)
    selected_centers.sort()

    corners: List[BrakeZone] = []
    used_apex_dist: List[float] = []
    min_action_sep_m = 60.0
    min_apex_sep_m = 55.0

    throttle_drop_threshold = 0.10
    brake_start_threshold = 0.06
    last_kept_action_idx: Optional[int] = None
    last_kept_apex_idx: Optional[int] = None
    last_kept_action_type: Optional[str] = None

    for center_idx in selected_centers:
        apex_lo = look_back_120[center_idx]
        apex_hi = look_ahead_120[center_idx]
        if apex_hi <= apex_lo:
            continue
        apex_idx = min(range(apex_lo, apex_hi + 1), key=lambda j: speed[j])
        if any(abs(cum_dist[apex_idx] - d) < min_apex_sep_m for d in used_apex_dist):
            continue

        action_search_start = max(1, look_back_250[apex_idx])
        action_idx = max(1, look_back_120[apex_idx])
        action_type = "flat"

        brake_idx: Optional[int] = None
        for j in range(apex_idx, action_search_start, -1):
            if samples[j - 1].brake < brake_start_threshold <= samples[j].brake:
                brake_idx = j
                break
        if brake_idx is None:
            peak_brake_pre = max(samples[j].brake for j in range(action_search_start, apex_idx + 1))
            if peak_brake_pre >= 0.12:
                last_brake_idx = max(
                    (j for j in range(action_search_start, apex_idx + 1) if samples[j].brake >= 0.12),
                    default=None,
                )
                if last_brake_idx is not None:
                    j = last_brake_idx
                    while j > action_search_start and samples[j - 1].brake >= brake_start_threshold:
                        j -= 1
                    brake_idx = j

        lift_idx: Optional[int] = None
        for j in range(apex_idx, action_search_start, -1):
            throttle_drop = samples[j - 1].throttle - samples[j].throttle
            if throttle_drop >= throttle_drop_threshold and samples[j].throttle <= 0.98:
                lift_idx = j
                break

        if brake_idx is not None:
            action_idx = brake_idx
            action_type = "brake"
        elif lift_idx is not None:
            action_idx = lift_idx
            action_type = "lift"

        if last_kept_action_idx is not None:
            action_gap_m = abs(cum_dist[action_idx] - cum_dist[last_kept_action_idx])
            apex_gap_m = abs(cum_dist[apex_idx] - cum_dist[last_kept_apex_idx or last_kept_action_idx])
            is_close = action_gap_m < min_action_sep_m
            if is_close:
                allow_close_brake_pair = (
                    action_type == "brake"
                    and last_kept_action_type == "brake"
                    and apex_gap_m >= 22.0
                    and _has_brake_reset(samples, last_kept_action_idx, action_idx)
                )
                if not allow_close_brake_pair:
                    continue

        seg_start = look_back_120[apex_idx]
        seg_end = look_ahead_120[apex_idx]
        seg_speed_hi = max(speed[seg_start : seg_end + 1])
        seg_speed_lo = min(speed[seg_start : seg_end + 1])
        speed_drop_kph = (seg_speed_hi - seg_speed_lo) * 3.6
        turn_near = abs(math.degrees(_angle_diff_rad(heading[look_ahead_120[apex_idx]], heading[look_back_120[apex_idx]])))
        turn_far = abs(math.degrees(_angle_diff_rad(heading[look_ahead_250[apex_idx]], heading[look_back_120[apex_idx]])))
        turn_strength = max(turn_near, turn_far * 0.7)

        if action_type == "flat":
            if turn_strength < 13.0:
                continue
            if speed_drop_kph < 10.0:
                continue

        end_i = min(n - 1, look_ahead_120[apex_idx])
        chunk = samples[action_idx : end_i + 1]
        driving_gears = [x.gear for x in chunk if x.gear > 0]
        min_gear = min(driving_gears) if driving_gears else max(samples[action_idx].gear, 1)
        approach_i = max(0, look_back_120[action_idx])
        min_speed_idx = min(range(seg_start, seg_end + 1), key=lambda j: speed[j])

        corners.append(
            BrakeZone(
                index=len(corners),
                start_pct=samples[action_idx].lap_pct,
                apex_pct=samples[apex_idx].lap_pct,
                end_pct=samples[end_i].lap_pct,
                action_type=action_type,
                peak_brake=max(x.brake for x in chunk),
                min_speed_mps=samples[min_speed_idx].speed_mps,
                min_gear=min_gear,
                approach_speed_mps=samples[approach_i].speed_mps,
            )
        )
        used_apex_dist.append(cum_dist[apex_idx])
        last_kept_action_idx = action_idx
        last_kept_apex_idx = apex_idx
        last_kept_action_type = action_type

    corners.sort(key=lambda z: z.start_pct)
    for idx, corner in enumerate(corners):
        corner.index = idx
    return corners


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


def _digit_word(n: int) -> str:
    words = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
    }
    return words[n]


def spell_number_digits(value: int) -> str:
    value = max(0, value)
    return "-".join(_digit_word(int(ch)) for ch in str(value))


def build_short_prepare_cue(zone: BrakeZone, current_gear: int, seconds_to_corner: int) -> str:
    min_kph = int(round(zone.min_speed_mps * 3.6))
    speed_spoken = spell_number_digits(min_kph)
    if zone.min_gear < current_gear:
        gear_part = f"downshift to {zone.min_gear}"
    elif zone.min_gear > current_gear:
        gear_part = f"upshift to {zone.min_gear}"
    else:
        gear_part = f"gear {zone.min_gear}"
    return f"In {seconds_to_corner} seconds: {speed_spoken}, {gear_part}."


def build_short_target_phrase(zone: BrakeZone, current_gear: int) -> str:
    min_kph = int(round(zone.min_speed_mps * 3.6))
    speed_spoken = spell_number_digits(min_kph)
    if zone.min_gear < current_gear:
        gear_part = f"downshift to {zone.min_gear}"
    elif zone.min_gear > current_gear:
        gear_part = f"upshift to {zone.min_gear}"
    else:
        gear_part = f"gear {zone.min_gear}"
    return f"{speed_spoken}, {gear_part}"


def reference_track_length_m(samples: List[RefSample]) -> Optional[float]:
    if len(samples) < 2:
        return None
    total = 0.0
    for i in range(1, len(samples)):
        total += _haversine_m(samples[i - 1].lat, samples[i - 1].lon, samples[i].lat, samples[i].lon)
    return total if total > 100.0 else None


def close_followup_zone(
    zones: List[BrakeZone], zone: BrakeZone, track_length_m: Optional[float], max_gap_m: float = 170.0
) -> Optional[BrakeZone]:
    if track_length_m is None:
        return None
    if zone.action_type != "brake":
        return None
    next_idx = zone.index + 1
    if next_idx >= len(zones):
        return None
    candidate = zones[next_idx]
    if candidate.action_type != "brake":
        return None
    d_pct = circular_delta(zone.start_pct, candidate.start_pct)
    d_m = d_pct * track_length_m
    if 0.0 < d_m <= max_gap_m:
        return candidate
    return None


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
    zones = extract_corners(samples=samples)
    if not zones:
        raise RuntimeError("No corner events found in the reference CSV")
    ref_track_len_m = reference_track_length_m(samples)

    print(f"Loaded {len(samples)} reference samples")
    print(f"Extracted {len(zones)} corners")

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
    min_cue_speed_mps = 2.5

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
        on_track_raw = ir["IsOnTrackCar"]
        on_track = bool(on_track_raw) if on_track_raw is not None else True

        if not on_track or speed_mps < min_cue_speed_mps:
            now = time.time()
            if now - last_wait_log >= 3.0:
                print("Waiting to drive... cues start once you're on track and moving.")
                last_wait_log = now
            last_pct = lap_pct
            time.sleep(0.1)
            continue

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
            # Trigger "Now." close to the reference brake-onset sample (zone.start_pct),
            # not at a large lead time. Window is short to align with actual brake application.
            if track_len:
                action_pct = _clamp(
                    (speed_mps * 0.20) / track_len,
                    min_lookahead_pct * 0.05,
                    max_lookahead_pct * 0.20,
                )
            else:
                action_pct = min_lookahead_pct * 0.10

            if (
                not state.prepare_done
                and d_pct <= lookahead_pct
            ):
                if d_m is not None and speed_mps > 0.5:
                    seconds_to_corner = max(1, int(round(d_m / speed_mps)))
                else:
                    seconds_to_corner = max(1, int(round(lookahead_seconds)))
                followup = close_followup_zone(zones, zone, ref_track_len_m)
                if followup is not None:
                    first_part = build_short_target_phrase(zone, gear)
                    second_part = build_short_target_phrase(followup, zone.min_gear)
                    speaker.say(f"In {seconds_to_corner} seconds: {first_part} then {second_part}.")
                    followup_key = (lap_count, followup.index)
                    followup_state = stage_state.setdefault(followup_key, ZoneAnnouncementState())
                    followup_state.prepare_done = True
                    followup_state.action_done = True
                else:
                    speaker.say(build_short_prepare_cue(zone, gear, seconds_to_corner))
                state.prepare_done = True
                last_spoken_time = now

            if (
                not state.action_done
                and d_pct <= action_pct
            ):
                if zone.action_type == "flat":
                    speaker.say("Flat.")
                elif zone.action_type == "lift" and zone.peak_brake < 0.05:
                    speaker.say("Lift now.")
                else:
                    speaker.say("Now.")
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
