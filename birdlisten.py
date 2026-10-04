#!/usr/bin/env python3
"""Listen to Reolink cameras and log which birds BirdNET hears.

Pipeline, once per call to main() (loop.py calls it back to back):

  camera RTSP stream --ffmpeg--> N-second mono 48 kHz WAV
                     --BirdNET (birdnetlib)--> [(species, confidence, t0, t1)]
                     --dedupe per window--> SQLite + stdout + optional ntfy push

Each camera is one microphone. Cameras are listened to one after another in a
single pass so the CPU (an old Intel MacBook) only ever runs one BirdNET
analysis at a time. A pass over two cameras with 30 s clips takes ~70 s.

Environment (all read at startup):
  CAMERAS         required. Comma-separated name=rtsp-url pairs, e.g.
                    front=rtsp://admin:pw@192.168.1.20:554/h264Preview_01_sub,back=rtsp://...
                  Use the sub-stream: smaller video, same audio. The camera's
                  "Record Audio" setting must be ON or the stream has no audio.
  LATITUDE        required for BirdNET's location/date species filter
  LONGITUDE       required
  CLIP_SECONDS    seconds of audio per camera per pass (default 30)
  MIN_CONFIDENCE  0..1, drop detections below this (default 0.5)
  DATA_DIR        where the SQLite db (and optional clips) live (default /data)
  KEEP_CLIPS      1 = keep the WAV of any clip that had a detection (default 0)
  NTFY_TOPIC      optional. If set, POST a line per new species to ntfy.sh
  NTFY_SERVER     default https://ntfy.sh
  NOTIFY_COOLDOWN_MIN  don't re-notify the same species within N min (default 60)
  TZ              for timestamps in messages

Exit codes: 0 ok (even with zero detections), 2 config error, 1 every camera
failed this pass (BirdNET model missing, ffmpeg missing, all cameras down).
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import re
import resource
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("birdlisten")


class ConfigError(RuntimeError):
    pass


# rtsp://user:PASSWORD@host -> rtsp://user:***@host, wherever it appears in a
# string. ffmpeg echoes the URL in several forms (percent-encoded, with a
# query string, mid-sentence), so scrub by pattern rather than exact match.
_CRED_RE = re.compile(r"(rtsps?://[^:/@\s]+:)[^@\s]+@")


def scrub(text: str) -> str:
    return _CRED_RE.sub(r"\1***@", text)


# ----------------------------------------------------------------- config
@dataclass(frozen=True)
class Camera:
    name: str
    rtsp: str

    def redacted(self) -> str:
        """rtsp://user:pass@host/... -> rtsp://user:***@host/... for logs."""
        return scrub(self.rtsp)


@dataclass(frozen=True)
class Config:
    cameras: tuple[Camera, ...]
    lat: float
    lon: float
    clip_seconds: int
    min_conf: float
    data_dir: Path
    keep_clips: bool
    ntfy_topic: str | None
    ntfy_server: str
    notify_cooldown: dt.timedelta


def parse_cameras(raw: str) -> tuple[Camera, ...]:
    """'front=rtsp://a,back=rtsp://b' -> (Camera, Camera). Commas inside URLs
    are not supported; Reolink URLs never contain them."""
    cams = []
    for item in filter(None, (s.strip() for s in raw.split(","))):
        if "=" not in item:
            raise ConfigError(f"CAMERAS entry {item!r} is not name=rtsp-url")
        name, url = item.split("=", 1)
        name, url = name.strip(), url.strip()
        if not name or not url.startswith("rtsp://"):
            raise ConfigError(f"CAMERAS entry {item!r}: need a name and an rtsp:// url")
        cams.append(Camera(name, url))
    if not cams:
        raise ConfigError("CAMERAS is empty")
    if len({c.name for c in cams}) != len(cams):
        raise ConfigError("CAMERAS names must be unique")
    return tuple(cams)


def load_config(env=os.environ) -> Config:
    def req(k: str) -> str:
        v = env.get(k, "").strip()
        if not v:
            raise ConfigError(f"missing required environment variable {k}")
        return v

    try:
        return Config(
            cameras=parse_cameras(req("CAMERAS")),
            lat=float(req("LATITUDE")),
            lon=float(req("LONGITUDE")),
            clip_seconds=int(env.get("CLIP_SECONDS", "30")),
            min_conf=float(env.get("MIN_CONFIDENCE", "0.5")),
            data_dir=Path(env.get("DATA_DIR", "/data")),
            keep_clips=env.get("KEEP_CLIPS", "0") == "1",
            ntfy_topic=env.get("NTFY_TOPIC", "").strip() or None,
            ntfy_server=env.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/"),
            notify_cooldown=dt.timedelta(minutes=int(env.get("NOTIFY_COOLDOWN_MIN", "60"))),
        )
    except ValueError as exc:
        raise ConfigError(f"bad numeric setting: {exc}") from exc


# ----------------------------------------------------------------- capture
def capture(cam: Camera, seconds: int, out: Path) -> None:
    """Pull `seconds` of audio from the camera into a mono 48 kHz WAV.

    -rtsp_transport tcp: Reolink drops UDP packets on Wi-Fi; TCP is reliable.
    -vn: drop video, we only want the AAC track.  48 kHz mono is what BirdNET
    expects; birdnetlib would resample anyway but doing it in ffmpeg is faster.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-rtsp_transport", "tcp",
        "-i", cam.rtsp,
        "-t", str(seconds),
        "-vn", "-ac", "1", "-ar", "48000", "-acodec", "pcm_s16le",
        "-y", str(out),
    ]
    # Give the stream time to connect plus the clip length plus slack.
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=seconds + 30)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg timed out after {seconds + 30}s") from exc
    if proc.returncode != 0:
        # ffmpeg's stderr echoes the URL, credentials included, in whatever
        # form it likes; scrub by pattern, never by matching the known URL.
        err = scrub(proc.stderr.strip()) or f"exit {proc.returncode}"
        raise RuntimeError(f"ffmpeg failed: {err.splitlines()[-1]}")
    size = out.stat().st_size if out.exists() else 0
    if size < 48000:  # pcm_s16le 48 kHz mono: 96 kB/s, so this is < 0.5 s
        raise RuntimeError(
            f"ffmpeg produced {size} bytes of audio (< 0.5 s). Either the camera's"
            " 'Record Audio' setting is off, or the stream dropped immediately."
        )


# ----------------------------------------------------------------- analyze
@dataclass(frozen=True)
class Detection:
    common_name: str
    scientific_name: str
    confidence: float
    start: float  # seconds into the clip
    end: float


_ANALYZER = None
# Cumulative (wall_s, cpu_s) spent loading the model, so listen_once can keep
# a first-pass model load out of that clip's analyze_s / cpu_s.
_MODEL_LOAD_COST = [0.0, 0.0]


def analyzer():
    """Load the BirdNET model once per process; it takes a few seconds."""
    global _ANALYZER
    if _ANALYZER is None:
        t0, c0 = time.perf_counter(), _cpu_seconds()
        from birdnetlib.analyzer import Analyzer  # heavy import, keep it lazy

        _ANALYZER = Analyzer()
        wall, cpu = time.perf_counter() - t0, _cpu_seconds() - c0
        _MODEL_LOAD_COST[0] += wall
        _MODEL_LOAD_COST[1] += cpu
        log.info("timing model_load_s=%.2f cpu_s=%.2f rss_mb=%.0f", wall, cpu, _max_rss_mb())
    return _ANALYZER


def analyze(wav: Path, cfg: Config, when: dt.datetime) -> list[Detection]:
    """Run BirdNET over the clip. lat/lon/date let BirdNET drop species that
    are not plausible at this place and time of year, which removes most
    false positives for free."""
    from birdnetlib import Recording

    rec = Recording(
        analyzer(), str(wav),
        lat=cfg.lat, lon=cfg.lon, date=when.date(),
        min_conf=cfg.min_conf,
    )
    rec.analyze()
    return [
        Detection(
            common_name=d["common_name"],
            scientific_name=d["scientific_name"],
            confidence=float(d["confidence"]),
            start=float(d["start_time"]),
            end=float(d["end_time"]),
        )
        for d in rec.detections
    ]


def best_per_species(dets: list[Detection]) -> dict[str, Detection]:
    """One clip often yields the same bird in several 3 s windows; keep the
    most confident window per species for storage and notification."""
    best: dict[str, Detection] = {}
    for d in dets:
        if d.common_name not in best or d.confidence > best[d.common_name].confidence:
            best[d.common_name] = d
    return best


# ----------------------------------------------------------------- store
SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
  id            INTEGER PRIMARY KEY,
  heard_at      TEXT NOT NULL,      -- ISO-8601 UTC, start of the clip
  camera        TEXT NOT NULL,
  common_name   TEXT NOT NULL,
  scientific_name TEXT NOT NULL,
  confidence    REAL NOT NULL,
  clip_offset_s REAL NOT NULL,
  clip_path     TEXT
);
CREATE INDEX IF NOT EXISTS idx_detections_species_time ON detections(common_name, heard_at);
CREATE TABLE IF NOT EXISTS notified (
  common_name TEXT PRIMARY KEY,
  last_sent   TEXT NOT NULL
);
"""


def open_db(data_dir: Path) -> sqlite3.Connection:
    data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(data_dir / "birdlisten.sqlite")
    conn.executescript(SCHEMA)
    return conn


def record(conn: sqlite3.Connection, when: dt.datetime, cam: Camera, d: Detection, clip: str | None) -> None:
    conn.execute(
        "INSERT INTO detections(heard_at,camera,common_name,scientific_name,confidence,clip_offset_s,clip_path)"
        " VALUES (?,?,?,?,?,?,?)",
        (when.isoformat(timespec="seconds"), cam.name, d.common_name, d.scientific_name,
         d.confidence, d.start, clip),
    )
    conn.commit()


def should_notify(conn: sqlite3.Connection, species: str, now: dt.datetime, cooldown: dt.timedelta) -> bool:
    row = conn.execute("SELECT last_sent FROM notified WHERE common_name=?", (species,)).fetchone()
    if row and now - dt.datetime.fromisoformat(row[0]) < cooldown:
        return False
    conn.execute(
        "INSERT INTO notified(common_name,last_sent) VALUES(?,?)"
        " ON CONFLICT(common_name) DO UPDATE SET last_sent=excluded.last_sent",
        (species, now.isoformat(timespec="seconds")),
    )
    conn.commit()
    return True


# ----------------------------------------------------------------- notify
def notify(cfg: Config, title: str, body: str) -> None:
    """ntfy.sh: free, no account, phone app. `NTFY_TOPIC` is effectively the
    password, so make it unguessable. Failures are logged, never fatal."""
    if not cfg.ntfy_topic:
        return
    req = urllib.request.Request(
        f"{cfg.ntfy_server}/{cfg.ntfy_topic}",
        data=body.encode(),
        headers={"Title": title, "Tags": "bird"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=10).read()
    except Exception as exc:  # noqa: BLE001
        log.warning("ntfy failed: %s", exc)


# ----------------------------------------------------------------- timing
# One `timing ...` INFO line per camera per pass, one per pass, one per model
# load. Sizing data for concurrent capture (issue #4); grep ' timing '.
def _cpu_seconds() -> float:
    """User+sys CPU of this process (all threads) plus reaped children."""
    s = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime


def _max_rss_mb() -> float:
    """High-water RSS of this process. ru_maxrss is KB on Linux, bytes on macOS."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def _timing_line(cam: str, clip_s: float, **fields) -> str:
    parts = [f"camera={cam}", f"clip_s={clip_s:.1f}"]
    for key, fmt in (("capture_s", "{:.2f}"), ("analyze_s", "{:.2f}"), ("detections", "{}"),
                     ("cpu_s", "{:.2f}"), ("rss_mb", "{:.0f}")):
        if fields.get(key) is not None:
            parts.append(f"{key}={fmt.format(fields[key])}")
    if fields.get("error"):
        # Single token so `key=value` parsing stays trivial.
        parts.append("error=" + re.sub(r"\s+", "_", fields["error"].strip())[:80])
    return "timing " + " ".join(parts)


# ----------------------------------------------------------------- one pass
def listen_once(cfg: Config, conn: sqlite3.Connection, dry_run: bool = False) -> int:
    """Capture + analyze every camera once. Returns the number of cameras that
    produced a usable clip (0 => the pass failed)."""
    ok = 0
    pass_t0 = time.perf_counter()
    analyze_total = 0.0
    for cam in cfg.cameras:
        when = dt.datetime.now(dt.timezone.utc)
        with tempfile.TemporaryDirectory(prefix="birdlisten-") as tmp:
            wav = Path(tmp) / f"{cam.name}.wav"
            t: dict = {}
            try:
                t0 = time.perf_counter()
                try:
                    capture(cam, cfg.clip_seconds, wav)
                finally:
                    t["capture_s"] = time.perf_counter() - t0
                load_wall, load_cpu = _MODEL_LOAD_COST
                t0, c0 = time.perf_counter(), _cpu_seconds()
                try:
                    dets = analyze(wav, cfg, when)
                finally:
                    # A first-call model load is logged on its own line; keep it out of this clip.
                    t["analyze_s"] = time.perf_counter() - t0 - (_MODEL_LOAD_COST[0] - load_wall)
                    t["cpu_s"] = _cpu_seconds() - c0 - (_MODEL_LOAD_COST[1] - load_cpu)
                    analyze_total += t["analyze_s"]
            except Exception as exc:  # noqa: BLE001 -- one camera failing must not stop the others
                reason = scrub(str(exc))
                log.error("%s: %s", cam.name, reason)
                log.info("%s", _timing_line(cam.name, cfg.clip_seconds, rss_mb=_max_rss_mb(), error=reason, **t))
                continue
            ok += 1

            best = best_per_species(dets)
            log.info("%s", _timing_line(cam.name, cfg.clip_seconds, detections=len(best), rss_mb=_max_rss_mb(), **t))
            if not best:
                log.info("%s: %ss, nothing above %.2f", cam.name, cfg.clip_seconds, cfg.min_conf)
                continue

            clip_path = None
            if cfg.keep_clips and not dry_run:
                dest = cfg.data_dir / "clips" / when.strftime("%Y-%m-%d")
                dest.mkdir(parents=True, exist_ok=True)
                clip_path = str(dest / f"{when.strftime('%H%M%S')}_{cam.name}.wav")
                shutil.copy(wav, clip_path)

            for name, d in sorted(best.items(), key=lambda kv: -kv[1].confidence):
                log.info("%s: %s (%s) %.0f%% at +%.0fs", cam.name, name, d.scientific_name, d.confidence * 100, d.start)
                if dry_run:
                    continue
                record(conn, when, cam, d, clip_path)
                if should_notify(conn, name, when, cfg.notify_cooldown):
                    local = when.astimezone().strftime("%H:%M")
                    notify(cfg, f"{name}", f"{local} on {cam.name} camera, {d.confidence:.0%} confidence")
    log.info("timing pass cameras=%d ok=%d wall_s=%.2f analyze_total_s=%.2f rss_mb=%.0f",
             len(cfg.cameras), ok, time.perf_counter() - pass_t0, analyze_total, _max_rss_mb())
    return ok


# ----------------------------------------------------------------- cli
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%S", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="capture and analyze, but write nothing and notify nobody")
    parser.add_argument("--check", action="store_true", help="validate config, ffmpeg, and the BirdNET model, then exit")
    parser.add_argument("--report", metavar="DAYS", type=int, help="print species heard in the last N days and exit")
    args = parser.parse_args()

    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.report is not None:
        return report(cfg, args.report)

    if args.check:
        return check(cfg)

    conn = open_db(cfg.data_dir)
    ok = listen_once(cfg, conn, dry_run=args.dry_run)
    conn.close()
    if ok == 0:
        log.error("every camera failed this pass")
        return 1
    return 0


def check(cfg: Config) -> int:
    """Pre-flight: ffmpeg present, model loads, and every camera actually
    yields audio (a 3 s capture each). The camera probe is what catches the
    real first-run failures: wrong RTSP path, bad password, Record Audio off."""
    problems = 0
    if shutil.which("ffmpeg") is None:
        print("ffmpeg: NOT FOUND", file=sys.stderr); problems += 1
    else:
        print("ffmpeg: ok")
    try:
        analyzer(); print("birdnet model: ok")
    except Exception as exc:  # noqa: BLE001
        print(f"birdnet model: FAILED ({exc})", file=sys.stderr); problems += 1
    for cam in cfg.cameras:
        with tempfile.TemporaryDirectory(prefix="birdlisten-check-") as tmp:
            try:
                capture(cam, 3, Path(tmp) / "probe.wav")
                print(f"camera {cam.name}: ok ({cam.redacted()})")
            except Exception as exc:  # noqa: BLE001
                print(f"camera {cam.name}: FAILED {scrub(str(exc))} ({cam.redacted()})", file=sys.stderr)
                problems += 1
    print(f"location: {cfg.lat}, {cfg.lon}; clip {cfg.clip_seconds}s; min_conf {cfg.min_conf}")
    print(f"notify: {'ntfy ' + cfg.ntfy_server if cfg.ntfy_topic else 'off'}")
    return 1 if problems else 0


def report(cfg: Config, days: int) -> int:
    conn = open_db(cfg.data_dir)
    # TEXT comparison is chronological only because every heard_at is written
    # by record() as a UTC-aware, fixed-width ISO string with +00:00. Keep it so.
    since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).isoformat(timespec="seconds")
    rows = conn.execute(
        "SELECT common_name, COUNT(*), MAX(confidence), MAX(heard_at) FROM detections"
        " WHERE heard_at >= ? GROUP BY common_name ORDER BY 2 DESC",
        (since,),
    ).fetchall()
    if not rows:
        print(f"no detections in the last {days} day(s)")
        return 0
    w = max(len(r[0]) for r in rows)
    print(f"{'species':<{w}}  heard  best   last")
    for name, n, best, last in rows:
        print(f"{name:<{w}}  {n:>5}  {best:>4.0%}  {last[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
