#!/usr/bin/env python3
"""Camera speech-to-text (issue #40): read the shared listener's inbox and
append what was said to one timestamped text file per camera per day.

  /audio/inbox/transcriber/<NN-cam>_r<run>_<UTC start>Z.wav    (listener.py)
  --> high-pass (cuts wind and rumble) --> Silero VAD --> faster-whisper
  --> TRANSCRIPT_DIR/<YYYY-MM-DD>/<cam>.txt, lines "[HH:MM:SS] text" (local TZ)

Audio with no speech never reaches Whisper, which keeps the idle cost near
zero (benchmark on the fleet Mac: VAD ~0.2% of a core per camera) and avoids
Whisper's habit of inventing text ("Thank you.") from silence. Segments
Whisper itself marks as unreliable are dropped (no_speech_prob and
avg_logprob, Whisper's own rule, plus a repetition check).

Segment boundaries: speech still running at the end of a file is not
transcribed yet. It is carried over and prepended to that camera's next file
when that file starts where this one ended, so a sentence cut by the 30 s
boundary is transcribed once, whole. Carried audio that has no successor
within FLUSH_AFTER_S (or exceeds MAX_CARRY_S) is transcribed on its own.

Privacy: transcripts are written owner-only (umask 077) and are NEVER logged.
Logs carry counts and timings only, because fleet logs end up in the GitHub
health issue. Transcripts older than RETENTION_DAYS are deleted.

Environment:
  INBOX_DIR        default /audio/inbox/transcriber; each file is deleted when done
  DATA_DIR         default /data (heartbeat for healthcheck.py)
  TRANSCRIPT_DIR   default DATA_DIR/transcripts
  TRANSCRIBE_CAMERAS  optional comma list; files from other cameras are deleted
                   unread (e.g. leave out a camera that hears the neighbours)
  RETENTION_DAYS   default 30 (1..3650)
  MODEL            default /opt/models/small.en (baked into the image)
  CPU_THREADS      default 2
  HIGHPASS_HZ      default 100 (0 = off)
  MIN_SPEECH_S     default 1.0 (0 = off): skip Whisper unless the VAD found at
                   least one speech region this long (padding included, so
                   1.0 is ~0.6 s of sound). Short car/bird/door blips are what
                   Whisper turns into "Thank you." on otherwise quiet clips.
  MIN_SPEECH_S_BY_CAMERA  optional per-camera override, e.g.
                   "west-rooftop=1.5,east-rooftop=1.5"
  NO_SPEECH_MAX, LOGPROB_MIN, COMPRESSION_MAX  segment filter (0.6, -1.0, 2.4)
  TZ               local time for file dates and line times

Usage:
  python transcriber.py              run until SIGTERM/SIGINT
  python transcriber.py --selftest   real model: speech fixture -> text, noise -> nothing
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import re
import signal
import sys
import tempfile
import threading
import time
import traceback
import wave
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

log = logging.getLogger("transcriber")

SR = 16000
# Inbox file name; the same pattern as stream.QUEUED_SPLIT_RE (a test keeps
# them equal). Copied, not imported, so this image needs none of birdlisten.
INBOX_RE = re.compile(r"^(\d{2,}-[A-Za-z0-9_-]+?)_(r\d{4,}_\d{8}T\d{6}Z\.wav)$")
SEG_TIME_RE = re.compile(r"_(\d{8}T\d{6})Z\.wav$")
DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
HEARTBEAT_FILE = "heartbeat"          # healthcheck.py reads DATA_DIR/heartbeat

CONTIGUOUS_S = 3.0                    # next file counts as the continuation within this gap
TAIL_S = 0.5                          # speech ending this close to the end is "still running"
MAX_CARRY_S = 25.0                    # never hold back more than this
FLUSH_AFTER_S = 75.0                  # carried audio with no successor by then is transcribed alone
PAD_S = 0.2                           # start the carried part this far before the speech


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    inbox_dir: Path = Path("/audio/inbox/transcriber")
    data_dir: Path = Path("/data")
    transcript_dir: Path = Path("/data/transcripts")
    cameras: frozenset[str] | None = None
    retention_days: int = 30
    model: str = "/opt/models/small.en"
    cpu_threads: int = 2
    highpass_hz: float = 100.0
    min_speech_s: float = 1.0
    min_speech_by_camera: tuple[tuple[str, float], ...] = ()
    no_speech_max: float = 0.6
    logprob_min: float = -1.0
    compression_max: float = 2.4
    tz: str = "UTC"
    poll_s: float = 1.0

    def min_speech_for(self, camera: str) -> float:
        return dict(self.min_speech_by_camera).get(camera, self.min_speech_s)


def _parse_min_speech_by_camera(raw: str) -> tuple[tuple[str, float], ...]:
    out = {}
    for item in (p.strip() for p in raw.split(",")):
        if not item:
            continue
        cam, sep, val = item.partition("=")
        cam = cam.strip()
        try:
            v = float(val) if sep and cam else None
        except ValueError:
            v = None
        if v is None or not 0 <= v <= 30:
            raise ConfigError(f"MIN_SPEECH_S_BY_CAMERA: expected camera=seconds (0..30), got {item!r}")
        out[cam] = v
    return tuple(sorted(out.items()))


def load_config(env=os.environ) -> Config:
    def path(name, default):
        p = Path(env.get(name, "").strip() or default)
        if not p.is_absolute():
            raise ConfigError(f"{name} must be an absolute path")
        return p

    def num(name, default, lo, hi, kind=float):
        raw = env.get(name, "").strip()
        if not raw:
            return default
        try:
            v = kind(raw)
        except ValueError:
            raise ConfigError(f"{name} must be a number, got {raw!r}") from None
        if not lo <= v <= hi:
            raise ConfigError(f"{name} must be {lo}..{hi}, got {raw}")
        return v

    data = path("DATA_DIR", "/data")
    cams_raw = env.get("TRANSCRIBE_CAMERAS", "").strip()
    cams = frozenset(c.strip() for c in cams_raw.split(",") if c.strip()) or None
    tz = env.get("TZ", "").strip() or "UTC"
    try:
        ZoneInfo(tz)
    except Exception:                                 # noqa: BLE001 -- any bad zone name
        raise ConfigError(f"TZ {tz!r} is not a known time zone") from None
    return Config(
        inbox_dir=path("INBOX_DIR", "/audio/inbox/transcriber"),
        data_dir=data,
        transcript_dir=path("TRANSCRIPT_DIR", str(data / "transcripts")),
        cameras=cams,
        retention_days=num("RETENTION_DAYS", 30, 1, 3650, int),
        model=env.get("MODEL", "").strip() or "/opt/models/small.en",
        cpu_threads=num("CPU_THREADS", 2, 1, 16, int),
        highpass_hz=num("HIGHPASS_HZ", 100.0, 0, 1000),
        min_speech_s=num("MIN_SPEECH_S", 1.0, 0, 30),
        min_speech_by_camera=_parse_min_speech_by_camera(env.get("MIN_SPEECH_S_BY_CAMERA", "")),
        no_speech_max=num("NO_SPEECH_MAX", 0.6, 0, 1),
        logprob_min=num("LOGPROB_MIN", -1.0, -10, 0),
        compression_max=num("COMPRESSION_MAX", 2.4, 1, 10),
        tz=tz,
    )


# ----------------------------------------------------------------- audio
def parse_name(name: str) -> tuple[str, dt.datetime] | None:
    """'01-garage_r0001_20261117T000238Z.wav' -> ('garage', aware UTC start), else None."""
    m = INBOX_RE.match(name)
    if not m:
        return None
    t = SEG_TIME_RE.search(name).group(1)
    start = dt.datetime.strptime(t, "%Y%m%dT%H%M%S").replace(tzinfo=dt.timezone.utc)
    return m.group(1).split("-", 1)[1], start


def read_wav(path: Path):
    """Mono s16 WAV at any rate -> float32 numpy array at 16 kHz."""
    import numpy as np
    with wave.open(str(path), "rb") as w:
        rate, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        if ch != 1 or width != 2:
            raise ValueError(f"expected mono 16-bit WAV, got {ch} ch, {8 * width}-bit")
        x = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
    if rate == SR:
        return x
    if rate % SR == 0:                               # 48 kHz from the listener: average groups of 3
        k = rate // SR
        x = x[: len(x) // k * k]
        return x.reshape(-1, k).mean(axis=1).astype(np.float32)
    idx = np.arange(0, len(x), rate / SR)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


def highpass(x, hz: float):
    """Remove everything below `hz` (wind, traffic rumble, mains hum) with an
    FFT mask and a short ramp. Whole-clip FFT: ~20 ms for 30 s on the Mac."""
    import numpy as np
    if hz <= 0 or len(x) == 0:
        return x
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    gain = np.clip((freqs - hz * 0.5) / (hz * 0.5), 0.0, 1.0)  # 0 below hz/2, ramp to 1 at hz
    return np.fft.irfft(spec * gain, n=len(x)).astype(np.float32)


# ----------------------------------------------------------------- model seams
@dataclass(frozen=True)
class Segment:
    start: float                      # seconds from the start of the audio passed in
    end: float
    text: str
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 1.0


def silero_vad(audio) -> list[tuple[float, float]]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    ts = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=500, speech_pad_ms=200))
    return [(t["start"] / SR, t["end"] / SR) for t in ts]


def whisper(cfg: Config):
    """Load the model once; return transcribe(audio) -> [Segment]."""
    from faster_whisper import WhisperModel
    model = WhisperModel(cfg.model, device="cpu", compute_type="int8", cpu_threads=cfg.cpu_threads,
                         local_files_only=True)

    def transcribe(audio) -> list[Segment]:
        segs, _info = model.transcribe(audio, language="en", beam_size=1, vad_filter=True,
                                       condition_on_previous_text=False)
        return [Segment(s.start, s.end, s.text, s.no_speech_prob, s.avg_logprob, s.compression_ratio)
                for s in segs]
    return transcribe


def keep(seg: Segment, cfg: Config) -> bool:
    """Whisper's own reliability rules: drop silence it is unsure of, very low
    confidence, and repetition loops."""
    if not seg.text.strip():
        return False
    if seg.no_speech_prob > cfg.no_speech_max and seg.avg_logprob < cfg.logprob_min:
        return False
    if seg.avg_logprob < cfg.logprob_min - 0.5:
        return False
    return seg.compression_ratio <= cfg.compression_max


# ----------------------------------------------------------------- output
class Writer:
    def __init__(self, root: Path, tz: str):
        self.root = Path(root)
        self.tz = ZoneInfo(tz)

    def write(self, camera: str, when_utc: dt.datetime, text: str) -> Path:
        local = when_utc.astimezone(self.tz)
        d = self.root / local.strftime("%Y-%m-%d")
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        p = d / f"{re.sub(r'[^A-Za-z0-9_-]', '_', camera)}.txt"
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            f.write(f"[{local:%H:%M:%S}] {' '.join(text.split())}\n")
        return p

    def prune(self, retention_days: int, today_local: dt.date) -> int:
        """Delete date folders older than retention_days. Only YYYY-MM-DD folders
        and the .txt files in them are touched."""
        if not self.root.is_dir():
            return 0
        cutoff = today_local - dt.timedelta(days=retention_days)
        removed = 0
        for d in self.root.iterdir():
            if not (d.is_dir() and not d.is_symlink() and DATE_DIR_RE.match(d.name)):
                continue
            if dt.date.fromisoformat(d.name) >= cutoff:
                continue
            for f in d.iterdir():
                if f.suffix == ".txt" and f.is_file():
                    f.unlink()
            try:
                d.rmdir()
                removed += 1
            except OSError:                           # something else is in there: leave it
                log.warning("transcripts %s not removed: not empty", d.name)
        return removed


# ----------------------------------------------------------------- per camera
@dataclass
class Carry:
    audio: object                     # numpy array
    start_utc: dt.datetime
    expected_next_utc: dt.datetime
    since_mono: float


@dataclass
class Stats:
    files: int = 0
    skipped: int = 0
    failed: int = 0
    lines: int = 0
    audio_s: float = 0.0
    speech_s: float = 0.0
    whisper_s: float = 0.0
    dropped_segments: int = 0
    short_skipped: int = 0            # had speech, but no region reached MIN_SPEECH_S


class Transcriber:
    def __init__(self, cfg: Config, *, transcribe, vad=silero_vad, clock=time.monotonic):
        self.cfg = cfg
        self.writer = Writer(cfg.transcript_dir, cfg.tz)
        self._transcribe = transcribe
        self._vad = vad
        self._clock = clock
        self.carry: dict[str, Carry] = {}
        self.stats = Stats()

    # -- one inbox file
    def process_file(self, path: Path) -> None:
        parsed = parse_name(path.name)
        if parsed is None:
            return
        camera, start = parsed
        try:
            if self.cfg.cameras is not None and camera not in self.cfg.cameras:
                self.stats.skipped += 1
                return
            audio = highpass(read_wav(path), self.cfg.highpass_hz)
            self.process_audio(camera, start, audio)
            self.stats.files += 1
        except Exception as exc:                      # one bad file never stops the loop
            self.stats.failed += 1
            log.error("%s: %s failed: %s", camera, path.name, type(exc).__name__ + (f": {exc}" if str(exc) else ""))
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("could not delete %s: %s", path.name, exc)

    def process_audio(self, camera: str, start: dt.datetime, audio) -> int:
        """Transcribe one clip (with any carried speech in front). Returns lines written."""
        import numpy as np
        dur = len(audio) / SR
        written = 0
        c = self.carry.pop(camera, None)
        base = start
        if c is not None:
            if abs((start - c.expected_next_utc).total_seconds()) <= CONTIGUOUS_S:
                audio = np.concatenate([c.audio, audio])
                base = c.start_utc
            else:                                     # a gap: what was carried ends there
                written += self._run(camera, c.start_utc, c.audio)
        total = len(audio) / SR
        speech = self._vad(audio)
        self.stats.audio_s += dur
        if speech:
            last_start, last_end = speech[-1]
            running = total - last_end <= TAIL_S
            cut = max(0.0, last_start - PAD_S)
            if running and cut > 0 and total - cut <= MAX_CARRY_S:
                i = int(cut * SR)
                self.carry[camera] = Carry(audio[i:], base + dt.timedelta(seconds=cut),
                                           start + dt.timedelta(seconds=dur), self._clock())
                audio, speech = audio[:i], speech[:-1]
        if speech:
            self.stats.speech_s += sum(e - s for s, e in speech)
            written += self._run(camera, base, audio, speech)
        return written

    def flush_idle(self, *, force: bool = False) -> int:
        """Transcribe carried speech whose continuation never came."""
        written = 0
        now = self._clock()
        for camera, c in list(self.carry.items()):
            if force or now - c.since_mono >= FLUSH_AFTER_S:
                del self.carry[camera]
                written += self._run(camera, c.start_utc, c.audio)
        return written

    def _run(self, camera: str, base: dt.datetime, audio, speech=None) -> int:
        if len(audio) < SR // 10:
            return 0
        need = self.cfg.min_speech_for(camera)
        if need > 0:
            if speech is None:                        # carried audio: not run through the VAD yet
                speech = self._vad(audio)
            if not any(e - s >= need for s, e in speech):
                self.stats.short_skipped += 1
                return 0
        t0 = self._clock()
        segs = self._transcribe(audio)
        self.stats.whisper_s += self._clock() - t0
        n = 0
        for s in segs:
            if not keep(s, self.cfg):
                self.stats.dropped_segments += 1
                continue
            self.writer.write(camera, base + dt.timedelta(seconds=s.start), s.text)
            n += 1
        self.stats.lines += n
        return n

    # -- the loop
    def poll_once(self) -> int:
        """Process every complete inbox file, oldest first. Returns how many."""
        try:
            names = os.listdir(self.cfg.inbox_dir)
        except FileNotFoundError:
            return 0
        todo = sorted(((p[1], n) for n in names if (p := parse_name(n))), key=lambda t: t)
        for _start, n in todo:
            _beat(self.cfg.data_dir)                  # a long backlog is progress, not a hang
            self.process_file(Path(self.cfg.inbox_dir) / n)
        self.flush_idle()
        return len(todo)


def _beat(data_dir: Path) -> None:
    p = Path(data_dir) / HEARTBEAT_FILE
    try:
        try:
            os.utime(p)
        except FileNotFoundError:
            p.touch()
    except OSError:
        pass


def run(cfg: Config, *, transcribe=None, vad=silero_vad, stop: threading.Event | None = None,
        signalled=lambda: False, summary_s: float = 300.0, clock=time.monotonic,
        today=lambda tz: dt.datetime.now(ZoneInfo(tz)).date()) -> int:
    """Until stop/signal. 0 after a signal, 1 if the inbox or model is unusable."""
    stop = stop or threading.Event()
    os.umask(0o077)
    if not Path(cfg.inbox_dir).is_dir():
        log.error("inbox %s not found (is the listener's audio volume mounted?)", cfg.inbox_dir)
        return 1
    if not os.access(cfg.inbox_dir, os.W_OK | os.X_OK):
        log.error("inbox %s not writable: finished files could not be deleted", cfg.inbox_dir)
        return 1
    Path(cfg.transcript_dir).mkdir(parents=True, exist_ok=True, mode=0o700)
    if transcribe is None:
        try:
            transcribe = whisper(cfg)
        except Exception as exc:                      # noqa: BLE001
            log.error("model %s failed to load: %s", cfg.model, type(exc).__name__)
            return 1
    t = Transcriber(cfg, transcribe=transcribe, vad=vad, clock=clock)
    cams = "all cameras" if cfg.cameras is None else ",".join(sorted(cfg.cameras))
    mins = ", ".join([f"min speech {cfg.min_speech_s:g}s"] + [f"{c}={v:g}s" for c, v in cfg.min_speech_by_camera])
    log.info("transcriber: inbox %s, %s, transcripts %s, keep %d days, %s", cfg.inbox_dir, cams,
             cfg.transcript_dir, cfg.retention_days, mins)
    next_summary = clock() + summary_s
    pruned_on = None
    try:
        while not (stop.is_set() or signalled()):
            _beat(cfg.data_dir)
            day = today(cfg.tz)
            if day != pruned_on:
                n = t.writer.prune(cfg.retention_days, day)
                if n:
                    log.info("removed %d transcript days older than %d days", n, cfg.retention_days)
                pruned_on = day
            t.poll_once()
            if clock() >= next_summary:
                s = t.stats
                log.info("transcriber summary files=%d skipped=%d failed=%d audio_s=%.0f speech_s=%.0f "
                         "whisper_s=%.1f lines=%d dropped_segments=%d short_skipped=%d carried=%d", s.files,
                         s.skipped, s.failed, s.audio_s, s.speech_s, s.whisper_s, s.lines, s.dropped_segments,
                         s.short_skipped, len(t.carry))
                t.stats = Stats()
                next_summary = clock() + summary_s
            stop.wait(cfg.poll_s)
    except Exception:
        log.error("transcriber crashed:\n%s", traceback.format_exc())
        return 1
    finally:
        try:
            t.flush_idle(force=True)                  # the carried tail of each camera
        except Exception:                             # noqa: BLE001
            log.warning("could not transcribe carried audio at shutdown")
        log.info("transcriber stopped")
    return 0


def main(argv=None, env=os.environ) -> int:
    argv = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    if argv == ["--selftest"]:
        return _selftest_main(env)
    if argv:
        print(__doc__)
        return 2
    try:
        cfg = load_config(env)
    except ConfigError as exc:
        log.error("config: %s", exc)
        return 2
    flag = {"stop": False}

    def handler(signum, _frame):
        flag["stop"] = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, handler)
    return run(cfg, signalled=lambda: flag["stop"])


# ----------------------------------------------------------------- CI self-test
def _write_wav(path: Path, x, rate: int = 48000) -> None:
    import numpy as np
    y = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(y.tobytes())


def selftest(env=os.environ, fixture: Path = Path(__file__).with_name("selftest") / "speech.wav") -> list[str]:
    """With the real model and VAD, in the image: the speech fixture (placed in
    a 30 s camera-like clip over noise) yields its words, and 30 s of noise and
    hum alone yields no line at all. Returns problems; [] = pass."""
    import numpy as np
    problems = []
    rng = np.random.default_rng(0)
    t = np.arange(30 * 48000) / 48000
    noise = (rng.normal(0, 0.003, len(t)) + 0.01 * np.sin(2 * np.pi * 60 * t)).astype(np.float32)
    speech = read_wav(fixture)
    up = np.repeat(speech, 3)                          # 16 -> 48 kHz, as the listener delivers
    clip = noise.copy()
    clip[5 * 48000: 5 * 48000 + len(up)] += up
    with tempfile.TemporaryDirectory(prefix="transcriber-selftest-") as tmp:
        tmp = Path(tmp)
        inbox, data = tmp / "inbox", tmp / "data"
        inbox.mkdir()
        _write_wav(inbox / "00-porch_r0001_20261117T120000Z.wav", clip)
        _write_wav(inbox / "01-yard_r0001_20261117T120000Z.wav", noise)
        cfg = load_config({**env, "INBOX_DIR": str(inbox), "DATA_DIR": str(data), "TZ": "UTC"})
        tr = Transcriber(cfg, transcribe=whisper(cfg))
        tr.poll_once()
        tr.flush_idle(force=True)
        porch = data / "transcripts" / "2026-11-17" / "porch.txt"
        yard = data / "transcripts" / "2026-11-17" / "yard.txt"
        text = porch.read_text() if porch.exists() else ""
        if not ("package" in text.lower() and "porch" in text.lower()):
            problems.append(f"speech clip: expected 'package' and 'porch', got {text!r}")
        if text and not text.startswith("[12:00:0"):
            problems.append(f"speech clip: expected a [12:00:0x] timestamp, got {text[:12]!r}")
        if yard.exists():
            problems.append(f"noise-only clip produced text: {yard.read_text()!r}")
        if os.listdir(inbox):
            problems.append(f"inbox not emptied: {os.listdir(inbox)}")
        if porch.exists() and porch.stat().st_mode & 0o077:
            problems.append(f"transcript is readable by others: {oct(porch.stat().st_mode)}")
    return problems


def _selftest_main(env) -> int:
    problems = selftest(env)
    if problems:
        print("transcriber selftest FAILED:")
        for p in problems:
            print(f"  {p}")
        return 1
    print("transcriber selftest ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
