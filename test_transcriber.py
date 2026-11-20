"""Tests for transcriber.py (issue #40). No model: a fake VAD finds the tone
bursts that stand in for speech, and a fake transcribe() reports where in the
audio it was given each burst sits, so carry-over and timestamps are checkable.
The real model runs in CI (`python transcriber.py --selftest` in the image).
Run: uv run --no-project --python 3.11 --with numpy --with pytest==8.3.4 pytest -q test_transcriber.py"""

from __future__ import annotations

import datetime as dt
import logging
import os
import threading
import wave
from pathlib import Path

import numpy as np
import pytest

import transcriber as tr

UTC = dt.timezone.utc
T0 = dt.datetime(2026, 11, 17, 20, 0, 0, tzinfo=UTC)        # 12:00:00 in Los Angeles


# ----------------------------------------------------------------- helpers
def clip(seconds=30.0, bursts=(), rate=tr.SR):
    """Quiet noise with 440 Hz 'speech' bursts at (start, end) seconds."""
    rng = np.random.default_rng(1)
    t = np.arange(int(seconds * rate)) / rate
    x = rng.normal(0, 0.001, len(t)).astype(np.float32)
    for s, e in bursts:
        m = (t >= s) & (t < e)
        x[m] += 0.5 * np.sin(2 * np.pi * 440 * t[m])
    return x.astype(np.float32)


def fake_vad(audio):
    """Runs of 10 ms frames whose RMS is clearly above the noise."""
    f = tr.SR // 100
    n = len(audio) // f
    loud = np.sqrt((audio[: n * f].reshape(n, f) ** 2).mean(axis=1)) > 0.05
    out, start = [], None
    for i, v in enumerate(list(loud) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start / 100, i / 100)); start = None
    return out


class FakeWhisper:
    """One Segment per burst, text 'at <offset> for <length>' within the audio given."""

    def __init__(self, **seg_kw):
        self.calls: list[float] = []                  # length of audio per call
        self.seg_kw = seg_kw

    def __call__(self, audio):
        self.calls.append(len(audio) / tr.SR)
        return [tr.Segment(s, e, f" at {s:.1f}  for {e - s:.1f} ", **self.seg_kw) for s, e in fake_vad(audio)]


def write_wav(path: Path, x, rate=48000) -> Path:
    if rate != tr.SR:
        x = np.repeat(x, rate // tr.SR)
    y = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(y.tobytes())
    return path


def name(cam_dir: str, start: dt.datetime) -> str:
    return f"{cam_dir}_r0001_{start:%Y%m%dT%H%M%S}Z.wav"


def make(tmp_path, **env) -> tuple[tr.Config, Path]:
    inbox = tmp_path / "inbox"; inbox.mkdir(exist_ok=True)
    cfg = tr.load_config({"INBOX_DIR": str(inbox), "DATA_DIR": str(tmp_path / "data"),
                          "TZ": "America/Los_Angeles", **env})
    return cfg, inbox


def lines(cfg, cam, day="2026-11-17") -> list[str]:
    p = Path(cfg.transcript_dir) / day / f"{cam}.txt"
    return p.read_text().splitlines() if p.exists() else []


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


# ----------------------------------------------------------------- config + names
def test_config_defaults_and_overrides():
    cfg = tr.load_config({})
    assert (cfg.inbox_dir, cfg.data_dir, cfg.transcript_dir, cfg.cameras, cfg.retention_days, cfg.tz) == \
        (Path("/audio/inbox/transcriber"), Path("/data"), Path("/data/transcripts"), None, 30, "UTC")
    cfg = tr.load_config({"DATA_DIR": "/d", "TRANSCRIBE_CAMERAS": "doorbell, garage", "RETENTION_DAYS": "7",
                          "HIGHPASS_HZ": "0", "TZ": "America/Los_Angeles"})
    assert (cfg.transcript_dir, cfg.cameras, cfg.retention_days, cfg.highpass_hz) == \
        (Path("/d/transcripts"), frozenset({"doorbell", "garage"}), 7, 0.0)


@pytest.mark.parametrize("env, msg", [
    ({"INBOX_DIR": "inbox"}, "absolute"),
    ({"RETENTION_DAYS": "0"}, "1..3650"),
    ({"RETENTION_DAYS": "a"}, "must be a number"),
    ({"TZ": "Mars/Olympus"}, "time zone"),
    ({"NO_SPEECH_MAX": "2"}, "0..1"),
    ({"MIN_SPEECH_S": "-1"}, "0..30"),
    ({"MIN_SPEECH_S_BY_CAMERA": "west-rooftop"}, "camera=seconds"),
    ({"MIN_SPEECH_S_BY_CAMERA": "west-rooftop=abc"}, "camera=seconds"),
    ({"MIN_SPEECH_S_BY_CAMERA": "=2"}, "camera=seconds"),
    ({"MIN_SPEECH_S_BY_CAMERA": "garage=99"}, "camera=seconds"),
])
def test_config_errors(env, msg):
    with pytest.raises(tr.ConfigError, match=msg):
        tr.load_config(env)


def test_config_min_speech_default_and_per_camera():
    cfg = tr.load_config({})
    assert cfg.min_speech_s == 1.0 and cfg.min_speech_for("garage") == 1.0
    cfg = tr.load_config({"MIN_SPEECH_S": "0.8",
                          "MIN_SPEECH_S_BY_CAMERA": " west-rooftop=1.5, east-rooftop = 2 ,"})
    assert cfg.min_speech_for("garage") == 0.8
    assert (cfg.min_speech_for("west-rooftop"), cfg.min_speech_for("east-rooftop")) == (1.5, 2.0)


def test_inbox_pattern_matches_stream():
    import stream
    assert tr.INBOX_RE.pattern == stream.QUEUED_SPLIT_RE.pattern


def test_parse_name():
    assert tr.parse_name("01-west-rooftop_r0002_20261117T200000Z.wav") == ("west-rooftop", T0)
    assert tr.parse_name("00-front_door__r0001_20261117T200000Z.wav")[0] == "front_door_"
    assert tr.parse_name("status.json") is None


# ----------------------------------------------------------------- audio
def test_read_wav_resamples_48k_and_rejects_stereo(tmp_path):
    x = clip(2.0, [(0.5, 1.0)])
    got = tr.read_wav(write_wav(tmp_path / "a.wav", x, rate=48000))
    assert len(got) == len(x) and np.allclose(got, x, atol=2e-4)
    p = tmp_path / "s.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b"\0\0\0\0" * 10)
    with pytest.raises(ValueError, match="mono"):
        tr.read_wav(p)


def test_highpass_cuts_rumble_keeps_voice_band():
    t = np.arange(tr.SR * 2) / tr.SR
    rumble, voice = np.sin(2 * np.pi * 30 * t), np.sin(2 * np.pi * 440 * t)
    rms = lambda a: float(np.sqrt((a ** 2).mean()))           # noqa: E731
    assert rms(tr.highpass(rumble.astype(np.float32), 100)) < 0.01 * rms(rumble)
    assert abs(rms(tr.highpass(voice.astype(np.float32), 100)) - rms(voice)) < 0.01
    assert tr.highpass(voice, 0) is voice


@pytest.mark.parametrize("kw, kept", [
    ({}, True),
    ({"text": "  "}, False),
    ({"no_speech_prob": 0.9, "avg_logprob": -1.2}, False),       # Whisper's own silence rule
    ({"no_speech_prob": 0.9, "avg_logprob": -0.3}, True),        # confident despite no-speech score
    ({"avg_logprob": -1.6}, False),                              # very low confidence
    ({"compression_ratio": 3.0}, False),                         # repetition loop
])
def test_keep(kw, kept):
    seg = tr.Segment(0, 1, kw.pop("text", "hello there"), **kw)
    assert tr.keep(seg, tr.Config()) is kept


# ----------------------------------------------------------------- output
def test_writer_local_time_date_folder_and_permissions(tmp_path):
    w = tr.Writer(tmp_path / "t", "America/Los_Angeles")
    late = dt.datetime(2026, 11, 17, 6, 30, 5, tzinfo=UTC)        # 22:30:05 on Nov 16 locally
    p = w.write("garage", late, "  the   door\nis open ")
    assert p == tmp_path / "t" / "2026-11-16" / "garage.txt"
    w.write("garage", late + dt.timedelta(seconds=7), "second")
    assert p.read_text() == "[22:30:05] the door is open\n[22:30:12] second\n"
    assert p.stat().st_mode & 0o777 == 0o600


def test_prune_only_old_date_folders(tmp_path):
    root = tmp_path / "t"
    for d in ("2026-10-01", "2026-10-17", "2026-10-18", "2026-11-17", "notes"):
        (root / d).mkdir(parents=True)
        (root / d / "garage.txt").write_text("x")
    (root / "2026-10-02").mkdir(); (root / "2026-10-02" / "keep.jpg").write_text("y")
    w = tr.Writer(root, "UTC")
    assert w.prune(30, dt.date(2026, 11, 17)) == 2                # 10-01 and 10-17 (cutoff 10-18)
    assert sorted(os.listdir(root)) == ["2026-10-02", "2026-10-18", "2026-11-17", "notes"]
    assert (root / "2026-10-02" / "keep.jpg").exists()             # not ours: folder kept


# ----------------------------------------------------------------- transcriber
def test_silence_never_reaches_whisper(tmp_path):
    cfg, _ = make(tmp_path)
    fw = FakeWhisper()
    t = tr.Transcriber(cfg, transcribe=fw, vad=fake_vad)
    assert t.process_audio("garage", T0, clip()) == 0
    assert fw.calls == [] and lines(cfg, "garage") == []


# The "Thank you." problem: a car, bird or door blip ~0.5 s long passes the VAD
# and Whisper fills it with a stock phrase. Clips whose longest speech region is
# under MIN_SPEECH_S never reach Whisper.
def test_short_blip_never_reaches_whisper(tmp_path):
    cfg, _ = make(tmp_path)
    fw = FakeWhisper()
    t = tr.Transcriber(cfg, transcribe=fw, vad=fake_vad)
    assert t.process_audio("garage", T0, clip(30, [(10.0, 10.5), (20.0, 20.6)])) == 0
    assert fw.calls == [] and lines(cfg, "garage") == [] and t.stats.short_skipped == 1


def test_one_long_region_sends_the_whole_clip_short_words_included(tmp_path):
    cfg, _ = make(tmp_path)
    fw = FakeWhisper()
    t = tr.Transcriber(cfg, transcribe=fw, vad=fake_vad)
    # a quick "yes" (0.4 s) in a conversation is kept once real speech is there
    assert t.process_audio("garage", T0, clip(30, [(5.0, 5.4), (12.0, 14.5)])) == 2
    assert fw.calls == [30.0] and t.stats.short_skipped == 0


def test_rooftop_override_is_stricter_than_the_default(tmp_path):
    cfg, _ = make(tmp_path, MIN_SPEECH_S_BY_CAMERA="west-rooftop=1.5")
    fw = FakeWhisper()
    t = tr.Transcriber(cfg, transcribe=fw, vad=fake_vad)
    burst = clip(30, [(10.0, 11.2)])                             # 1.2 s
    assert t.process_audio("west-rooftop", T0, burst) == 0
    assert t.process_audio("garage", T0, burst) == 1
    assert lines(cfg, "west-rooftop") == [] and lines(cfg, "garage") == ["[12:00:10] at 10.0 for 1.2"]
    assert t.process_audio("west-rooftop", T0 + dt.timedelta(seconds=30), clip(30, [(3.0, 5.0)])) == 1


def test_min_speech_zero_turns_the_gate_off(tmp_path):
    cfg, _ = make(tmp_path, MIN_SPEECH_S="0")
    t = tr.Transcriber(cfg, transcribe=FakeWhisper(), vad=fake_vad)
    assert t.process_audio("garage", T0, clip(30, [(10.0, 10.3)])) == 1


def test_short_carried_tail_with_no_continuation_is_gated_too(tmp_path):
    cfg, _ = make(tmp_path)
    fw = FakeWhisper()
    t = tr.Transcriber(cfg, transcribe=fw, vad=fake_vad)
    t.process_audio("garage", T0, clip(30, [(29.6, 30.0)]))       # 0.4 s blip at the boundary: carried
    assert "garage" in t.carry
    assert t.flush_idle(force=True) == 0 and fw.calls == [] and t.stats.short_skipped == 1


def test_carry_joined_across_the_boundary_counts_as_one_region(tmp_path):
    cfg, _ = make(tmp_path)
    t = tr.Transcriber(cfg, transcribe=FakeWhisper(), vad=fake_vad)
    # 0.6 s at the end of one clip + 0.6 s at the start of the next = one 1.2 s region
    assert t.process_audio("garage", T0, clip(30, [(29.4, 30.0)])) == 0
    assert t.process_audio("garage", T0 + dt.timedelta(seconds=30), clip(30, [(0.0, 0.6)])) == 1
    assert len(lines(cfg, "garage")) == 1


def test_speech_mid_clip_gets_its_own_timestamp(tmp_path):
    cfg, _ = make(tmp_path)
    fw = FakeWhisper()
    t = tr.Transcriber(cfg, transcribe=fw, vad=fake_vad)
    assert t.process_audio("garage", T0, clip(30, [(12.0, 14.5)])) == 1
    assert lines(cfg, "garage") == ["[12:00:12] at 12.0 for 2.5"]
    assert fw.calls == [30.0] and not t.carry


def test_speech_across_the_boundary_is_carried_and_transcribed_once_whole(tmp_path):
    cfg, _ = make(tmp_path)
    fw = FakeWhisper()
    t = tr.Transcriber(cfg, transcribe=fw, vad=fake_vad)
    # burst A mid clip, burst B runs from 27 s into the next clip until 2 s
    assert t.process_audio("garage", T0, clip(30, [(5.0, 6.0), (27.0, 30.0)])) == 1
    assert "garage" in t.carry and lines(cfg, "garage") == ["[12:00:05] at 5.0 for 1.0"]
    assert t.process_audio("garage", T0 + dt.timedelta(seconds=30), clip(30, [(0.0, 2.0), (20.0, 21.0)])) == 2
    # B once, whole (3 s + 2 s), stamped where it began in the first clip; then C
    assert lines(cfg, "garage") == ["[12:00:05] at 5.0 for 1.0", "[12:00:27] at 0.2 for 5.0",
                                    "[12:00:50] at 23.2 for 1.0"]


def test_carry_with_a_gap_is_flushed_on_its_own(tmp_path):
    cfg, _ = make(tmp_path)
    t = tr.Transcriber(cfg, transcribe=FakeWhisper(), vad=fake_vad)
    t.process_audio("garage", T0, clip(30, [(28.0, 30.0)]))
    t.process_audio("garage", T0 + dt.timedelta(seconds=90), clip(30, [(0.0, 1.0)]))   # listener was down
    assert lines(cfg, "garage") == ["[12:00:28] at 0.2 for 2.0", "[12:01:30] at 0.0 for 1.0"]


def test_carry_is_per_camera(tmp_path):
    cfg, _ = make(tmp_path)
    t = tr.Transcriber(cfg, transcribe=FakeWhisper(), vad=fake_vad)
    t.process_audio("garage", T0, clip(30, [(28.0, 30.0)]))
    t.process_audio("doorbell", T0 + dt.timedelta(seconds=30), clip(30, [(0.0, 1.0)]))
    assert lines(cfg, "doorbell") == ["[12:00:30] at 0.0 for 1.0"] and "garage" in t.carry


def test_idle_carry_flushes_after_timeout_and_on_force(tmp_path):
    cfg, _ = make(tmp_path)
    clock = Clock()
    t = tr.Transcriber(cfg, transcribe=FakeWhisper(), vad=fake_vad, clock=clock)
    t.process_audio("garage", T0, clip(30, [(28.0, 30.0)]))
    clock.t += tr.FLUSH_AFTER_S - 1
    assert t.flush_idle() == 0 and lines(cfg, "garage") == []
    clock.t += 2
    assert t.flush_idle() == 1 and lines(cfg, "garage") == ["[12:00:28] at 0.2 for 2.0"]
    t.process_audio("doorbell", T0, clip(30, [(29.0, 30.0)]))
    assert t.flush_idle(force=True) == 1


def test_continuous_speech_longer_than_max_carry_is_not_held(tmp_path):
    cfg, _ = make(tmp_path)
    fw = FakeWhisper()
    t = tr.Transcriber(cfg, transcribe=fw, vad=fake_vad)
    t.process_audio("garage", T0, clip(30, [(1.0, 30.0)]))       # 29 s > MAX_CARRY_S
    assert not t.carry and fw.calls == [30.0]


def test_unreliable_segments_are_dropped(tmp_path):
    cfg, _ = make(tmp_path)
    t = tr.Transcriber(cfg, transcribe=FakeWhisper(no_speech_prob=0.95, avg_logprob=-1.4), vad=fake_vad)
    assert t.process_audio("garage", T0, clip(30, [(10.0, 11.0)])) == 0
    assert t.stats.dropped_segments == 1 and lines(cfg, "garage") == []


def test_poll_oldest_first_allowlist_bad_files_and_deletes(tmp_path, caplog):
    cfg, inbox = make(tmp_path, TRANSCRIBE_CAMERAS="garage")
    a = write_wav(inbox / name("01-garage", T0 + dt.timedelta(seconds=30)), clip(30, [(1.0, 2.0)]))
    b = write_wav(inbox / name("01-garage", T0), clip(30, [(3.0, 4.0)]))
    other = write_wav(inbox / name("00-doorbell", T0), clip(30, [(3.0, 4.0)]))
    bad = inbox / name("01-garage", T0 + dt.timedelta(seconds=60)); bad.write_bytes(b"RIFFjunk")
    keep_me = inbox / "notes.txt"; keep_me.write_text("x")
    fw = FakeWhisper()
    t = tr.Transcriber(cfg, transcribe=fw, vad=fake_vad)
    assert t.poll_once() == 4
    assert lines(cfg, "garage") == ["[12:00:03] at 3.0 for 1.0", "[12:00:31] at 1.0 for 1.0"]
    assert lines(cfg, "doorbell") == []
    assert sorted(os.listdir(inbox)) == ["notes.txt"]
    assert (t.stats.files, t.stats.skipped, t.stats.failed) == (2, 1, 1)
    assert any("failed" in r.getMessage() and r.levelno == logging.ERROR for r in caplog.records)
    assert (Path(cfg.data_dir) / "heartbeat").exists()             # beats per file during a backlog


def test_run_logs_counts_never_transcript_text(tmp_path, caplog):
    cfg, inbox = make(tmp_path)
    secret = "the alarm code is four four two one"
    write_wav(inbox / name("01-garage", T0), clip(30, [(3.0, 4.0)]))
    stop = threading.Event()
    calls = []

    def say(audio):
        calls.append(1)
        stop.set()                                   # one pass is enough
        return [tr.Segment(0.0, 1.0, secret)]

    with caplog.at_level(logging.INFO, logger="transcriber"):
        assert tr.run(cfg, transcribe=say, vad=fake_vad, stop=stop, summary_s=0) == 0
    assert calls and lines(cfg, "garage") == [f"[12:00:00] {secret}"]
    assert (Path(cfg.data_dir) / "heartbeat").exists()
    msgs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("transcriber: inbox") for m in msgs)
    assert not [m for m in msgs if "alarm" in m]


def test_run_missing_inbox_is_fatal(tmp_path, caplog):
    cfg = tr.load_config({"INBOX_DIR": str(tmp_path / "nope"), "DATA_DIR": str(tmp_path / "d")})
    assert tr.run(cfg, transcribe=FakeWhisper(), vad=fake_vad, stop=threading.Event()) == 1
    assert any("not found" in r.getMessage() for r in caplog.records)


def test_run_prunes_old_days_once_per_day(tmp_path, monkeypatch):
    import dataclasses
    cfg, _ = make(tmp_path, RETENTION_DAYS="2")
    cfg = dataclasses.replace(cfg, poll_s=0.01)
    old = Path(cfg.transcript_dir) / "2026-11-01"; old.mkdir(parents=True); (old / "garage.txt").write_text("x")
    stop = threading.Event()
    pruned = []
    real = tr.Writer.prune
    monkeypatch.setattr(tr.Writer, "prune", lambda self, *a: pruned.append(a) or real(self, *a))
    calls = [0]

    def today(_tz):
        calls[0] += 1
        if calls[0] >= 5:
            stop.set()
        return dt.date(2026, 11, 17) if calls[0] < 3 else dt.date(2026, 11, 18)

    assert tr.run(cfg, transcribe=FakeWhisper(), vad=fake_vad, stop=stop, today=today, summary_s=1e9) == 0
    assert not old.exists()
    assert [a[1] for a in pruned] == [dt.date(2026, 11, 17), dt.date(2026, 11, 18)]   # once per new day


@pytest.mark.skipif(not __import__("importlib").util.find_spec("faster_whisper"),
                    reason="needs faster-whisper and the model (CI runs it in the image)")
def test_selftest_real_model():
    assert tr.selftest() == []
