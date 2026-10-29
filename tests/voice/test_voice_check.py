from __future__ import annotations

from lightwork.voice.check import MicProbe, score


def _ev(pos, cmd=None):
    return (pos, "voice", {"command": cmd}) if cmd else (pos, "voice_rejected", {"transcript": "x"})


def test_score_counts_v3_outcomes():
    expect = [(1.0, "done"), (4.0, "next"), (7.0, "-"), (10.0, "why"), (13.0, "back"), (16.0, "-")]
    events = [_ev(1.9, "done"), _ev(4.8, "back"), _ev(7.9), _ev(10.9, "why"), _ev(11.5, "why"), _ev(16.8, "done")]
    r = score(expect, events)
    assert (r["commands"], r["correct"], r["wrong"], r["distractors"], r["distractors_accepted"]) == (4, 1, 2, 2, 1)
    assert r["pass"] is False
    clean = score(expect[:3], [_ev(1.5, "done"), _ev(4.5, "next"), _ev(7.5)])
    assert clean["correct"] == 2 and clean["pass"] is True


def test_mic_probe_rate_and_wav_buffer():
    probe = MicProbe(keep_s=0.1)
    probe.feed_pcm(b"\0" * 3840)
    probe.t0 -= 1.0  # pretend the first frame arrived 1 s ago
    probe.feed_pcm(b"\1" * 48_000)
    assert 47_000 < probe.rate() < 48_100
    assert len(probe.pcm) == 4800
