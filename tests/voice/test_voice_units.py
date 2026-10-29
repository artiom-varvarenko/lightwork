from __future__ import annotations

import io
import math
import random
import wave

import numpy as np
import pytest

from lightwork.contracts.stepgraph import StepGraph
from lightwork.voice.grammar import COMMANDS, MAX_NOUNS, build_keywords, parse_command, stepgraph_nouns
from lightwork.voice.pcm import FRAME_BYTES, MAX_BACKLOG_BYTES, PcmRing, RmsVad, level_dbfs, wav_bytes
from lightwork.voice.policy import MAX_TRIES, NETWORK_MAX_DELAY_S, Action, Backoff, close_action, status_action
from lightwork.voice.service import realtime_url

SILENCE = bytes(FRAME_BYTES)
TONE = (8000 * np.sin(2 * math.pi * 440 * np.arange(FRAME_BYTES // 2) / 24_000)).astype("<i2").tobytes()


@pytest.mark.parametrize(
    "text, cmd",
    [("done", "done"), ("Done.", "done"), ("  NEXT!  ", "next"), ("Why?", "why"), ("repeat", "repeat"),
     ("Back.", "back")],
)
def test_grammar_accepts_exact_commands(text, cmd):
    assert parse_command(text) == cmd


@pytest.mark.parametrize(
    "text",
    ["I'm done with that", "next one is blue", "done done", "done..", "", "   ", "dun", "say done when finished",
     "Next, please", "go back", "why not", "done!?"],
)
def test_grammar_rejects_distractors(text):
    assert parse_command(text) is None


def test_keywords_commands_first_dedup_and_cap():
    nouns = ["Pipette", "pipette", "done", "", *[f"noun{i}" for i in range(30)]]
    kws = build_keywords(nouns)
    assert kws[: len(COMMANDS)] == list(COMMANDS)
    assert kws[len(COMMANDS)] == "Pipette"
    assert len(kws) == len(COMMANDS) + MAX_NOUNS
    assert build_keywords(kws) == kws  # idempotent: callers may pass nouns or a full keyword list


def test_stepgraph_nouns(synthetic_kit_run):
    graph = StepGraph.model_validate_json((synthetic_kit_run / "stepgraph.json").read_text())
    nouns = stepgraph_nouns(graph)
    assert nouns[: len(graph.vocabulary)] == graph.vocabulary
    assert {o.noun for o in graph.objects} <= set(nouns)


def test_ring_keeps_newest_half_second_and_pads():
    ring = PcmRing()
    frames = [bytes([i]) * FRAME_BYTES for i in range(1, 21)]  # 1.6 s
    for f in frames:
        ring.push(f)
    out = b"".join(ring.pop(FRAME_BYTES) for _ in range(7))  # 0.56 s requested, 0.5 s buffered
    assert out[MAX_BACKLOG_BYTES - FRAME_BYTES:MAX_BACKLOG_BYTES] == frames[-1]  # newest audio last...
    assert out[MAX_BACKLOG_BYTES:] == bytes(7 * FRAME_BYTES - MAX_BACKLOG_BYTES)  # ...then zero padding
    assert ring.pop(4) == bytes(4)


def test_ring_drops_odd_trailing_byte():
    ring = PcmRing()
    ring.push(b"\x01\x02\x03")
    assert len(ring) == 2


def _run_vad(vad, plan):
    out, t = [], 0.0
    for kind, seconds in plan:
        for _ in range(round(seconds / 0.08)):
            t += 0.08
            u = vad.push(TONE if kind == "tone" else SILENCE, t)
            if u is not None:
                out.append(u)
    return out


def test_vad_cuts_command_sized_utterances_only():
    vad = RmsVad()
    utts = _run_vad(vad, [("silence", 1.0), ("tone", 0.8), ("silence", 1.0), ("tone", 0.08), ("silence", 1.0),
                          ("tone", 5.0), ("silence", 1.0), ("tone", 0.4), ("silence", 1.0)])
    assert [round(u.t_end - u.t_start + 0.08, 2) for u in utts] == [0.8, 0.4]
    assert all(len(u.pcm) % FRAME_BYTES == 0 and len(u.pcm) >= 0.4 * 48_000 for u in utts)
    assert vad.last_end_t == pytest.approx(utts[-1].t_end)


def test_level_and_wav():
    assert level_dbfs(SILENCE) == -120.0
    assert -18 < level_dbfs(TONE) < -12
    with wave.open(io.BytesIO(wav_bytes(TONE))) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 24_000, 1920)


@pytest.mark.parametrize(
    "code, reason, acked, ours, repeat, kind",
    [
        (1000, "", True, True, False, "done"),
        (1000, "", True, False, False, "reconnect"),
        (1008, "invalid credentials", False, False, False, "off"),
        (1008, "audio backlog exceeded", True, False, False, "reconnect"),
        (1008, "audio backlog exceeded", True, False, True, "off"),
        (1011, "Max session duration reached", True, False, False, "reconnect"),
        (1011, "boom", True, False, False, "backoff"),
        (1013, "", True, False, False, "backoff"),
        (1006, "", True, False, False, "backoff"),
        (None, "", False, False, False, "backoff"),
    ],
)
def test_close_policy(code, reason, acked, ours, repeat, kind):
    assert close_action(code, reason, acked=acked, ended_by_us=ours, repeat_1008=repeat).kind == kind


def test_close_policy_backoff_bases():
    assert close_action(1013, "", acked=True, ended_by_us=False, repeat_1008=False).base_s == 2.0
    assert close_action(1011, "x", acked=True, ended_by_us=False, repeat_1008=False).base_s == 0.5
    assert not close_action(1006, "", acked=True, ended_by_us=False, repeat_1008=False).limited


@pytest.mark.parametrize("status, kind", [(401, "off"), (403, "fallback"), (404, "fallback"), (429, "backoff"),
                                          (503, "backoff"), (400, "off")])
def test_status_policy(status, kind):
    assert status_action(status).kind == kind


def test_backoff_schedule():
    b = Backoff(random.Random(0))
    limited = Action("backoff", "x", 0.5)
    delays = [b.next(limited) for _ in range(MAX_TRIES)]
    assert all(0.8 * 0.5 * 2**i <= d <= 1.2 * 0.5 * 2**i for i, d in enumerate(delays))
    assert b.next(limited) is None
    b.reset()
    net = Action("backoff", "net", 0.5, limited=False)
    delays = [b.next(net) for _ in range(10)]
    assert all(d is not None and d <= NETWORK_MAX_DELAY_S * 1.2 for d in delays)


def test_realtime_url():
    assert realtime_url("https://api.meta.ai/v1") == "wss://api.meta.ai/v1/asr/realtime"
    assert realtime_url("http://127.0.0.1:9/v1") == "ws://127.0.0.1:9/v1/asr/realtime"
