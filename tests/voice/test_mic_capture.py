"""The server's microphone and its gate: a table that stops hearing must not stay deaf, and a hold can be explained."""

from __future__ import annotations

import os
import signal
import stat
import time

import numpy as np

from lightwork.voice.capture import FRAME_BYTES, MicGate, ServerMic, hold_report


def fake_recorder(tmp_path, monkeypatch, body: str) -> None:
    """``pw-record`` on the PATH, a script: 4 Nov the real one died of a snapshot's USR2."""
    exe = tmp_path / "pw-record"
    exe.write_text(f"#!/bin/sh\n{body}\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")


def wait_until(check, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(0.02)
    return False


def test_a_snapshots_signal_does_not_kill_the_recorder(tmp_path, monkeypatch):
    fake_recorder(tmp_path, monkeypatch, "exec cat /dev/zero")
    frames = []
    mic = ServerMic(frames.append, "c920")
    assert mic.start() is None
    try:
        assert wait_until(lambda: frames)
        pid = mic.proc.pid
        os.kill(pid, signal.SIGUSR2)  # what `systemctl --user kill -s USR2 lightwork-play` does to every process of the unit
        os.kill(pid, signal.SIGUSR1)
        time.sleep(0.2)
        assert mic.proc.poll() is None and mic.proc.pid == pid and mic.restarts == 0
        frames.clear()
        assert wait_until(lambda: frames)
    finally:
        mic.close()
    assert mic.proc.poll() is not None  # close() still ends it


def test_a_recorder_that_ends_is_started_again(tmp_path, monkeypatch, capsys):
    fake_recorder(tmp_path, monkeypatch, f"head -c {FRAME_BYTES} /dev/zero; exit 3")
    frames = []
    mic = ServerMic(frames.append, "c920")
    assert mic.start() is None
    try:
        assert wait_until(lambda: mic.restarts >= 1 and len(frames) >= 2)
    finally:
        mic.close()
    assert all(len(f) == FRAME_BYTES for f in frames)
    assert "pw-record ended (code 3)" in capsys.readouterr().out
    assert "restarted" in mic.status()


def test_the_gate_hands_over_each_hold_and_drops_the_phones_voice_from_before_it():
    got, holds = [], []
    phone_quiet = [False]
    gate = MicGate(got.append, pre_roll=2, keep_pre_roll=lambda: phone_quiet[0], on_hold=lambda *h: holds.append(h))
    gate(b"phone")
    gate.set_open(True)
    assert got == []  # the moment before held the phone's voice
    gate(b"word")
    gate.set_open(False)
    assert got == [b"word"] and holds[0][0] == b"word" and holds[0][1] >= 0.0
    phone_quiet[0] = True
    gate(b"before")
    gate.set_open(True)
    gate.set_open(False)
    assert got[-1] == b"before" and holds[1][0] == b"before"


def test_a_transcript_is_of_words_said_into_the_mic_only_shortly_after_it_shut():
    gate = MicGate(lambda _: None)
    assert not gate.recently_open(8.0)  # never opened
    gate.set_open(True)
    assert gate.recently_open(8.0)
    gate.set_open(False)
    assert gate.recently_open(8.0) and not gate.recently_open(0.0)


def test_the_report_of_a_hold_tells_a_dead_mic_from_a_quiet_room():
    assert "NO AUDIO" in hold_report(b"", 3.2)
    tone = (2000 * np.sin(np.arange(FRAME_BYTES) / 8)).astype("<i2").tobytes()
    line = hold_report(tone, 1.0)
    assert "1.0 s open" in line and "dBFS" in line
