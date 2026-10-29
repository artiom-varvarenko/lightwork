"""VoiceService against the local fake realtime server (web:V1, web:V2, grammar, mute, rotation, close codes, D14)."""

from __future__ import annotations

import json
import time

import httpx

from lightwork.contracts.render import Tts
from lightwork.voice.grammar import COMMANDS


def _live(svc, kit, timeout=5.0):
    assert kit.wait_until(lambda: svc.hud["voice"] == "live", timeout), svc.hud


def _speak(kit, svc, seconds=0.4, tail=0.6):
    kit.Feeder(svc, [("tone", seconds), ("silence", tail)]).start().join()


def test_v1_handshake_and_real_time_ingress(fake_upstream, voice_service, voice_kit):
    fake, base = fake_upstream(below_rt_limit_s=1.0, backlog_limit_s=1.0)
    svc, events, _ = voice_service(base, keywords=["pipette", "well plate"])
    _live(svc, voice_kit)
    feeder = voice_kit.Feeder(svc, [("silence", 0.5), ("tone", 0.4), ("silence", 1.6)]).start()
    feeder.join()
    sess = fake.sessions[0]
    hs = sess.handshake
    assert hs["authorization"] == {"accessToken": "Bearer LLM_TEST"}
    assert (hs["audioEncoding"], hs["mode"], hs["model"]) == ("PCM_24KHZ", "ENDPOINTING", "muse-voice-transcribe-1.0")
    assert hs["keywords"] == [*COMMANDS, "pipette", "well plate"]
    assert hs["languageBias"] == ["English"] and hs["partialMode"] == "CUMULATIVE" and hs["emitAudioProgress"] is False
    frames = sess.frames
    assert {n for _, n in frames} == {voice_kit.FRAME_BYTES}
    rate = (len(frames) - 1) * voice_kit.FRAME_BYTES / (frames[-1][0] - frames[0][0])
    assert abs(rate - 48_000) / 48_000 < 0.03, rate
    assert sess.close_code is None and fake.open_sessions() == [sess]
    stats = svc.stats()
    assert stats["bytes_in"] == round(2.5 / 0.08) * voice_kit.FRAME_BYTES
    assert stats["closes"] == []


def test_v2_mic_gap_is_padded_and_voice_resumes(fake_upstream, voice_service, voice_kit):
    fake, base = fake_upstream(below_rt_limit_s=1.0)
    svc, events, _ = voice_service(base)
    _live(svc, voice_kit)
    fake.say("Next.")
    voice_kit.Feeder(svc, [("silence", 0.3), ("gap", 2.5), ("tone", 0.4), ("silence", 0.6)]).start().join()
    assert voice_kit.wait_until(lambda: events, 2.0)
    assert events == [("voice", {"command": "next", "transcript": "Next.", "session_id": fake.sessions[0].id,
                                 "turn_id": "1"})]
    assert len(fake.sessions) == 1 and fake.sessions[0].close_code is None


def test_browser_burst_is_trimmed_not_forwarded(fake_upstream, voice_service, voice_kit):
    fake, base = fake_upstream(backlog_limit_s=0.6)
    svc, _, _ = voice_service(base)
    _live(svc, voice_kit)
    voice_kit.Feeder(svc, [("burst", 3.0), ("silence", 1.0)]).start().join()
    sess = fake.sessions[0]
    assert sess.close_code is None
    assert svc.stats()["dropped_backlog_bytes"] >= 2.4 * 48_000
    elapsed = time.monotonic() - sess.frames[0][0]
    assert sess.bytes / 48_000 <= elapsed + 0.2


def test_grammar_end_to_end_and_duplicates(fake_upstream, voice_service, voice_kit):
    fake, base = fake_upstream(duplicate_complete=True)
    svc, events, huds = voice_service(base)
    _live(svc, voice_kit)
    fake.say("done", "I'm done with that", "Repeat.", "next one is blue")
    for _ in range(4):
        _speak(voice_kit, svc)
    assert voice_kit.wait_until(lambda: len(events) == 4, 2.0), events
    sid = fake.sessions[0].id
    assert events == [
        ("voice", {"command": "done", "transcript": "done", "session_id": sid, "turn_id": "1"}),
        ("voice_rejected", {"transcript": "I'm done with that"}),
        ("voice", {"command": "repeat", "transcript": "Repeat.", "session_id": sid, "turn_id": "3"}),
        ("voice_rejected", {"transcript": "next one is blue"}),
    ]
    assert any(h["turn"] == "speech" for h in huds) and huds[-1]["heard"] == "next one is blue"
    assert huds[-1]["accepted"] is False


def test_dry_run_reaches_hud_not_planner(fake_upstream, voice_service, voice_kit):
    fake, base = fake_upstream()
    svc, events, huds = voice_service(base, dry_run=True)
    _live(svc, voice_kit)
    fake.say("Done.")
    _speak(voice_kit, svc)
    assert voice_kit.wait_until(lambda: svc.hud["command"] == "done", 2.0)
    assert svc.hud["dry_run"] is True and svc.hud["accepted"] is True
    assert events == []


def test_tts_mute_sends_silence_then_tail_unmutes(fake_upstream, voice_service, voice_kit):
    fake, base = fake_upstream()
    svc, events, _ = voice_service(base)
    _live(svc, voice_kit)
    fake.say("done")
    svc.on_tts(Tts(id="s1", state="start"))
    assert svc.hud["muted"] is True
    _speak(voice_kit, svc)  # the speaker saying "say done when finished"
    assert fake.sessions[0].turns == 0 and events == []
    svc.on_tts(Tts(id="s1", state="end"))
    _speak(voice_kit, svc, seconds=0.16, tail=0.0)  # inside the 300 ms tail: still muted
    voice_kit.Feeder(svc, [("silence", 0.4)]).start().join()
    assert fake.sessions[0].turns == 0
    _speak(voice_kit, svc)
    assert voice_kit.wait_until(lambda: events, 2.0)
    assert events[0][1]["command"] == "done"
    assert svc.stats()["bytes_muted"] > 0 and svc.hud["muted"] is False


def test_set_muted_flag(fake_upstream, voice_service, voice_kit):
    fake, base = fake_upstream()
    svc, events, _ = voice_service(base)
    _live(svc, voice_kit)
    svc.set_muted(True)
    _speak(voice_kit, svc)
    assert fake.sessions[0].turns == 0
    svc.set_muted(False)
    time.sleep(0.35)
    fake.say("why")
    _speak(voice_kit, svc)
    assert voice_kit.wait_until(lambda: events, 2.0) and events[0][1]["command"] == "why"


def test_rotation_opens_new_session_then_ends_old(fake_upstream, voice_service, voice_kit):
    fake, base = fake_upstream(below_rt_limit_s=1.0)
    svc, events, _ = voice_service(base, rotate_after_s=0.8)
    _live(svc, voice_kit)
    voice_kit.Feeder(svc, [("silence", 2.0)]).start().join()
    assert len(fake.sessions) >= 2
    first = fake.sessions[0]
    assert first.ended_by_client and first.close_code == 1000
    assert all(c in (1000, None) for c in (s.close_code for s in fake.sessions))
    fake.say("back")
    _speak(voice_kit, svc)
    assert voice_kit.wait_until(lambda: events, 2.0)
    live_ids = {s.id for s in fake.sessions if s.close_code is None or s.turns}
    assert events[0][1]["command"] == "back" and events[0][1]["session_id"] in live_ids


def test_set_keywords_rotates_and_stales_open_turns(fake_upstream, voice_service, voice_kit):
    fake, base = fake_upstream()
    svc, events, _ = voice_service(base)
    _live(svc, voice_kit)
    first = fake.sessions[0]
    svc.set_keywords(["breadboard"])
    assert voice_kit.wait_until(lambda: len(fake.sessions) == 2 and fake.sessions[1].handshake is not None)
    assert fake.sessions[1].handshake["keywords"] == [*COMMANDS, "breadboard"]
    assert voice_kit.wait_until(lambda: first.close_code == 1000) and first.ended_by_client
    assert svc.keywords == [*COMMANDS, "breadboard"]


def test_turn_open_across_step_change_is_rejected(fake_upstream, voice_service, voice_kit):
    fake, base = fake_upstream()
    svc, events, _ = voice_service(base)
    _live(svc, voice_kit)
    fake.say("done")
    feeder = voice_kit.Feeder(svc, [("tone", 0.6), ("silence", 0.6)]).start()
    assert voice_kit.wait_until(lambda: svc.hud["turn"] == "speech", 2.0)
    svc.step_changed()
    feeder.join()
    assert voice_kit.wait_until(lambda: events, 2.0)
    assert events == [("voice_rejected", {"transcript": "done"})]


def test_handshake_rejected_turns_voice_off(fake_upstream, voice_service, voice_kit):
    fake, base = fake_upstream(key="LLM_OTHER")
    svc, _, _ = voice_service(base)
    assert voice_kit.wait_until(lambda: svc.hud["voice"] == "error")
    assert "invalid credentials" in svc.hud["error"]
    time.sleep(0.8)
    assert len(fake.sessions) == 1  # 1008 in the handshake is never retried


def test_max_session_duration_reconnects(fake_upstream, voice_service, voice_kit):
    fake, base = fake_upstream(max_session_s=0.5)
    svc, _, _ = voice_service(base)
    feeder = voice_kit.Feeder(svc, [("silence", 2.0)]).start()
    assert voice_kit.wait_until(lambda: len([s for s in fake.sessions if s.handshake]) >= 2, 3.0)
    assert fake.sessions[0].close_code == 1011
    feeder.join()
    assert svc.hud["voice"] in ("live", "reconnecting", "connecting")


def test_unsolicited_1000_reconnects(fake_upstream, voice_service, voice_kit):
    fake, base = fake_upstream()
    svc, _, _ = voice_service(base)
    _live(svc, voice_kit)
    fake.close_active(1000)
    assert voice_kit.wait_until(lambda: len(fake.open_sessions()) == 1 and len(fake.sessions) == 2, 3.0)
    _live(svc, voice_kit)


def test_second_midstream_1008_turns_voice_off(fake_upstream, voice_service, voice_kit):
    fake, base = fake_upstream()
    svc, _, _ = voice_service(base)
    _live(svc, voice_kit)
    fake.close_active(1008, "audio backlog exceeded")
    assert voice_kit.wait_until(lambda: len(fake.open_sessions()) == 1 and len(fake.sessions) == 2, 3.0)
    _live(svc, voice_kit)
    fake.close_active(1008, "audio backlog exceeded")
    assert voice_kit.wait_until(lambda: svc.hud["voice"] == "error", 3.0)
    assert "1008" in svc.hud["error"]
    assert [c for c, _ in svc.stats()["closes"]] == [1008, 1008]


def test_no_key_stays_off(voice_service, voice_kit, no_network):
    svc, _, _ = voice_service("http://127.0.0.1:9/v1", key=None)
    assert svc.hud["voice"] == "off" and "MODEL_API_KEY" in svc.hud["error"]
    svc.feed_pcm(voice_kit.TONE)  # still a harmless sink


def test_d14_file_fallback_when_realtime_denied(fake_upstream, voice_service, voice_kit):
    fake, base = fake_upstream(reject_status=403)
    posts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posts.append(request)
        return httpx.Response(200, json={"sessionId": "file-1", "transcript": "Repeat.", "audioDurationMs": 900,
                                         "turns": []})

    svc, events, _ = voice_service(base, http_transport=httpx.MockTransport(handler))
    assert voice_kit.wait_until(lambda: svc.hud["voice"] == "fallback")
    assert svc.mode == "file"
    voice_kit.Feeder(svc, [("silence", 0.5), ("tone", 0.5), ("silence", 0.8)]).start().join()
    assert voice_kit.wait_until(lambda: events, 2.0)
    assert events == [("voice", {"command": "repeat", "transcript": "Repeat.", "session_id": "file-1",
                                 "turn_id": "1"})]
    req = posts[0]
    assert req.url.path == "/v1/asr/transcribe" and req.headers["authorization"] == "Bearer LLM_TEST"
    body = req.read()
    marker = b'name="request"'
    request_json = json.loads(body.split(marker, 1)[1].split(b"\r\n\r\n", 1)[1].split(b"\r\n--", 1)[0])
    assert request_json["mode"] == "PUSH_TO_TALK" and request_json["audioEncoding"] == "WAV"
    assert request_json["keywords"][:5] == list(COMMANDS)
    assert b"RIFF" in body and b"WAVE" in body


def test_forced_file_mode_skips_realtime(fake_upstream, voice_service, voice_kit):
    fake, base = fake_upstream()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"sessionId": "f", "transcript": "I'm done with that"})

    svc, events, _ = voice_service(base, mode="file", http_transport=httpx.MockTransport(handler))
    voice_kit.Feeder(svc, [("tone", 0.5), ("silence", 0.8)]).start().join()
    assert voice_kit.wait_until(lambda: events, 2.0)
    assert events == [("voice_rejected", {"transcript": "I'm done with that"})]
    assert fake.connections == 0
