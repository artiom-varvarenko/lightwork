from __future__ import annotations

import socket

import pytest

from lightwork.cli import discover, main
from lightwork.config import load_settings, parse_dotenv
from lightwork.testing.synth import well_plate
from lightwork.vision.source import ImageSequenceSource


def test_no_network_blocks_outbound_allows_loopback(no_network):
    with pytest.raises(RuntimeError, match="blocked by no_network"):
        socket.create_connection(("93.184.216.34", 443), timeout=1)
    with pytest.raises(RuntimeError, match="blocked by no_network"):
        socket.getaddrinfo("api.meta.ai", 443)
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    c = socket.create_connection(srv.getsockname(), timeout=1)
    c.close()
    srv.close()


def test_dotenv_and_env_precedence(tmp_path):
    (tmp_path / ".env").write_text('MODEL_API_KEY="LLM_1_x"\nLW_BUDGET_USD=50 # cap\n# c\nexport LW_DEV_ALLOW_NON_META=1\n')
    assert parse_dotenv("A='b c'\n")["A"] == "b c"
    s = load_settings(tmp_path, environ={"LW_BUDGET_USD": "10"})
    assert s.model_api_key == "LLM_1_x"
    assert s.budget_usd == 10.0
    assert s.dev_allow_non_meta is True
    assert "LLM_1_x" not in repr(s)


def test_cli_discovers_commands(capsys):
    names = {m.COMMAND for m in discover()}
    assert "doctor" in names
    assert main([]) == 2


def test_synth_plate_and_sequence_source():
    a, geom = well_plate({"A1": 1.0, "A2": 0.5})
    b, _ = well_plate({"A1": 1.0, "A2": 0.5})
    assert (a == b).all()
    x, y = geom.corners[0]
    assert a[int(y), int(x), 2] < a[int(y), int(x) + 60, 2] < 230  # A1 darker than A2 on red
    src = ImageSequenceSource([a, b], fps=10)
    frames = list(src)
    assert [f.idx for f in frames] == [0, 1] and frames[1].t_mono == pytest.approx(0.1)
