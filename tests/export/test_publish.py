"""Exercise the shell release gates without GitHub credentials or any real push."""
import os
import subprocess
from pathlib import Path

import pytest

from lightwork.runs.export import export_site

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "publish-viewer.sh"


@pytest.mark.usefixtures("no_network")
@pytest.mark.parametrize("visibility", ["public", "refused"])
def test_dry_run_rejects_synthetic_or_private_target(tmp_path, synthetic_kit_run, visibility):
    site = tmp_path / "site"
    export_site(synthetic_kit_run, site, allow_synthetic=True)
    tools = tmp_path / "bin"
    tools.mkdir()
    # GitHub visibility is the external boundary; the bundle validator and shell gate are real.
    gh = tools / "gh"
    gh.write_text(f"#!/bin/sh\nprintf '%s\\n' '{visibility}'\n")
    gh.chmod(0o755)
    git = tools / "git"
    marker = tmp_path / "git-invoked"
    git.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit 99\n")
    git.chmod(0o755)
    env = {**os.environ, "PATH": str(tools) + os.pathsep + os.environ["PATH"], "UV_OFFLINE": "1"}
    result = subprocess.run(["bash", str(SCRIPT), "--repo", "test/public", "--site", str(site), "--dry-run"], env=env, capture_output=True, text=True)
    assert result.returncode == 1
    assert ("not publishable" if visibility == "public" else "not public") in result.stderr
    assert not marker.exists()


def test_publish_requires_explicit_mode():
    result = subprocess.run(["bash", str(SCRIPT), "--repo", "test/public"], capture_output=True, text=True)
    assert result.returncode == 2
