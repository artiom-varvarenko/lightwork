from pathlib import Path
import json
import shutil
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare-public.sh'
# The release script stays in the private working repo; the public copy is its output.
pytestmark = pytest.mark.skipif(not SCRIPT.exists(), reason='release script is not part of the public copy')


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True, text=True)


@pytest.fixture
def release_repo(tmp_path):
    root = tmp_path / 'source'
    (root / 'scripts').mkdir(parents=True)
    shutil.copyfile(SCRIPT, root / 'scripts' / SCRIPT.name)
    git(root, 'init', '-q')
    git(root, 'config', 'user.name', 'Release test')
    git(root, 'config', 'user.email', 'release-test@example.invalid')
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'Test repository')
    return root


def commit_manifest(root, provenance):
    path = root / 'runs' / 'example' / 'manifest.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'provenance': provenance}))
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'Test manifest')


def prepare(root):
    out = root.parent / 'public'
    result = subprocess.run(['bash', str(root / 'scripts' / SCRIPT.name), str(out)],
                            text=True, capture_output=True, timeout=30)
    return result, out


@pytest.mark.parametrize('provenance', ['dev-standin', 'synthetic', 'unknown'])
def test_non_meta_committed_run_never_creates_public_copy(release_repo, provenance):
    commit_manifest(release_repo, provenance)
    result, out = prepare(release_repo)
    assert result.returncode == 1
    assert not out.exists()


def test_historical_standin_is_not_hidden_by_relabeling(release_repo):
    commit_manifest(release_repo, 'dev-standin')
    commit_manifest(release_repo, 'meta')
    result, out = prepare(release_repo)
    assert result.returncode == 1
    assert not out.exists()


def test_committed_development_tree_is_refused(release_repo):
    path = release_repo / 'runs-dev' / 'private.json'
    path.parent.mkdir()
    path.write_text('{}')
    git(release_repo, 'add', '.')
    git(release_repo, 'commit', '-qm', 'Development-only recording')
    result, out = prepare(release_repo)
    assert result.returncode == 1
    assert not out.exists()


def test_empty_evidence_is_not_a_publishable_demo(release_repo):
    result, out = prepare(release_repo)
    assert result.returncode == 1
    assert not out.exists()
