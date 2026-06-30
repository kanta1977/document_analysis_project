import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIXTURE_ZIP = Path(__file__).parent / "fixtures" / "mini_corpus.zip"


@pytest.fixture(scope="session")
def fixture_zip() -> Path:
    assert FIXTURE_ZIP.exists(), "run: python tests/fixtures/make_fixture.py"
    return FIXTURE_ZIP


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT
