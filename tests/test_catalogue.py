"""`camber catalogue fetch` has to leave a catalogue the readers can open.

The UIUC archive is a zip and every reader takes the extracted tree, so a
fetch that stopped at the download left a new user one undocumented step from
a working install.
"""

import zipfile
from pathlib import Path

import pytest

from camber.catalogue.fetch import FetchError, fetch, unpack
from camber.catalogue.sources import SOURCES, Source


def _source(**kw) -> Source:
    return Source(key="demo", name="Demo data", url="https://example.invalid/demo.zip",
                  terms="test only", redistributable=False, **kw)


def _zip(path: Path, entries: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, text in entries.items():
            zf.writestr(name, text)
    return path


def test_the_uiuc_archive_unpacks_to_the_tree_the_readers_use():
    from camber import paths
    assert SOURCES["uiuc"].unpacks_to == paths.propdb().name


def test_an_archive_is_extracted_once(tmp_path):
    archive = _zip(tmp_path / "demo.zip", {"Demo/volume-1/a.txt": "1"})
    src = _source(unpacks_to="Demo")
    assert unpack(src, archive, tmp_path) is True
    assert (tmp_path / "Demo" / "volume-1" / "a.txt").read_text() == "1"
    assert unpack(src, archive, tmp_path) is False


def test_an_entry_outside_the_root_is_refused(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    archive = _zip(root / "demo.zip", {"Demo/ok.txt": "", "../escaped.txt": "x"})
    with pytest.raises(FetchError, match="outside"):
        unpack(_source(unpacks_to="Demo"), archive, root)
    assert not (tmp_path / "escaped.txt").exists()


def test_an_archive_missing_its_directory_is_an_error(tmp_path):
    archive = _zip(tmp_path / "demo.zip", {"Other/a.txt": ""})
    with pytest.raises(FetchError, match="not in the archive"):
        unpack(_source(unpacks_to="Demo"), archive, tmp_path)


def test_a_cached_download_is_extracted_without_fetching_again(tmp_path, monkeypatch):
    """The common case after an interrupted first run: the zip is there, the
    tree is not. No network is needed to finish."""
    monkeypatch.setitem(SOURCES, "demo", _source(unpacks_to="Demo"))
    _zip(tmp_path / "demo.zip", {"Demo/a.txt": ""})

    def no_network(*a, **k):
        raise AssertionError("fetched again")
    monkeypatch.setattr("urllib.request.urlopen", no_network)

    r = fetch("demo", tmp_path)
    assert r.cached and "extracted" in r.message
    assert (tmp_path / "Demo" / "a.txt").exists()
