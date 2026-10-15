"""LIC-01 as an executable guard.

None of Camber's data sources grants redistribution rights, so none of their
data may enter the repository. That is a licence obligation, and a
licence obligation that depends on someone remembering is one that will
eventually be broken by a stray `git add -A`.
"""

import subprocess
from pathlib import Path

import pytest

from camber.catalogue.sources import SOURCES, notice, redistributable

ROOT = Path(__file__).resolve().parent.parent


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files"],
                         capture_output=True, text=True, check=False)
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


class TestSources:
    def test_every_source_states_its_terms(self):
        for s in SOURCES.values():
            assert s.terms.strip(), f"{s.key} has no recorded terms"
            assert s.citation.strip(), f"{s.key} has no citation"

    def test_none_is_currently_redistributable(self):
        """If this ever changes, it must be a deliberate, checked decision."""
        assert redistributable() == [], (
            "a source is marked redistributable; confirm its licence actually "
            "says so before shipping any of its data")

    def test_notice_names_every_source(self):
        text = notice()
        for s in SOURCES.values():
            assert s.name in text and s.url in text

    def test_notice_file_is_present_and_current(self):
        f = ROOT / "NOTICE"
        assert f.exists(), "NOTICE must ship with the code"
        assert f.read_text().strip() == notice().strip(), (
            "NOTICE is stale; regenerate it from camber.catalogue.sources")


class TestNoSourceDataInTheTree:
    def test_raw_data_directory_is_not_tracked(self):
        offenders = [f for f in tracked_files() if f.startswith("data/raw")]
        assert not offenders, f"source data is tracked by git: {offenders[:5]}"

    def test_no_uiuc_archive_or_extract_is_tracked(self):
        bad = [f for f in tracked_files()
               if "propDB" in f or "UIUC-prop" in f]
        assert not bad, f"UIUC data in the tree: {bad[:5]}"

    def test_no_large_data_blobs_are_tracked(self):
        """A redistributable-looking blob is the shape the mistake takes."""
        big = []
        for f in tracked_files():
            p = ROOT / f
            if p.is_file() and p.suffix in {".zip", ".csv", ".dat", ".txt", ".eng", ".rse"}:
                if p.stat().st_size > 200_000:
                    big.append((f, p.stat().st_size))
        assert not big, f"large data files tracked: {big[:3]}"

    def test_shipped_json_is_derived_not_measured(self):
        """interference.json holds design descriptors from our own CFD runs,
        not third-party measurements."""
        import json
        p = ROOT / "data" / "interference.json"
        if not p.exists():
            pytest.skip("no fitted model present")
        d = json.loads(p.read_text())
        assert set(d) >= {"x", "y", "length_scale"}
        assert len(d["x"][0]) == 4, "features are the 4 derived descriptors"
