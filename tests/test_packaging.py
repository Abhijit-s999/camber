"""PKG-02: an installed Camber carries everything a checkout reads from disk.

In a checkout the app finds its templates, example pilots, fitted interference
model and OpenFOAM wrapper next to the package. A wheel has no "next to", so
each of those is copied inside it by pyproject's force-include. Adding a pilot
or a template without listing it there would ship an app that works on the
developer's machine and nowhere else; these tests make that a failure here.
"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
INCLUDED = PYPROJECT["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]


def _shipped(relative: str) -> None:
    assert relative in INCLUDED, f"{relative} is read at run time but not in the wheel"
    assert INCLUDED[relative] == f"camber/{relative}", (
        f"{relative} must land at camber/{relative}, where camber.paths looks")
    assert (ROOT / relative).exists()


def test_every_template_is_in_the_wheel():
    from camber.app.project import TEMPLATES
    for file in TEMPLATES.values():
        _shipped(f"examples/{file}")


def test_every_example_pilot_is_in_the_wheel():
    pilots = sorted((ROOT / "examples" / "pilots").glob("*.py"))
    assert pilots
    for p in pilots:
        _shipped(p.relative_to(ROOT).as_posix())


def test_the_shipped_data_and_the_openfoam_wrapper_are_in_the_wheel():
    for f in sorted((ROOT / "data").glob("*.json")):
        _shipped(f.relative_to(ROOT).as_posix())
    _shipped("scripts/foam-env.sh")


def test_nothing_from_the_bench_is_installed():
    """A `bench` launcher would point at a module the wheel leaves out (AI-04)."""
    for name, target in PYPROJECT["project"].get("scripts", {}).items():
        assert not target.startswith("bench"), f"script {name!r} -> {target}"
    assert not any(src.startswith("bench") for src in INCLUDED)


def test_the_version_has_one_source():
    import camber
    assert PYPROJECT["tool"]["hatch"]["version"]["path"] == "camber/__init__.py"
    assert "version" in PYPROJECT["project"]["dynamic"]
    assert camber.__version__.count(".") == 2


def test_the_source_distribution_is_an_explicit_list():
    """PKG-05: hatch's default sdist is the tree minus .gitignore, so it ships
    whatever is present but untracked -- notes kept out of git only through
    .git/info/exclude, and the bench (AI-04). The list is explicit, and it holds
    everything the wheel is built from."""
    sdist = PYPROJECT["tool"]["hatch"]["build"]["targets"]["sdist"]["only-include"]
    assert not any(p in ("", ".", "docs") or p.startswith("bench") for p in sdist)
    for src in INCLUDED:
        assert any(src == p or src.startswith(p + "/") for p in sdist), (
            f"{src} goes into the wheel but is not in the sdist it is built from")
