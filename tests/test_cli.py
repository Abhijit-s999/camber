"""The commands in the README work from an installed wheel, not only a checkout.

The README and docs/FLYING.md say `camber inspect examples/quad5.toml`. That
path exists in a checkout and nowhere else: someone who installed the wheel
has no `examples/` folder, and used to get a Python traceback. The wheel
carries the examples inside the package (PKG-02), so the CLI finds them there.
"""

import shutil
from pathlib import Path

import pytest

from camber.cli import _example, _load_pilot, main

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def elsewhere(tmp_path, monkeypatch):
    """A working directory with no examples/ folder in it, like a wheel user's."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_a_readme_path_finds_the_bundled_example(elsewhere, capsys):
    assert main(["inspect", "examples/quad5.toml"]) == 0
    assert "bundled example" in capsys.readouterr().err


def test_an_example_can_be_named_without_its_folder_or_suffix(elsewhere):
    assert _example("quad5").name == "quad5.toml"
    assert _example("quad5.toml").is_file()


def test_a_file_that_is_really_there_wins(elsewhere):
    mine = elsewhere / "examples" / "quad5.toml"
    mine.parent.mkdir()
    shutil.copy(ROOT / "examples" / "quad5.toml", mine)
    assert _example("examples/quad5.toml") == Path("examples/quad5.toml")


def test_the_example_pilots_are_found_the_same_way(elsewhere):
    pytest.importorskip("mujoco")
    assert _load_pilot("examples/pilots/flip.py", None) is not None


def test_a_missing_file_is_an_error_message_not_a_traceback(elsewhere, capsys):
    assert main(["inspect", "no-such-vehicle.toml"]) == 2
    assert "no such file" in capsys.readouterr().err.lower()


def test_no_feasible_design_is_an_error_not_a_front(elsewhere, capsys):
    """An all-infeasible front is the least-bad design, not an answer: it must
    not be printed as one, or written for `generate` to grow a frame on."""
    spec = (ROOT / "examples" / "quad5.toml").read_text()
    assert "distance = 0.040" in spec
    (elsewhere / "impossible.toml").write_text(spec.replace("distance = 0.040",
                                                            "distance = 5.0"))
    assert main(["optimise", "impossible.toml", "--population", "8",
                 "--generations", "2", "--out", "best.json"]) == 1
    assert "no feasible design" in capsys.readouterr().err
    assert not (elsewhere / "best.json").exists()


def test_aero_without_openfoam_says_so_before_writing_anything(elsewhere, capsys,
                                                               monkeypatch):
    """OpenFOAM is optional. Without it, `camber aero` used to write a case
    folder and then die inside blockMesh with a traceback."""
    monkeypatch.setenv("FOAM_INSTALL", str(elsewhere / "no-openfoam-here"))
    assert main(["aero", "examples/racer5.toml", "--out", "aero"]) == 3
    assert "needs OpenFOAM" in capsys.readouterr().err
    assert not (elsewhere / "aero").exists()


def test_step_of_a_grown_frame_is_refused_in_a_sentence(elsewhere, capsys):
    """A grown frame is a mesh and STEP carries solids; `cad` said so through
    a traceback."""
    assert main(["cad", str(ROOT / "examples" / "racer5_framed.json"), "out.step"]) == 2
    assert "export it as .stl" in capsys.readouterr().err
    assert not (elsewhere / "out.step").exists()
