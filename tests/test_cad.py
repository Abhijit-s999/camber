"""CAD export. Skipped cleanly when build123d is absent -- it is optional."""

from pathlib import Path

import pytest

from camber.spec import io

build123d = pytest.importorskip("build123d", reason="CAD export is optional")

from camber.geometry import cad  # noqa: E402

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "quad5.toml"


@pytest.fixture
def vehicle():
    return io.load(EXAMPLE)


def test_bounding_box_is_in_millimetres(vehicle):
    """Camber is metres; STEP is millimetres. The 180 mm chassis proves it."""
    x, y, z = cad.bounding_box_mm(vehicle)
    assert x == pytest.approx(180.0, abs=0.1)
    assert y == pytest.approx(80.0, abs=0.1)
    assert 40.0 < z < 60.0


def test_step_export(vehicle, tmp_path):
    out = cad.export(vehicle, tmp_path / "v.step")
    assert out.exists() and out.stat().st_size > 1000
    head = out.read_text(errors="ignore")[:2000]
    assert "ISO-10303" in head
    assert "MILLI" in head.upper(), "STEP must declare millimetres"


def test_stl_export(vehicle, tmp_path):
    out = cad.export(vehicle, tmp_path / "v.stl")
    assert out.exists() and out.stat().st_size > 100


def test_unsupported_format_is_rejected(vehicle, tmp_path):
    with pytest.raises(ValueError, match="unsupported export format"):
        cad.export(vehicle, tmp_path / "v.dwg")


def test_geometryless_vehicle_is_rejected(tmp_path):
    v = io.from_dict({
        "name": "ghost",
        "bodies": [{"name": "b", "type": "body", "mass": 1.0,
                    "shape": {"kind": "point"}}],
    })
    with pytest.raises(ValueError, match="no bodies with extent"):
        cad.build(v)


def test_rotation_reaches_the_solid(vehicle, tmp_path):
    """A tilted camera must actually be tilted in the exported geometry."""
    plain = io.from_dict({
        "name": "t",
        "bodies": [{"name": "b", "type": "body", "mass": 1.0,
                    "shape": {"kind": "box", "lx": 0.2, "ly": 0.02, "lz": 0.02}}],
    })
    turned = io.from_dict({
        "name": "t",
        "bodies": [{"name": "b", "type": "body", "mass": 1.0,
                    "rotation_deg": [0.0, 0.0, 90.0],
                    "shape": {"kind": "box", "lx": 0.2, "ly": 0.02, "lz": 0.02}}],
    })
    px, py, _ = cad.bounding_box_mm(plain)
    tx, ty, _ = cad.bounding_box_mm(turned)
    assert px == pytest.approx(ty, abs=0.01)
    assert py == pytest.approx(tx, abs=0.01)
