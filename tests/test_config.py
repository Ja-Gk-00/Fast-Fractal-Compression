import importlib
from pathlib import Path

import pytest


def _import_any(names, err):
    for n in names:
        try:
            return importlib.import_module(n)
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError(err)


def config_m():
    return _import_any(
        ("fastfractal.io.config", "fastfractal.config", "io.config", "config"),
        "config module not found",
    )


def _yaml_available():
    try:
        import yaml  # noqa: F401

        return True
    except Exception:
        return False


def test_load_yaml_requires_pyyaml_or_raises(tmp_path: Path):
    m = config_m()
    p = tmp_path / "x.yml"
    p.write_text("a: 1\n", encoding="utf-8")
    if _yaml_available():
        cfg = m.load_yaml(p)
        assert cfg["a"] == 1
    else:
        with pytest.raises(ValueError):
            m.load_yaml(p)


def test_load_yaml_empty_returns_empty_dict(tmp_path: Path):
    m = config_m()
    if not _yaml_available():
        pytest.skip("PyYAML not available")
    p = tmp_path / "empty.yml"
    p.write_text("", encoding="utf-8")
    assert m.load_yaml(p) == {}


def test_load_yaml_root_must_be_mapping(tmp_path: Path):
    m = config_m()
    if not _yaml_available():
        pytest.skip("PyYAML not available")
    p = tmp_path / "list.yml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        m.load_yaml(p)


def test_get_section_missing_returns_empty():
    m = config_m()
    assert m.get_section({}, "missing") == {}


def test_get_section_valid_mapping():
    m = config_m()
    cfg = {"a": {"x": 1}}
    assert m.get_section(cfg, "a") == {"x": 1}


def test_get_section_non_mapping_raises():
    m = config_m()
    cfg = {"a": 123}
    with pytest.raises(ValueError):
        m.get_section(cfg, "a")


def test_pick_int_default_and_alt_key():
    m = config_m()
    assert m.pick_int({}, "k", None, 7) == 7
    assert m.pick_int({"alt": 5}, "k", "alt", 7) == 5


def test_pick_int_accepts_int_and_integral_float():
    m = config_m()
    assert m.pick_int({"k": 3}, "k", None, 0) == 3
    assert m.pick_int({"k": 3.0}, "k", None, 0) == 3


def test_pick_int_rejects_bool_and_non_integral_float():
    m = config_m()
    with pytest.raises(ValueError):
        m.pick_int({"k": True}, "k", None, 0)
    with pytest.raises(ValueError):
        m.pick_int({"k": 3.5}, "k", None, 0)


def test_pick_opt_int_accepts_int_and_integral_float_and_alt_key():
    m = config_m()
    assert m.pick_opt_int({"k": 4}, "k", None, None) == 4
    assert m.pick_opt_int({"k": 4.0}, "k", None, None) == 4
    assert m.pick_opt_int({"alt": 9}, "k", "alt", None) == 9


def test_pick_opt_int_rejects_bool_and_non_integral_float():
    m = config_m()
    with pytest.raises(ValueError):
        m.pick_opt_int({"k": False}, "k", None, None)
    with pytest.raises(ValueError):
        m.pick_opt_int({"k": 2.2}, "k", None, None)


def test_pick_float_default_and_numeric_and_alt_key():
    m = config_m()
    assert m.pick_float({}, "k", None, 1.5) == 1.5
    assert m.pick_float({"k": 2}, "k", None, 0.0) == 2.0
    assert m.pick_float({"k": 2.25}, "k", None, 0.0) == 2.25
    assert m.pick_float({"alt": 9}, "k", "alt", 0.0) == 9.0


def test_pick_float_rejects_bool_and_non_numeric():
    m = config_m()
    with pytest.raises(ValueError):
        m.pick_float({"k": True}, "k", None, 0.0)
    with pytest.raises(ValueError):
        m.pick_float({"k": "x"}, "k", None, 0.0)


def test_pick_bool_default_and_accepts_bool_and_0_1_and_alt_key():
    m = config_m()
    assert m.pick_bool({}, "k", None, True) is True
    assert m.pick_bool({"k": False}, "k", None, True) is False
    assert m.pick_bool({"k": 0}, "k", None, True) is False
    assert m.pick_bool({"k": 1}, "k", None, False) is True
    assert m.pick_bool({"alt": 1}, "k", "alt", False) is True


def test_pick_bool_rejects_other_ints_and_non_bool():
    m = config_m()
    with pytest.raises(ValueError):
        m.pick_bool({"k": 2}, "k", None, False)
    with pytest.raises(ValueError):
        m.pick_bool({"k": "true"}, "k", None, False)
    with pytest.raises(ValueError):
        m.pick_bool({"k": 1.0}, "k", None, False)


def test_pick_str_default_and_accepts_str_and_alt_key():
    m = config_m()
    assert m.pick_str({}, "k", None, "d") == "d"
    assert m.pick_str({"k": "x"}, "k", None, "d") == "x"
    assert m.pick_str({"alt": "y"}, "k", "alt", "d") == "y"


def test_pick_str_rejects_non_str():
    m = config_m()
    with pytest.raises(ValueError):
        m.pick_str({"k": 1}, "k", None, "d")
    with pytest.raises(ValueError):
        m.pick_str({"k": True}, "k", None, "d")


def test_load_yaml_from_repo_test_data():
    m = config_m()
    if not _yaml_available():
        pytest.skip("PyYAML not available")
    p = Path(__file__).resolve().parent / "test-configs" / "config_test.yml"
    cfg = m.load_yaml(p)
    assert cfg["general"]["max_block"] == 16
    assert cfg["general"]["use_quadtree"] is False
    assert cfg["general"]["name"] == "demo"
    assert cfg["optional"]["maybe_null"] is None
    assert cfg["flags"]["enabled"] == 1
