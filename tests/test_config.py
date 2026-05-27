import pytest

from skillopt.config import Config


def test_from_dict_nests_sections():
    cfg = Config.from_dict({
        "benchmark": "hotpotqa",
        "metric": "em",
        "model": {"target_model": "m1"},
        "train": {"num_epochs": 5, "batch_size": 4},
    })
    assert cfg.metric == "em"
    assert cfg.model.target_model == "m1"
    assert cfg.train.num_epochs == 5


def test_apply_overrides_routes_to_sections():
    cfg = Config()
    cfg.apply_overrides({"target_model": "x", "num_epochs": 9, "metric": "em"})
    assert cfg.model.target_model == "x"
    assert cfg.train.num_epochs == 9
    assert cfg.metric == "em"


def test_apply_overrides_ignores_none():
    cfg = Config()
    original = cfg.model.target_model
    cfg.apply_overrides({"target_model": None})
    assert cfg.model.target_model == original


def test_apply_overrides_unknown_key_raises():
    with pytest.raises(KeyError):
        Config().apply_overrides({"nonexistent": 1})
