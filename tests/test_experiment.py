import pytest

from ssvep_b2b.config import load_config
from ssvep_b2b.experiment import selected_scenarios


def test_scenario_selection_preserves_requested_order():
    config = load_config()
    selected = selected_scenarios(config, ["C&W+FGSM", "pgd+mim"])
    assert selected == [["fgsm", "cw"], ["mim", "pgd"]]


def test_scenario_selection_deduplicates_equivalent_subsets():
    config = load_config()
    selected = selected_scenarios(config, ["fgsm+cw", "cw+fgsm"])
    assert selected == [["fgsm", "cw"]]


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError):
        selected_scenarios(load_config(), ["fgsm+unknown"])
