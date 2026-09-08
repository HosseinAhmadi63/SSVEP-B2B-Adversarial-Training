import itertools

from ssvep_b2b.config import load_config, scenario_slug


def test_paper_configuration_contains_every_nonempty_attack_subset():
    config = load_config()
    scenarios = {frozenset(item) for item in config.values["scenarios"]}
    attacks = ["fgsm", "bim", "cw", "mim", "pgd"]
    expected = {
        frozenset(items)
        for size in range(1, len(attacks) + 1)
        for items in itertools.combinations(attacks, size)
    }
    assert scenarios == expected


def test_paper_dataset_trial_counts_and_split():
    config = load_config()
    datasets = config.section("datasets")
    nakanishi = datasets["Nakanishi2015"]
    lee = datasets["Lee2019_SSVEP"]
    assert len(nakanishi["subjects"]) * nakanishi["classes"] * nakanishi["trials_per_class"] == 1620
    assert len(lee["subjects"]) * lee["classes"] * lee["trials_per_class"] == 10800
    assert config.section("split")["test_size"] == 0.2


def test_scenario_slug_is_stable():
    assert scenario_slug(["fgsm", "cw", "pgd"]) == "fgsm+cw+pgd"
