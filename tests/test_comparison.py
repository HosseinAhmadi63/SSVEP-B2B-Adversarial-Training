import pandas as pd

from ssvep_b2b.comparison import _scenario_key


def test_scenario_key_normalizes_publication_and_run_labels():
    labels = pd.Series(["FGSM + C&W + PGD", "fgsm+cw+pgd"])
    assert labels.map(_scenario_key).nunique() == 1
