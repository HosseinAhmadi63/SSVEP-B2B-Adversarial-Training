import numpy as np

from ssvep_b2b.metrics import classification_metrics, signal_to_noise_ratio


def test_multiclass_metrics_are_exact_for_perfect_predictions():
    labels = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    probabilities = np.eye(4)[labels]
    result = classification_metrics(labels, probabilities)
    assert result["accuracy"] == 1.0
    assert result["auc_macro_ovr"] == 1.0


def test_signal_to_noise_ratio_uses_power_ratio():
    clean = np.ones(100)
    attacked = clean + 0.1
    assert np.isclose(signal_to_noise_ratio(clean, attacked), 20.0)
