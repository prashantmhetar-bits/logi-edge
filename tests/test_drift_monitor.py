import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "monitoring"))

from drift_monitor import bin_index, distribution, extract_score, psi  # noqa: E402


def test_bin_edges_follow_spec():
    # [0,0.25) [0.25,0.50) [0.50,0.75) [0.75,1.0]
    assert [bin_index(x) for x in (0.0, 0.2499, 0.25, 0.4999, 0.5, 0.75, 1.0)] == [0, 0, 1, 1, 2, 3, 3]


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_scores_outside_unit_interval_rejected(bad):
    with pytest.raises(ValueError):
        bin_index(bad)


def test_identical_distributions_have_zero_psi():
    counts, _ = distribution([0.97] * 295 + [0.6] * 5)
    assert psi(counts, counts) == pytest.approx(0.0, abs=1e-12)


def test_clean_window_stays_below_recovery_threshold():
    reference, _ = distribution([0.97] * 295 + [0.6] * 5)
    window, _ = distribution([0.97] * 100)          # empty bin vs reference must not inflate PSI
    assert psi(reference, window) < 0.10


def test_drift_crosses_alert_threshold_within_five_minutes():
    reference, _ = distribution([0.97] * 295 + [0.6] * 5)
    # one inference per 10 s -> 30 anomalous inferences in 5 minutes, window = 100
    window, _ = distribution([0.97] * 70 + [0.01] * 30)
    assert psi(reference, window) > 0.25


def test_psi_grows_with_drift():
    reference, _ = distribution([0.97] * 300)
    values = [psi(reference, distribution([0.97] * (100 - k) + [0.01] * k)[0]) for k in (0, 5, 15, 30)]
    assert values == sorted(values)


def test_extract_score_normal_prob_and_max_conf():
    payload = {"confidence": 0.9, "probabilities": [0.05, 0.05, 0.9]}
    assert extract_score(payload, "normal_prob") == pytest.approx(0.05)
    assert extract_score(payload, "max_conf") == pytest.approx(0.9)


def test_extract_score_requires_probabilities():
    with pytest.raises(KeyError):
        extract_score({"confidence": 0.9}, "normal_prob")
