import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))

from temperature_rule import apply_rule, temperature_class  # noqa: E402

SETPOINT_C = 4.0


@pytest.mark.parametrize("temp_c, expected", [
    (4.0, 0), (5.0, 0), (3.0, 0),      # within +/-1 C of setpoint: Normal
    (5.5, 1), (7.0, 1), (2.5, 1),      # 1-3 C outside: Warning
    (7.01, 2), (30.0, 2), (0.0, 2),    # more than 3 C outside, either side: Critical
])
def test_bands_follow_brief(temp_c, expected):
    assert temperature_class(temp_c, SETPOINT_C) == expected


def test_steady_breach_the_models_miss_is_escalated():
    # M3 calls a steady 8 C window with normal vibration Normal (deployment_recommendation.md section 4)
    assert apply_rule(0, 8.0, SETPOINT_C) == 2
    assert apply_rule(0, 0.5, SETPOINT_C) == 2     # freezing


def test_rule_never_lowers_the_model_class():
    assert apply_rule(2, 4.0, SETPOINT_C) == 2     # vibration fault at setpoint stays Critical
    assert apply_rule(1, 4.0, SETPOINT_C) == 1
