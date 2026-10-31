"""Temperature safety rule applied after the model (Task F3).

The brief defines the classes partly by temperature alone: Normal within +/-1 C of
setpoint, Warning 1-3 C outside it, Critical more than 3 C outside it. The models do
not enforce this for a steady (no longer drifting) breach or a cold excursion - see
deployment/deployment_recommendation.md section 4 - so the service raises the model's
class to the class these bands require. The rule never lowers the model's class.
"""

WARNING_DEVIATION_C = 1.0
CRITICAL_DEVIATION_C = 3.0


def temperature_class(temp_mean_c, setpoint_c):
    """Class the brief's temperature bands require for a window's mean temperature."""
    deviation = abs(temp_mean_c - setpoint_c)
    if deviation > CRITICAL_DEVIATION_C:
        return 2
    if deviation > WARNING_DEVIATION_C:
        return 1
    return 0


def apply_rule(model_class, temp_mean_c, setpoint_c):
    """Final class: the model's class, raised to the temperature band's class when that is higher."""
    return max(model_class, temperature_class(temp_mean_c, setpoint_c))
