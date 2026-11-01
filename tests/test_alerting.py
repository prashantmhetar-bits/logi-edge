import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))

from alerting import AlertLog, AlertTracker


def feed(tracker, statuses):
    return [tracker.update(s, s, 4.0, 0.9, float(i)) for i, s in enumerate(statuses)]


def test_healthy_start_sends_nothing():
    assert feed(AlertTracker("truck_001", "m.tflite"), ["NORMAL", "NORMAL"]) == [
        None,
        None,
    ]


def test_one_alert_per_state_change_not_per_window():
    alerts = [
        a
        for a in feed(
            AlertTracker("truck_001", "m.tflite"),
            [
                "NORMAL",
                "WARNING",
                "WARNING",
                "CRITICAL",
                "CRITICAL",
                "WARNING",
                "NORMAL",
            ],
        )
        if a
    ]
    assert [(a["previous_status"], a["status"], a["event"]) for a in alerts] == [
        ("NORMAL", "WARNING", "ESCALATED"),
        ("WARNING", "CRITICAL", "ESCALATED"),
        ("CRITICAL", "WARNING", "DEESCALATED"),
        ("WARNING", "NORMAL", "CLEARED"),
    ]
    assert [a["seq"] for a in alerts] == [1, 2, 3, 4]


def test_fault_present_at_start_is_alerted():
    first = feed(AlertTracker("truck_001", "m.tflite"), ["CRITICAL"])[0]
    assert first["event"] == "ESCALATED" and first["previous_status"] is None


def test_unknown_status_rejected():
    with pytest.raises(ValueError):
        AlertTracker("truck_001", "m.tflite").update("BROKEN", "NORMAL", 4.0, 0.9, 0.0)


def test_alert_log_appends_json_lines(tmp_path):
    log = AlertLog(tmp_path / "sub" / "alerts.jsonl")
    tracker = AlertTracker("truck_001", "m.tflite")
    for a in feed(tracker, ["WARNING", "CRITICAL"]):
        assert log.append(a)
    rows = [
        json.loads(line)
        for line in (tmp_path / "sub" / "alerts.jsonl").read_text().splitlines()
    ]
    assert [r["status"] for r in rows] == ["WARNING", "CRITICAL"]
    assert rows[0]["truck_id"] == "truck_001" and rows[0]["model"] == "m.tflite"


def test_alert_log_disabled_when_path_empty():
    assert AlertLog("").append({"x": 1}) is False
