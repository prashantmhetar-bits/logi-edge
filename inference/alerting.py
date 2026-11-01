import json
import os
from pathlib import Path

SEVERITY = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}


class AlertTracker:

    def __init__(self, truck_id, model_name):
        self.truck_id = truck_id
        self.model_name = model_name
        self.previous_status = None
        self.seq = 0

    def update(self, status, model_status, temp_mean_c, confidence, timestamp):
        if status not in SEVERITY:
            raise ValueError(f"unknown status {status!r}")
        previous = self.previous_status
        self.previous_status = status
        if status == previous or (previous is None and status == "NORMAL"):
            return None

        self.seq += 1
        if previous is None or SEVERITY[status] > SEVERITY[previous]:
            event = "ESCALATED"
        elif status == "NORMAL":
            event = "CLEARED"
        else:
            event = "DEESCALATED"
        return {
            "truck_id": self.truck_id,
            "seq": self.seq,
            "event": event,
            "status": status,
            "previous_status": previous,
            "model_status": model_status,
            "temp_mean_c": round(float(temp_mean_c), 3),
            "confidence": round(float(confidence), 4),
            "model": self.model_name,
            "timestamp": timestamp,
        }


class AlertLog:

    def __init__(self, path):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, alert):
        if not self.path:
            return False
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(alert, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return True
