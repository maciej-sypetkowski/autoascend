class StatsLogger:
    def __init__(self):
        self._values = {
            "elbereth_write": 0,
            "untrap_success": 0,
        }

    def log_event(self, name):
        self._values[name] += 1

    def get_stats_dict(self):
        return self._values