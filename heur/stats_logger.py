class StatsLogger:
    def __init__(self):
        self._values = {
            "agent_panic": 0,
            "elbereth_write": 0,
            "container_untrap_success": 0,
            "container_untrap_fail": 0,
            "untrap_success": 0,
            "triggered_undetected_trap": 0,
        }

    def log_event(self, name):
        self._values[name] += 1

    def get_stats_dict(self):
        return self._values
