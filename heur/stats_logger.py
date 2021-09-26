class StatsLogger:
    def __init__(self):
        self._values = {
            "agent_panic": 0,
            "elbereth_write": 0,
            "container_untrap_success": 0,
            "container_untrap_fail": 0,
            "untrap_success": 0,
            "triggered_undetected_trap": 0,
            "sokoban_dropped": 0,
        }
        self._max_values = {
            "search_diff": -float('inf'),
        }
        self._keys = list(self._values) + list(self._max_values)

    def log_event(self, name):
        self._values[name] += 1

    def log_max_value(self, name, value):
        self._max_values[name] = max(self._max_values[name], value)

    def get_stats_dict(self):
        ret = dict()
        ret.update(self._values)
        ret.update(self._max_values)
        return ret
