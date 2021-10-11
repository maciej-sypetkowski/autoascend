from collections import defaultdict


class StatsLogger:
    def __init__(self):
        self._values = {
            "agent_panic": 0,
            "elbereth_write": 0,
            "container_untrap_success": 0,
            "container_untrap_fail": 0,
            "untrap_success": 0,
            "triggered_undetected_trap": 0,
            "allow_walk_traps": 0,
            "allow_attack_all": 0,
            "sokoban_dropped": 0,
            "wait_in_fight": 0,
            "melee_gas_spore": 0,
        }
        self._max_values = {
            "search_diff": -float('inf'),
        }

        self._cumulative_values = {
            "max_turns_on_position": defaultdict(int),
        }

        self._keys = list(self._values) + list(self._max_values) + list(self._cumulative_values)

    def log_cumulative_value(self, name, key, value):
        self._cumulative_values[name][key] += value

    def log_event(self, name):
        self._values[name] += 1

    def log_max_value(self, name, value):
        self._max_values[name] = max(self._max_values[name], value)

    def get_stats_dict(self):
        ret = dict()
        ret.update(self._values)
        ret.update(self._max_values)
        ret.update({k: max(v.values()) for k, v in self._cumulative_values.items()})
        return ret
