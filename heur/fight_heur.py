import numpy as np

import utils
from glyph import G


def _draw_around(priority, y, x, value, radius=1, operation='add'):
    # TODO: optimize
    for y1 in range(y - radius, y + radius + 1):
        for x1 in range(x - radius, x + radius + 1):
            if max(abs(y1 - y), abs(x1 - x)) != radius:
                continue
            if 0 <= y1 < priority.shape[0] and 0 <= x1 < priority.shape[1]:
                if operation == 'add':
                    priority[y1, x1] += value
                elif operation == 'max':
                    priority[y1, x1] = max(priority[y1, x1], value)
                else:
                    assert 0, operation


def _draw_ranged(priority, y, x, value, radius=1):
    # TODO: optimize
    for direction_y in (-1, 0, 1):
        for direction_x in (-1, 0, 1):
            if direction_y != 0 or direction_x != 0:
                for i in range(radius):
                    y1 = y + direction_y * i
                    x1 = x + direction_x * i
                    if 0 <= y1 < priority.shape[0] and 0 <= x1 < priority.shape[1]:
                        priority[y1, x1] += value


def draw_monster_priority(agent, monster, priority, walkable):
    _, y, x, mon, _ = monster

    # don't move into the monster
    priority[y, x] = float('nan')

    if agent.blstats.hitpoints <= 8:
        # stay out of melee range
        _draw_around(priority, y, x, -10, radius=1)
    elif mon.mname in ('lichen', 'newt'):
        # weak monster - freely engage in melee
        _draw_around(priority, y, x, 6, radius=1, operation='max')
        _draw_around(priority, y, x, 5, radius=2, operation='max')
    elif 'mold' in mon.mname:
        # freely engage in melee
        # prioritize staying in ranged distance
        _draw_around(priority, y, x, 6, radius=1, operation='max')
        _draw_around(priority, y, x, 5, radius=2, operation='max')

        # prioritize staying in ranged weapons line of fire
        _draw_ranged(priority, y, x, 2, radius=3)

    elif mon.mname in ('leprechaun',):
        # stay away
        _draw_around(priority, y, x, -10, radius=1)
        _draw_around(priority, y, x, -5, radius=2)
        # _draw_around(priority, y, x, 5, radius=2, operation='max')

        # prioritize staying in ranged weapons line of fire
        _draw_ranged(priority, y, x, 6, radius=3)
    elif mon.mname in ('floating eye',) and agent.get_ranged_combinations():
        # stay away
        _draw_around(priority, y, x, -20, radius=1)
        _draw_around(priority, y, x, -5, radius=2)
        # _draw_around(priority, y, x, 5, radius=2, operation='max')

        # prioritize staying in ranged weapons line of fire
        _draw_ranged(priority, y, x, 6, radius=3)
    else:
        # engage, but ensure striking first if possible
        _draw_around(priority, y, x, -5, radius=1)
        _draw_around(priority, y, x, 5, radius=2, operation='max')


def wielding_ranged_weapon(agent):
    for item in agent.inventory.items:
        if item.is_launcher() and item.equipped:
            return True
    return False


def wielding_melee_weapon(agent):
    for item in agent.inventory.items:
        if item.is_weapon() and item.equipped:
            return True
    return False


def melee_monster_priority(agent, monsters, mon):
    ret = 0
    if agent.blstats.hitpoints <= 8:
        ret -= 5
    else:
        ret += 15
    if wielding_ranged_weapon(agent):
        ret -= 6
    if not wielding_melee_weapon(agent):
        ret -= 5
    if mon.mname in ('floating eye',):
        ret -= 1
        if agent.get_ranged_combinations():
            ret -= 19
    return ret


def ranged_monster_priority(agent, y, x, mon):
    ret = 0
    if not (agent.blstats.y == y or agent.blstats.x == x or abs(agent.blstats.y - y) == abs(agent.blstats.x - x)):
        return None

    if max(abs(agent.blstats.x - x), abs(agent.blstats.y - y)) in (1, 2):
        ret -= 5

    # TODO: check if there is a walkable path

    ranged_combinations = agent.get_ranged_combinations()
    if not ranged_combinations:
        return None

    # TODO: select best
    launcher, ammo = ranged_combinations[0]

    if launcher is not None and not launcher.equipped:
        ret -= 5

    # search for obstacles along the line of shot
    assert y != agent.blstats.y or x != agent.blstats.x
    dir_y = np.sign(y - agent.blstats.y)
    dir_x = np.sign(x - agent.blstats.x)
    y1, x1 = agent.blstats.y + dir_y, agent.blstats.x + dir_x
    while y1 != y or x1 != x:
        if agent.glyphs[y1, x1] in G.PETS or not agent.current_level().walkable[y1, x1]:
            ret -= 100
        y1 += dir_y
        x1 += dir_x
        assert 0 <= y1 < agent.glyphs.shape[0]
        assert 0 <= x1 < agent.glyphs.shape[1]

    # TODO: limited range
    ret += 11

    return ret


def get_available_actions(agent, monsters):
    actions = []
    for _, y, x, mon, _ in monsters:
        if utils.adjacent((y, x), (agent.blstats.y, agent.blstats.x)):
            priority = melee_monster_priority(agent, monsters, mon)
            actions.append((priority, 'melee', y, x))

        ranged_pr = ranged_monster_priority(agent, y, x, mon)
        if ranged_pr is not None:
            actions.append((ranged_pr, 'ranged', y, x))
    return actions


def build_priority_map(agent):
    walkable = agent.current_level().walkable
    priority = np.zeros(walkable.shape, dtype=float)
    monsters = agent.get_visible_monsters()
    for m in monsters:
        draw_monster_priority(agent, m, priority, walkable)
    priority[~walkable] = float('nan')

    # use relative priority to te current position
    priority -= priority[agent.blstats.y, agent.blstats.x]

    return priority, get_available_actions(agent, monsters)
