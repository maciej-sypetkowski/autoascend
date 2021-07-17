import numpy as np

import utils
from glyph import G

ONLY_RANGED_SLOW_MONSTERS = ['floating eye']
COLD_MONSTERS = ['blue jelly', 'brown mold']

# TODO
EXPLODING_MONSTERS = ['yellow light', 'gas spore']


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


def _draw_ranged(priority, y, x, value, walkable, radius=1):
    # TODO: optimize
    for direction_y in (-1, 0, 1):
        for direction_x in (-1, 0, 1):
            if direction_y != 0 or direction_x != 0:
                for i in range(radius):
                    y1 = y + direction_y * i
                    x1 = x + direction_x * i
                    if 0 <= y1 < priority.shape[0] and 0 <= x1 < priority.shape[1]:
                        if not walkable[y1, x1]:
                            break
                        priority[y1, x1] += value


def draw_monster_priority1(agent, monster, priority, walkable):
    _, y, x, mon, _ = monster

    # don't move into the monster
    priority[y, x] = float('nan')

    if mon.mname in ('lichen', 'newt'):
        # weak monster - freely engage in melee
        _draw_around(priority, y, x, 2, radius=1, operation='max')
        _draw_around(priority, y, x, 1, radius=2, operation='max')
    elif 'mold' in mon.mname and mon.mname not in COLD_MONSTERS:
        if agent.blstats.hitpoints >= 15 or agent.blstats.hitpoints == agent.blstats.max_hitpoints:
            # freely engage in melee
            _draw_around(priority, y, x, 2, radius=1, operation='max')
            _draw_around(priority, y, x, 1, radius=2, operation='max')
    elif mon.mname in COLD_MONSTERS:
        if len(agent.get_ranged_combinations()):
            _draw_around(priority, y, x, 1, radius=2, operation='max')
        elif agent.blstats.hitpoints > 12:
            _draw_around(priority, y, x, 2, radius=1, operation='max')
            _draw_around(priority, y, x, 1, radius=2, operation='max')
    elif mon.mname in ONLY_RANGED_SLOW_MONSTERS:  # and agent.get_ranged_combinations():
        # ignore
        pass
    else:
        if agent.blstats.hitpoints > 8:
            # engage, but ensure striking first if possible
            _draw_around(priority, y, x, 3, radius=2, operation='max')


def draw_monster_priority2(agent, monster, priority, walkable):
    _, y, x, mon, _ = monster

    if agent.blstats.hitpoints <= 8:
        # stay out of melee range
        _draw_around(priority, y, x, -10, radius=1)
    elif 'mold' in mon.mname and mon.mname not in COLD_MONSTERS:
        # prioritize staying in ranged weapons line of fire
        if len(agent.get_ranged_combinations()):
            _draw_ranged(priority, y, x, 2, walkable, radius=5)
    elif mon.mname in COLD_MONSTERS:
        if len(agent.get_ranged_combinations()):
            _draw_ranged(priority, y, x, 6, walkable, radius=5)
        elif agent.blstats.hitpoints < 20 or agent.blstats.hitpoints == agent.blstats.max_hitpoints:
            _draw_around(priority, y, x, -15, radius=1)
    elif mon.mname in ('leprechaun',):
        # stay away
        _draw_around(priority, y, x, -10, radius=1)
        _draw_around(priority, y, x, -5, radius=2)
        # _draw_around(priority, y, x, 5, radius=2, operation='max')

        # prioritize staying in ranged weapons line of fire
        if len(agent.get_ranged_combinations()):
            _draw_ranged(priority, y, x, 6, walkable, radius=5)
    elif mon.mname in ONLY_RANGED_SLOW_MONSTERS:  # and agent.get_ranged_combinations():
        # ignore
        pass
    else:
        # engage, but ensure striking first if possible
        _draw_around(priority, y, x, -9, radius=1)


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
    ret = 1
    if agent.blstats.hitpoints > 8:
        ret += 15
    if wielding_ranged_weapon(agent):
        ret -= 6
    if mon.mname in COLD_MONSTERS:
        if agent.blstats.hitpoints < 20 and agent.blstats.hitpoints != agent.blstats.max_hitpoints:
            ret -= 15
        ret -= 1
    # if not wielding_melee_weapon(agent):
    #     ret -= 5
    if mon.mname in ONLY_RANGED_SLOW_MONSTERS:
        # ret -= 1
        # if agent.get_ranged_combinations():
        #     ret -= 19
        ret -= 100
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
        draw_monster_priority1(agent, m, priority, walkable)
    for m in monsters:
        draw_monster_priority2(agent, m, priority, walkable)
    priority[~walkable] = float('nan')

    # use relative priority to te current position
    priority -= priority[agent.blstats.y, agent.blstats.x]

    return priority, get_available_actions(agent, monsters)
