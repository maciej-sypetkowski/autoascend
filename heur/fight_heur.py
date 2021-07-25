import numpy as np
from scipy import signal
from itertools import chain

import utils
from glyph import G, MON

ONLY_RANGED_SLOW_MONSTERS = ['floating eye', 'blue jelly', 'brown mold']
# COLD_MONSTERS = ['brown mold']
COLD_MONSTERS = []

# TODO
EXPLODING_MONSTERS = ['yellow light', 'gas spore']

WEAK_MONSTERS = ['lichen', 'newt']

# TODO: nymph?
WEIRD_MONSTERS = ['leprechaun', 'nymph']


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


def _draw_ranged(priority, y, x, value, walkable, radius=1, operation='add'):
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
                        if operation == 'add':
                            priority[y1, x1] += value
                        elif operation == 'max':
                            priority[y1, x1] = max(priority[y1, x1], value)
                        else:
                            assert 0, operation


def draw_monster_priority_positive(agent, monster, priority, walkable):
    _, y, x, mon, _ = monster

    # don't move into the monster
    priority[y, x] = float('nan')

    if mon.mname in WEAK_MONSTERS:
        # weak monster - freely engage in melee
        _draw_around(priority, y, x, 2, radius=1, operation='max')
        _draw_around(priority, y, x, 1, radius=2, operation='max')
    elif 'mold' in mon.mname and mon.mname not in COLD_MONSTERS + ONLY_RANGED_SLOW_MONSTERS:
        if agent.blstats.hitpoints >= 15 or agent.blstats.hitpoints == agent.blstats.max_hitpoints:
            # freely engage in melee
            _draw_around(priority, y, x, 2, radius=1, operation='max')
            _draw_around(priority, y, x, 1, radius=2, operation='max')
    # elif mon.mname in COLD_MONSTERS:
    #     if len(agent.inventory.get_ranged_combinations()):
    #         _draw_around(priority, y, x, 1, radius=2, operation='max')
    #     elif agent.blstats.hitpoints > 12:
    #         _draw_around(priority, y, x, 2, radius=1, operation='max')
    #         _draw_around(priority, y, x, 1, radius=2, operation='max')
    elif mon.mname in ONLY_RANGED_SLOW_MONSTERS:  # and agent.inventory.get_ranged_combinations():
        # ignore
        pass
    else:
        if agent.blstats.hitpoints > 8 and not wielding_ranged_weapon(agent):
            # engage, but ensure striking first if possible
            _draw_around(priority, y, x, 3, radius=2, operation='max')
        if wielding_ranged_weapon(agent):
            _draw_ranged(priority, y, x, 4, walkable, radius=6, operation='max')


def is_monster_faster(agent, monster):
    _, y, x, mon, _ = monster
    # TOOD: implement properly
    return 'bat' in mon.mname or 'dog' in mon.mname or 'cat' in mon.mname \
           or 'kitten' in mon.mname or 'pony' in mon.mname or 'horse' in mon.mname \
           or 'bee' in mon.mname or 'fox' in mon.mname


def draw_monster_priority_negative(agent, monster, priority, walkable):
    _, y, x, mon, _ = monster

    if agent.blstats.hitpoints <= 8 and not is_monster_faster(agent, monster) and not mon.mname in WEAK_MONSTERS \
            and not mon.mname in ONLY_RANGED_SLOW_MONSTERS:
        # stay out of melee range
        _draw_around(priority, y, x, -10, radius=1)
        if not len(agent.inventory.get_ranged_combinations()):
            _draw_ranged(priority, y, x, -1, walkable, radius=6)
    elif 'mold' in mon.mname and mon.mname not in COLD_MONSTERS + ONLY_RANGED_SLOW_MONSTERS:
        # prioritize staying in ranged weapons line of fire
        if len(agent.inventory.get_ranged_combinations()):
            _draw_ranged(priority, y, x, 2, walkable, radius=5)
    # elif mon.mname in COLD_MONSTERS:
    #     if len(agent.inventory.get_ranged_combinations()):
    #         _draw_ranged(priority, y, x, 6, walkable, radius=5)
    #     elif agent.blstats.hitpoints < 20 or agent.blstats.hitpoints == agent.blstats.max_hitpoints:

    #         _draw_around(priority, y, x, -15, radius=1)
    elif mon.mname in WEIRD_MONSTERS:
        # stay away
        _draw_around(priority, y, x, -10, radius=1)
        # prioritize staying in ranged weapons line of fire
        if len(agent.inventory.get_ranged_combinations()):
            _draw_ranged(priority, y, x, 6, walkable, radius=5)
    elif mon.mname in ONLY_RANGED_SLOW_MONSTERS:  # and agent.inventory.get_ranged_combinations():
        # ignore
        pass
    else:
        if mon.mname not in WEAK_MONSTERS:
            # engage, but ensure striking first if possible
            _draw_around(priority, y, x, -9, radius=1)
            if not len(agent.inventory.get_ranged_combinations()):
                _draw_ranged(priority, y, x, -1, walkable, radius=6)


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


def melee_monster_priority(agent, monsters, monster):
    _, y, x, mon, _ = monster
    ret = 1
    if agent.blstats.hitpoints > 8 or is_monster_faster(agent, monster):
        ret += 15
    if wielding_ranged_weapon(agent) and not is_monster_faster(agent, monster):
        ret -= 6
    if mon.mname in COLD_MONSTERS:
        if agent.blstats.hitpoints < 20 and agent.blstats.hitpoints != agent.blstats.max_hitpoints:
            ret -= 15
        ret -= 1
    # if not wielding_melee_weapon(agent):
    #     ret -= 5
    if mon.mname in ONLY_RANGED_SLOW_MONSTERS:
        # ret -= 1
        # if agent.inventory.get_ranged_combinations():
        #     ret -= 19
        ret -= 100
    return ret


def _line_dis_from(agent, y, x):
    return max(abs(agent.blstats.x - x), abs(agent.blstats.y - y))


def ranged_monster_priority(agent, y, x, taret_monster, monsters):
    ret = 0
    if not (agent.blstats.y == y or agent.blstats.x == x or abs(agent.blstats.y - y) == abs(agent.blstats.x - x)):
        return None

    closest_mon_dis = float('inf')
    for monster in monsters:
        if monster is taret_monster:
            continue
        _, my, mx, mon, _ = monster
        if mon not in WEAK_MONSTERS + ONLY_RANGED_SLOW_MONSTERS:
            closest_mon_dis = min(closest_mon_dis, _line_dis_from(agent, my, mx))

    if closest_mon_dis == 1:
        ret -= 5

    dis = _line_dis_from(agent, y, x)
    if dis in (1, 2):
        ret -= 5
    if dis == 1:
        ret -= 6

    launcher, ammo = agent.inventory.get_best_ranged_set()
    if ammo is None:
        return None

    if dis > agent.character.get_range(launcher, ammo):
        return None

    if launcher is not None and not launcher.equipped:
        ret -= 5

    # search for obstacles along the line of shot
    assert y != agent.blstats.y or x != agent.blstats.x
    dir_y = np.sign(y - agent.blstats.y)
    dir_x = np.sign(x - agent.blstats.x)
    y1, x1 = agent.blstats.y + dir_y, agent.blstats.x + dir_x
    while y1 != y or x1 != x:
        if agent.glyphs[y1, x1] in G.PETS or agent.glyphs[y1, x1] in G.MONS \
                or not agent.current_level().walkable[y1, x1]:
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
    for monster in monsters:
        _, y, x, mon, _ = monster
        if utils.adjacent((y, x), (agent.blstats.y, agent.blstats.x)):
            priority = melee_monster_priority(agent, monsters, monster)
            actions.append((priority, 'melee', y, x, monster))

        ranged_pr = ranged_monster_priority(agent, y, x, monster, monsters)
        if ranged_pr is not None:
            actions.append((ranged_pr, 'ranged', y, x, monster))
    return actions


def get_corridors_priority_map(walkable):
    k = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    wall_count = signal.convolve2d((~walkable).astype(int), k, boundary='symm', mode='same')
    corridor_mask = (wall_count == 6).astype(int)
    corridor_mask[~walkable] = 0
    corridor_dilated = signal.convolve2d(corridor_mask.astype(int), k, boundary='symm', mode='same')
    return corridor_mask + corridor_dilated >= 1


def build_priority_map(agent):
    walkable = agent.current_level().walkable
    priority = np.zeros(walkable.shape, dtype=float)
    monsters = agent.get_visible_monsters()
    for m in monsters:
        draw_monster_priority_positive(agent, m, priority, walkable)
    for m in monsters:
        draw_monster_priority_negative(agent, m, priority, walkable)
    priority[~walkable] = float('nan')

    # TODO: figure out how to use corridors priority so that it improves the score
    # if len([m for m in monsters if m[3].mname not in chain(ONLY_RANGED_SLOW_MONSTERS, WEAK_MONSTERS)]) >= 4:
    #     priority += get_corridors_priority_map(walkable)
    # for _, _, _, mon, _ in monsters:
    #     if ord(mon.mlet) == MON.S_ANT:
    #         priority += get_corridors_priority_map(walkable)
    #         break

    # use relative priority to te current position
    priority -= priority[agent.blstats.y, agent.blstats.x]

    return priority, get_available_actions(agent, monsters)
