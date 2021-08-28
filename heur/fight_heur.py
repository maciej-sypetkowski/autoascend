from itertools import product

import nle.nethack as nh
import numpy as np
from scipy import signal

import objects as O
import utils
from glyph import G

ONLY_RANGED_SLOW_MONSTERS = ['floating eye', 'blue jelly', 'brown mold', 'gas spore']
# COLD_MONSTERS = ['brown mold']
# COLD_MONSTERS = []

# TODO
EXPLODING_MONSTERS = ['yellow light', 'gas spore', 'flaming sphere', 'freezing sphere', 'shocking sphere']

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
                for i in range(1, radius + 1):
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
    elif 'mold' in mon.mname and mon.mname not in ONLY_RANGED_SLOW_MONSTERS:
        if agent.blstats.hitpoints >= 15 or agent.blstats.hitpoints == agent.blstats.max_hitpoints:
            # freely engage in melee
            _draw_around(priority, y, x, 2, radius=1, operation='max')
            _draw_around(priority, y, x, 1, radius=2, operation='max')
        if len(agent.inventory.get_ranged_combinations()):
            _draw_ranged(priority, y, x, 1, walkable, radius=7, operation='max')
    # elif mon.mname in COLD_MONSTERS:
    #     if len(agent.inventory.get_ranged_combinations()):
    #         _draw_around(priority, y, x, 1, radius=2, operation='max')
    #     elif agent.blstats.hitpoints > 12:
    #         _draw_around(priority, y, x, 2, radius=1, operation='max')
    #         _draw_around(priority, y, x, 1, radius=2, operation='max')
    elif mon.mname in ONLY_RANGED_SLOW_MONSTERS:  # and agent.inventory.get_ranged_combinations():
        if len(agent.inventory.get_ranged_combinations()):
            _draw_ranged(priority, y, x, 1, walkable, radius=7, operation='max')
    else:
        if agent.blstats.hitpoints > 8 and not wielding_ranged_weapon(agent):
            # engage, but ensure striking first if possible
            _draw_around(priority, y, x, 3, radius=2, operation='max')
        if wielding_ranged_weapon(agent):
            _draw_ranged(priority, y, x, 4, walkable, radius=7, operation='max')


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
            # prefer avoiding being in line of fire
            _draw_ranged(priority, y, x, -1, walkable, radius=7)

    if mon.mname in EXPLODING_MONSTERS:
        _draw_around(priority, y, x, -10, radius=1)
        if mon.mname not in ONLY_RANGED_SLOW_MONSTERS:
            _draw_around(priority, y, x, -5, radius=2)
        _draw_ranged(priority, y, x, 4, walkable, radius=7)
    elif 'mold' in mon.mname and mon.mname not in ONLY_RANGED_SLOW_MONSTERS:
        # prioritize staying in ranged weapons line of fire
        if len(agent.inventory.get_ranged_combinations()):
            _draw_ranged(priority, y, x, 2, walkable, radius=7)
    elif mon.mname in WEIRD_MONSTERS:
        # stay away
        _draw_around(priority, y, x, -10, radius=1)
        # prioritize staying in ranged weapons line of fire
        if len(agent.inventory.get_ranged_combinations()):
            _draw_ranged(priority, y, x, 6, walkable, radius=7)
    elif mon.mname in ONLY_RANGED_SLOW_MONSTERS:  # and agent.inventory.get_ranged_combinations():
        # ignore
        pass
    else:
        if mon.mname not in WEAK_MONSTERS:
            # engage, but ensure striking first if possible
            _draw_around(priority, y, x, -9, radius=1)
            if not len(agent.inventory.get_ranged_combinations()):
                _draw_ranged(priority, y, x, -1, walkable, radius=7)

    if mon.mname == 'purple worm' and len(agent.inventory.get_ranged_combinations()):
        _draw_around(priority, y, x, -10, radius=1)


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
    if mon.mname in EXPLODING_MONSTERS:
        ret -= 17
    # if not wielding_melee_weapon(agent):
    #     ret -= 5
    if mon.mname in ONLY_RANGED_SLOW_MONSTERS:
        ret -= 100
    return ret


def _line_dis_from(agent, y, x):
    return max(abs(agent.blstats.x - x), abs(agent.blstats.y - y))


def ranged_priority(agent, dy, dx, monsters):
    ret = 11

    closest_mon_dis = float('inf')
    for monster in monsters:
        _, my, mx, mon, _ = monster
        assert my != agent.blstats.y or mx != agent.blstats.x
        if mon.mname not in WEAK_MONSTERS + ONLY_RANGED_SLOW_MONSTERS:
            closest_mon_dis = min(closest_mon_dis, _line_dis_from(agent, my, mx))

    if closest_mon_dis == 1:
        ret -= 5

    launcher, ammo = agent.inventory.get_best_ranged_set()
    if ammo is None:
        return None

    if launcher is not None and not launcher.equipped:
        ret -= 5

    y, x = agent.blstats.y, agent.blstats.x
    while True:
        y += dy
        x += dx
        if not 0 <= y < agent.glyphs.shape[0] or not 0 <= x < agent.glyphs.shape[1]:
            return None

        if agent.glyphs[y, x] in G.PETS or not agent.current_level().walkable[y, x]:
            return None

        if agent.glyphs[y, x] in G.MONS:
            monster = [m for m in monsters if m[1] == y and m[2] == x]
            if not monster:
                # there is a monster that shouldn't be attacked
                return None
            assert len(monster) == 1
            _, _, _, mon, _ = monster[0]
            dis = _line_dis_from(agent, y, x)
            if dis > agent.character.get_range(launcher, ammo):
                return None
            if dis in (1, 2):
                ret -= 5
            if dis == 1:
                ret -= 6
                if mon.mname == 'gas spore':  # only gas spore ?
                    ret -= 100
            return ret, y, x, monster[0]


def get_potential_wand_usages(agent, monsters, dy, dx):
    for item in agent.inventory.items:
        if len(item.objs) == 1 and item.objs[0] == O.from_name('magic missile', nh.WAND_CLASS):
            pass
    return []


def get_available_actions(agent, monsters):
    actions = []

    # melee attack actions
    for monster in monsters:
        _, y, x, mon, _ = monster
        if utils.adjacent((y, x), (agent.blstats.y, agent.blstats.x)):
            priority = melee_monster_priority(agent, monsters, monster)
            actions.append((priority, 'melee', y, x, monster))

    # ranged attack actions
    for dy, dx in product([-1, 0, 1], [-1, 0, 1]):
        if dy != 0 or dx != 0:
            ranged_pr = ranged_priority(agent, dy, dx, monsters)
            if ranged_pr is not None:
                pri, y, x, monster = ranged_pr
                actions.append((pri, 'ranged', y, x, monster))

            actions.extend(get_potential_wand_usages(agent, monsters, dy, dx))

    # pickup items actions
    projectiles_below_me = [i for i in agent.inventory.items_below_me
                            if i.is_thrown_projectile() or i.is_fired_projectile()]
    my_launcher, ammo = agent.inventory.get_best_ranged_set(additional_ammo=[i for i in projectiles_below_me])
    to_pickup = []
    for item in agent.inventory.items_below_me:
        if item.is_thrown_projectile() or (my_launcher is not None and item.is_fired_projectile(launcher=my_launcher)):
            to_pickup.append(item)
    if to_pickup:
        actions.append((15, 'pickup', to_pickup))

    return actions


def get_corridors_priority_map(walkable):
    k = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    wall_count = signal.convolve2d((~walkable).astype(int), k, boundary='symm', mode='same')
    corridor_mask = (wall_count == 6).astype(int)
    corridor_mask[~walkable] = 0
    corridor_dilated = signal.convolve2d(corridor_mask.astype(int), k, boundary='symm', mode='same')
    return corridor_mask + corridor_dilated >= 1


def get_priorities(agent):
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
