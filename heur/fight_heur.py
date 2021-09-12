from collections import defaultdict
from itertools import product

import nle.nethack as nh
import numpy as np
from scipy import signal

import objects as O
import utils
from glyph import G
from item import flatten_items

ONLY_RANGED_SLOW_MONSTERS = ['floating eye', 'blue jelly', 'brown mold', 'gas spore']
# COLD_MONSTERS = ['brown mold']
# COLD_MONSTERS = []

# TODO
EXPLODING_MONSTERS = ['yellow light', 'gas spore', 'flaming sphere', 'freezing sphere', 'shocking sphere']

INSECTS = ['giant ant', 'killer bee', 'soldier ant', 'fire ant', 'giant beetle', 'queen bee']

WEAK_MONSTERS = ['lichen', 'newt', 'shrieker', 'grid bug']

# TODO: nymph?
WEIRD_MONSTERS = ['leprechaun', 'nymph']


def consider_melee_only_ranged_if_hp_full(agent, monster):
    return monster[3].mname in ('brown mold', 'blue jelly') and agent.blstats.hitpoints == agent.blstats.max_hitpoints


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
    elif mon.mname in ONLY_RANGED_SLOW_MONSTERS:  # and agent.inventory.get_ranged_combinations():
        if consider_melee_only_ranged_if_hp_full(agent, monster):
            _draw_around(priority, y, x, 2, radius=1, operation='max')
            _draw_around(priority, y, x, 1, radius=2, operation='max')
        if len(agent.inventory.get_ranged_combinations()):
            _draw_ranged(priority, y, x, 1, walkable, radius=7, operation='max')
    else:
        if not imminent_death_on_melee(agent, monster) and not wielding_ranged_weapon(agent):
            # engage, but ensure striking first if possible
            if mon.mmove <= 12:
                _draw_around(priority, y, x, 3, radius=2, operation='max')
            else:
                _draw_around(priority, y, x, 3, radius=3, operation='max')
        if wielding_ranged_weapon(agent):
            _draw_ranged(priority, y, x, 4, walkable, radius=7, operation='max')
        elif len(agent.inventory.get_ranged_combinations()):
            _draw_ranged(priority, y, x, 1, walkable, radius=7, operation='max')


def is_monster_faster(agent, monster):
    _, y, x, mon, _ = monster
    # TOOD: implement properly
    return 'bat' in mon.mname or 'dog' in mon.mname or 'cat' in mon.mname \
           or 'kitten' in mon.mname or 'pony' in mon.mname or 'horse' in mon.mname \
           or 'bee' in mon.mname or 'fox' in mon.mname


def imminent_death_on_melee(agent, monster):
    if is_dangerous_monster(monster):
        return agent.blstats.hitpoints <= 15
    return agent.blstats.hitpoints <= 8


def draw_monster_priority_negative(agent, monster, priority, walkable):
    _, y, x, mon, _ = monster

    if imminent_death_on_melee(agent, monster) and not mon.mname in WEAK_MONSTERS \
            and not mon.mname in ONLY_RANGED_SLOW_MONSTERS:
        if mon.mmove <= 12:
            _draw_around(priority, y, x, -10, radius=1)
        else:
            if utils.adjacent((agent.blstats.y, agent.blstats.x), (y, x)):
                # no point in running -- monster is fast
                pass
            else:
                _draw_around(priority, y, x, -10, radius=2)
                _draw_around(priority, y, x, -5, radius=1)

        if not len(agent.inventory.get_ranged_combinations()):
            # prefer avoiding being in line of fire
            _draw_ranged(priority, y, x, -1, walkable, radius=7)

    # if agent.blstats.hitpoints <= 8 and not is_monster_faster(agent, monster) and not mon.mname in WEAK_MONSTERS \
    #         and not mon.mname in ONLY_RANGED_SLOW_MONSTERS:
    #     # stay out of melee range
    #     _draw_around(priority, y, x, -10, radius=1)
    #     if not len(agent.inventory.get_ranged_combinations()):
    #         # prefer avoiding being in line of fire
    #         _draw_ranged(priority, y, x, -1, walkable, radius=7)

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
        if not consider_melee_only_ranged_if_hp_full(agent, monster):
            ret -= 100
            if mon.mname == 'floating eye':
                ret -= 10
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
        ret -= 11

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


def inside(agent, y, x):
    return 0 <= y < agent.glyphs.shape[0] and 0 <= x < agent.glyphs.shape[1]


def get_next_states(agent, wand, y, x, dy, dx):
    if not inside(agent, y, x) or not agent.current_level().walkable[y, x]:
        can_bounce = wand.is_ray_wand()
        if not can_bounce:
            return []
        if dy == 0 or dx == 0:
            return [(y - dy, x - dx, -dy, -dx, 1.0, 1)]
        # TODO: diagonal
        side1 = (y, x - dx)
        side2 = (y - dy, x)
        side1_wall = not inside(agent, *side1) or not agent.current_level().walkable[side1]
        side2_wall = not inside(agent, *side2) or not agent.current_level().walkable[side2]
        dy1, dx1 = side2[0] - side1[0], side2[1] - side1[1]
        dy2, dx2 = side1[0] - side2[0], side1[1] - side2[1]
        if side1_wall and side2_wall:
            return [(y - dy, x - dx, -dy, -dx, 1.0, 1)]
        elif not side1_wall and not side2_wall:
            return [(y - dy, x - dx, -dy, -dx, 1 / 20, 1),
                    (y + dy1, x + dx1, dy1, dx1, 19 / 40, 1),
                    (y + dy2, x + dx2, dy2, dx2, 19 / 40, 1)]
        elif side1_wall:
            return [(y + dy1, x + dx1, dy1, dx1, 1.0, 1)]
        elif side2_wall:
            return [(y + dy2, x + dx2, dy2, dx2, 1.0, 1)]
        else:
            assert 0
    return [(y + dy, x + dx, dy, dx, 1.0, 0)]


def _simulate_wand_path(agent, wand, monsters, y, x, dy, dx, range_left, hit_targets, probability):
    if range_left < 0:
        return

    for y, x, dy, dx, next_prob, range_penalty in get_next_states(agent, wand, y, x, dy, dx):
        range_left -= range_penalty
        monster = [m for m in monsters if m[1] == y and m[2] == x]
        if monster:
            assert len(monster) == 1
            monster = monster[0]
            # For each monster hit, range decreases by 2.
            range_left -= 2
        elif inside(agent, y, x) and agent.glyphs[y, x] in G.PETS:
            monster = 'pet'
            # For each monster hit, range decreases by 2.
            range_left -= 2
        elif agent.blstats.y == y and agent.blstats.x == x:
            monster = 'self'
            range_left -= 2
        else:
            monster = None

        hit_targets[(y, x, monster)] += probability * next_prob

        _simulate_wand_path(agent, wand, monsters, y, x, dy, dx, range_left - 1, hit_targets, 1.0)


def simulate_wand_path(agent, wand, monsters, dy, dx):
    """ Returns list of tuples (y, x, hit_object, expected_hit_count).
    """
    y, x = agent.blstats.y, agent.blstats.x

    # TODO: random range left from 6 or 7 to 13
    hit_targets = defaultdict(int)
    _simulate_wand_path(agent, wand, monsters, y, x, dy, dx, 13, hit_targets, 1.0)
    for (y, x, hit_object), expected_hit_count in hit_targets.items():
        yield y, x, hit_object, expected_hit_count


def is_dangerous_monster(monster):
    _, y, x, mon, _ = monster
    is_pet = 'dog' in mon.mname or 'cat' in mon.mname or 'kitten' in mon.mname or 'pony' in mon.mname \
             or 'horse' in mon.mname
    return is_pet or mon.mname in INSECTS

def get_potential_wand_usages(agent, monsters, dy, dx):
    ret = []
    player_hp_ratio = agent.blstats.hitpoints / agent.blstats.max_hitpoints
    # for item in flatten_items(agent.inventory.items):
    #     item = self.inventory.move_to_inventory(item)
    for item in agent.inventory.items:
        targeted_monsters = set()
        if not item.is_offensive_usable_wand():
            continue
        priority = 0
        # print('--------------', dy, dx)
        for y, x, monster, p in simulate_wand_path(agent, item, monsters, dy, dx):
            # print(y, x, monster, p)
            if monster == 'pet':
                priority -= p * 20
            elif monster == 'self':
                priority -= p * 30
            elif monster is not None:
                _, y, x, mon, _ = monster
                if mon.mname in WEAK_MONSTERS:
                    priority += min(p, 1) * 1
                elif is_dangerous_monster(monster):
                    priority += p * 25
                else:
                    priority += min(p, 1) * 10
                targeted_monsters.add((y, x, monster))
        if targeted_monsters:
            # priority = priority * (1 - player_hp_ratio) - 10
            priority = priority - 15
            if agent.inventory.engraving_below_me.lower() == 'elbereth':
                priority -= 100
            ret.append((priority, 'zap', dy, dx, item, targeted_monsters))
    return ret


def elbereth_action(agent, monsters):
    if agent.inventory.engraving_below_me.lower() == 'elbereth':
        return []
    if not agent.can_engrave():
        return []
    adj_monsters_count = 0
    for monster in monsters:
        _, my, mx, mon, _ = monster
        if mon.mname in ONLY_RANGED_SLOW_MONSTERS:
            continue
        if not utils.adjacent((my, mx), (agent.blstats.y, agent.blstats.x)):
            continue
        multiplier = np.clip(20 / agent.blstats.hitpoints, 1.0, 1.5)
        if is_monster_faster(agent, monster):
            multiplier *= 2
        if mon in WEAK_MONSTERS:
            adj_monsters_count += 0.1 * multiplier
            continue
        adj_monsters_count += 1 * multiplier
        if is_dangerous_monster(monster):
            adj_monsters_count += 2 * multiplier

    player_hp_ratio = (agent.blstats.hitpoints / agent.blstats.max_hitpoints) ** 0.5
    if agent.blstats.hitpoints < 30 and adj_monsters_count > 0:
        return [(-15 + 20 * adj_monsters_count * (1 - player_hp_ratio), 'elbereth')]
    return []


def wait_action(agent, monsters):
    if agent.inventory.engraving_below_me.lower() == 'elbereth':
        player_hp_ratio = agent.blstats.hitpoints / agent.blstats.max_hitpoints
        priority = 25 - player_hp_ratio * 40
        return [(priority, 'wait')]
    return []


def get_available_actions(agent, monsters):
    actions = []

    # melee attack actions
    for monster in monsters:
        _, y, x, mon, _ = monster
        if utils.adjacent((y, x), (agent.blstats.y, agent.blstats.x)):
            priority = melee_monster_priority(agent, monsters, monster)
            if agent.inventory.engraving_below_me.lower() == 'elbereth':
                priority -= 100
            actions.append((priority, 'melee', y, x, monster))

    # ranged attack actions
    for dy, dx in product([-1, 0, 1], [-1, 0, 1]):
        if dy != 0 or dx != 0:
            ranged_pr = ranged_priority(agent, dy, dx, monsters)
            if ranged_pr is not None:
                pri, y, x, monster = ranged_pr
                if agent.inventory.engraving_below_me.lower() == 'elbereth':
                    pri -= 100
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

    # actions.extend(elbereth_action(agent, monsters))
    # actions.extend(wait_action(agent, monsters))

    return actions


def get_corridors_priority_map(walkable):
    k = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    wall_count = signal.convolve2d((~walkable).astype(int), k, boundary='symm', mode='same')
    corridor_mask = (wall_count == 6).astype(int)
    corridor_mask[~walkable] = 0
    corridor_dilated = signal.convolve2d(corridor_mask.astype(int), k, boundary='symm', mode='same')
    return corridor_mask + corridor_dilated >= 1


def get_priorities(agent):
    """ Returns a pair (move priority heatmap, other actions (with priorities) list) """
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
