from enum import IntEnum, auto

import nle.nethack as nh
import numpy as np

import objects as O
import soko_solver
import utils
from character import Character
from exceptions import AgentPanic
from glyph import Hunger, G
from item import Item, ItemPriorityBase
from level import Level
from strategy import Strategy


class ItemPriority(ItemPriorityBase):
    def __init__(self, agent):
        self.agent = agent

    def split(self, items, forced_items, weight_capacity):
        remaining_weight = weight_capacity
        ret = {}
        for item in forced_items:
            remaining_weight -= item.weight()
            ret[item] = item.count

        def add_item(item, count=None):
            nonlocal ret, remaining_weight
            assert isinstance(item, Item)
            if remaining_weight < 0:
                return

            how_many_already = ret.get(item, 0)
            max_to_add = int(remaining_weight // item.unit_weight())
            if count is not None:
                max_to_add = min(max_to_add, count)
            ret[item] = min(item.count, how_many_already + max_to_add)
            remaining_weight -= item.unit_weight() * (ret[item] - how_many_already)

        for allow_unknown_status in [False, True]:
            item = self.agent.inventory.get_best_weapon(items=forced_items + items,
                                                        allow_unknown_status=allow_unknown_status)
            if item is not None:
                if item not in ret:
                    add_item(item)

            for item in self.agent.inventory.get_best_armorset(items=forced_items + items,
                                                               allow_unknown_status=allow_unknown_status):
                if item is not None:
                    if item not in ret:
                        add_item(item)

        for item in items:
            if item.is_ambiguous():
                if item.object in [
                        O.from_name('healing', nh.POTION_CLASS),
                        O.from_name('extra healing', nh.POTION_CLASS),
                        O.from_name('full healing', nh.POTION_CLASS)]:
                    add_item(item)

                if self.agent.character.role in [Character.RANGER, Character.ROGUE,
                                                 Character.SAMURAI, Character.TOURIST] and \
                        (item.is_launcher() or item.is_fired_projectile()):
                    add_item(item)

        for item in sorted(filter(lambda i: i.is_thrown_projectile(), items),
                           key=lambda i: -utils.calc_dps(*self.agent.character.get_melee_bonus(i))):
            add_item(item)

        for item in sorted(filter(lambda i: i.is_food(), items), key=lambda x: -x.nutrition_per_weight()):
            add_item(item)

        for item in sorted(items, key=lambda i: i.unit_weight()):
            if item.category in [nh.POTION_CLASS, nh.RING_CLASS, nh.AMULET_CLASS, nh.WAND_CLASS, nh.SCROLL_CLASS,
                                 nh.TOOL_CLASS, nh.COIN_CLASS]:
                add_item(item)

        categories = [nh.WEAPON_CLASS, nh.ARMOR_CLASS, nh.TOOL_CLASS, nh.FOOD_CLASS, nh.GEM_CLASS, nh.AMULET_CLASS,
                      nh.RING_CLASS, nh.COIN_CLASS, nh.POTION_CLASS, nh.SCROLL_CLASS, nh.SPBOOK_CLASS, nh.WAND_CLASS]
        for item in sorted(items, key=lambda i: i.unit_weight()):
            if item.category in categories:
                if item.status == Item.UNKNOWN:
                    add_item(item)

        return [ret.get(item, 0) for item in items]


class Milestone(IntEnum):
    FIND_GNOMISH_MINES = auto()
    FIND_SOKOBAN = auto()
    SOLVE_SOKOBAN = auto()
    GO_DOWN = auto() # TODO


class GlobalLogic:
    def __init__(self, agent):
        self.agent = agent
        self.milestone = Milestone(1)
        self.step_completion_log = {}  # Milestone -> (step, turn)

        self.item_priority = ItemPriority(self.agent)

    @utils.debug_log('solving sokoban')
    @Strategy.wrap
    def solve_sokoban_strategy(self):
        # TODO: refactor
        if not utils.isin(self.agent.current_level().objects, G.TRAPS).any():
            yield False
        yield True

        while 1:
            wall_map = utils.isin(self.agent.current_level().objects, G.WALL)
            for smap, answer in soko_solver.maps.items():
                sokomap = soko_solver.convert_map(smap)
                offset = np.array(min(zip(*wall_map.nonzero()))) - \
                         np.array(min(zip(*(sokomap.sokomap == soko_solver.WALL).nonzero())))
                mask = wall_map[offset[0] : offset[0] + sokomap.sokomap.shape[0],
                                offset[1] : offset[1] + sokomap.sokomap.shape[1]]
                if (mask & (sokomap.sokomap == soko_solver.WALL) == mask).all():
                    break
            else:
                assert 0, 'sokomap not found'

            possible_mimics = set()
            last_resort_move = None
            for (y, x), (dy, dx) in answer:
                boulder_map = utils.isin(self.agent.glyphs, G.BOULDER)
                mask = boulder_map[offset[0] : offset[0] + sokomap.sokomap.shape[0],
                                   offset[1] : offset[1] + sokomap.sokomap.shape[1]]
                ty, tx = offset[0] + y - dy, offset[1] + x - dx,
                soko_boulder_mask = sokomap.sokomap == soko_solver.BOULDER
                if self.agent.bfs()[ty, tx] != -1 and \
                        ((soko_boulder_mask | mask) == soko_boulder_mask).all() and \
                        self.agent.glyphs[ty + dy, tx + dx] in G.BOULDER:
                    self.agent.go_to(ty, tx, debug_tiles_args=dict(color=(255, 255, 255), is_path=True))
                    with self.agent.atom_operation():
                        direction = self.agent.calc_direction(ty, tx, ty + dy, tx + dx)
                        self.agent.direction(direction)
                        if 'You hear a monster behind the boulder.' in self.agent.message:
                            # TODO: go deal with that monster
                            self.agent.exploration.explore1(None).run()
                            # TODO: destroy bolder
                            # TODO: do the same in last resort move

                    possible_mimics = set()
                    last_resort_move = None

                    if not utils.isin(self.agent.current_level().objects, G.TRAPS).any():
                        return

                if (~soko_boulder_mask | mask).all():
                    if self.agent.bfs()[ty, tx] != -1 and \
                            self.agent.glyphs[ty + dy, tx + dx] in G.BOULDER and \
                            last_resort_move is None:
                        last_resort_move = (ty, tx, dy, dx)

                    if not possible_mimics:
                        for mim_y, mim_x in zip(*(~soko_boulder_mask & mask).nonzero()):
                            possible_mimics.add((mim_y + offset[0], mim_x + offset[1]))

                sokomap.move(y, x, dy, dx)

            # sokoban configuration is not in the list. Check for mimics
            with self.agent.env.debug_log('mimics'):
                with self.agent.env.debug_tiles(possible_mimics, color=(255, 0, 0, 128)):
                    for mim_y, mim_x in sorted(possible_mimics):
                        if (self.agent.bfs()[max(0, mim_y - 1) : mim_y + 2, max(0, mim_x - 1) : mim_x + 2] != -1).any():
                            self.agent.go_to(mim_y, mim_x, stop_one_before=True,
                                             debug_tiles_args=dict(color=(255, 0, 0), is_path=True))
                            for _ in range(3):
                                self.agent.search()

            # last resort move (sometimes we can see a mimic diagonally but cannot reach it)
            with self.agent.env.debug_log('last_resort'):
                if last_resort_move is not None:
                    ty, tx, dy, dx = last_resort_move
                    self.agent.go_to(ty, tx, debug_tiles_args=dict(color=(255, 255, 255), is_path=True))
                    self.agent.move(ty + dy, tx + dx)
                    continue

            assert 0, 'sakomap unsolvable'

    @Strategy.wrap
    def wait_out_unexpected_state_strategy(self):
        yielded = False
        while (
                self.agent.character.prop.blind or
                self.agent.character.prop.confusion or
                self.agent.character.prop.stun or
                self.agent.character.prop.hallu or
                self.agent.character.prop.polymorph):
            if not yielded:
                yield True
                yielded = True

            self.agent.direction('.')

        if not yielded:
            yield False

    @Strategy.wrap
    def indentify_items_on_altar(self):
        mask = utils.isin(self.agent.current_level().objects, G.ALTAR)
        if not mask.any():
            yield False

        dis = self.agent.bfs()
        mask &= dis != -1
        if not mask.any():
            yield False

        yield any((item.status == Item.UNKNOWN for item in self.agent.inventory.items
                   if item.can_be_dropped_from_inventory()))

        (ty, tx), *_ = zip(*(mask & (dis == dis[mask].min())).nonzero())
        self.agent.go_to(ty, tx)
        items_to_drop = [item for item in self.agent.inventory.items
                         if item.can_be_dropped_from_inventory() and item.status == Item.UNKNOWN]
        if not items_to_drop:
            raise AgentPanic('items to drop on altar vanished')
        self.agent.inventory.drop(items_to_drop)

    @Strategy.wrap
    def current_strategy(self):
        if self.milestone == Milestone.FIND_GNOMISH_MINES and \
                self.agent.current_level().dungeon_number == Level.GNOMISH_MINES:
            self.milestone = Milestone(int(self.milestone) + 1)
        elif self.milestone == Milestone.FIND_SOKOBAN and \
                self.agent.current_level().dungeon_number == Level.SOKOBAN:
            self.milestone = Milestone(int(self.milestone) + 1)
        elif self.milestone == Milestone.SOLVE_SOKOBAN and \
                self.agent.current_level().key() == (Level.SOKOBAN, 1):
            self.milestone = Milestone(int(self.milestone) + 1)

        go_to_strategy = lambda y, x: (
                self.agent.exploration.explore1(None)
                .preempt(self.agent, [self.agent.exploration.go_to_strategy(y, x)])
                .until(self.agent, lambda: (self.agent.blstats.y, self.agent.blstats.x) == (y, x))
        )

        # TODO: unconditioned indentify_items_on_altar on changing level
        yield from (
            (self.agent.exploration.go_to_level_strategy(
                Level.GNOMISH_MINES if self.milestone == Milestone.FIND_GNOMISH_MINES else Level.SOKOBAN, 1,
                go_to_strategy, self.agent.exploration.explore1(None)) \
            .before(utils.assert_strategy('end'))).strategy()
        )

    def global_strategy(self):
        return (
            self.current_strategy()
            .preempt(self.agent, [
                self.agent.exploration.explore1(0),
                self.agent.exploration.explore1(None)
                    .until(self.agent, lambda: self.agent.blstats.score >= 950 and
                                               self.agent.blstats.hitpoints >= 0.9 * self.agent.blstats.max_hitpoints)
            ])
            .preempt(self.agent, [
                #self.indentify_items_on_altar().every(50),
            ])
            .preempt(self.agent, [
                self.solve_sokoban_strategy()
                .condition(lambda: self.agent.current_level().dungeon_number == Level.SOKOBAN)
            ])
            .preempt(self.agent, [
                self.wait_out_unexpected_state_strategy(),
            ])
            .preempt(self.agent, [
                self.agent.eat1().every(5).condition(lambda: self.agent.blstats.hunger_state >= Hunger.NOT_HUNGRY),
                self.agent.eat_from_inventory().every(5),
            ]).preempt(self.agent, [
                self.agent.fight2(),
            ]).preempt(self.agent, [
                self.agent.emergency_strategy(),
            ])
        )
