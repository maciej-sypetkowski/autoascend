from enum import IntEnum, auto

import nle.nethack as nh
import numpy as np
from nle.nethack import actions as A

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
    MAX_NUMBER_OF_ITEMS = 26 * 2  # + coin slot
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
            if remaining_weight < 0 or len(ret) >= ItemPriority.MAX_NUMBER_OF_ITEMS:  # TODO: coin slot
                return

            how_many_already = ret.get(item, 0)
            max_to_add = int(remaining_weight // item.unit_weight())
            if count is not None:
                max_to_add = min(max_to_add, count)
            ret[item] = min(item.count, how_many_already + max_to_add)
            remaining_weight -= item.unit_weight() * (ret[item] - how_many_already)

        for allow_unknown_status in [False, True]:
            item = self.agent.inventory.get_best_melee_weapon(items=forced_items + items,
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
            if item.is_unambiguous():
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

        # TODO: take nh.COIN_CLASS once shopping is implemented.
        # You have to drop all coins not to be attacked by a vault guard

        for item in sorted(items, key=lambda i: i.unit_weight()):
            if item.category in [nh.POTION_CLASS, nh.RING_CLASS, nh.AMULET_CLASS, nh.WAND_CLASS, nh.SCROLL_CLASS,
                                 nh.TOOL_CLASS]:
                if not isinstance(item.objs[0], O.Container) or item.objs[0].desc is not None:
                    add_item(item)

        categories = [nh.WEAPON_CLASS, nh.ARMOR_CLASS, nh.TOOL_CLASS, nh.FOOD_CLASS, nh.GEM_CLASS, nh.AMULET_CLASS,
                      nh.RING_CLASS, nh.POTION_CLASS, nh.SCROLL_CLASS, nh.SPBOOK_CLASS, nh.WAND_CLASS]
        for item in sorted(items, key=lambda i: i.unit_weight()):
            if item.category in categories and (not isinstance(item.objs[0], O.Container) or item.objs[0].desc is not None):
                if item.status == Item.UNKNOWN:
                    add_item(item)

        return [ret.get(item, 0) for item in items]


class Milestone(IntEnum):
    BE_ON_FIRST_LEVEL = auto()
    FIND_GNOMISH_MINES = auto()
    FIND_MINETOWN = auto()
    FIND_SOKOBAN = auto()
    SOLVE_SOKOBAN = auto()
    FIND_MINES_END = auto()
    GO_DOWN = auto() # TODO


class GlobalLogic:
    def __init__(self, agent):
        self.agent = agent
        self.milestone = Milestone(1)
        self.step_completion_log = {}  # Milestone -> (step, turn)

        self.item_priority = ItemPriority(self.agent)

        self.oracle_level = None
        self.minetown_level = None

    def update(self):
        if not self.agent.character.prop.hallu:
            if utils.isin(self.agent.glyphs, G.ORACLE).any():
                if self.oracle_level is None:
                    self.oracle_level = self.agent.current_level().key()
                else:
                    assert self.oracle_level == self.agent.current_level().key()

            if self.agent.current_level().dungeon_number == Level.GNOMISH_MINES and \
                    utils.isin(self.agent.glyphs, G.SHOPKEEPER).any():
                if self.minetown_level is None:
                    self.minetown_level = self.agent.current_level().key()
                else:
                    assert self.minetown_level == self.agent.current_level().key()

    @utils.debug_log('solving sokoban')
    @Strategy.wrap
    def solve_sokoban_strategy(self):
        # TODO: refactor
        if not utils.isin(self.agent.current_level().objects, G.TRAPS).any():
            yield False
        yield True

        def push_bolder(ty, tx, dy, dx):
            while 1:
                self.agent.go_to(ty, tx, debug_tiles_args=dict(color=(255, 255, 255), is_path=True))
                with self.agent.atom_operation():
                    direction = self.agent.calc_direction(ty, tx, ty + dy, tx + dx)
                    self.agent.direction(direction)
                    message = self.agent.message

                if (self.agent.blstats.y, self.agent.blstats.x) == (ty, tx):
                    assert 'You hear a monster behind the boulder.' in message or \
                           'You try to move the boulder, but in vain.' in message, message
                    if self.agent.bfs()[ty + dy, tx + dx] != -1:
                        self.agent.go_to(ty + dy, tx + dx, debug_tiles_args=dict(color=(255, 255, 255), is_path=True))
                        continue
                    else:
                        pickaxe = None
                        for item in self.agent.inventory.items:
                            if item.is_unambiguous() and \
                                    item.object in [O.from_name('pick-axe'), O.from_name('dwarvish mattock')]:
                                pickaxe = item
                                break
                        if pickaxe is not None:
                            with self.agent.atom_operation():
                                self.agent.step(A.Command.APPLY)
                                self.agent.type_text(self.agent.inventory.items.get_letter(pickaxe))
                                self.agent.direction(direction)
                            return
                else:
                    return

                # TODO: not sure what to do
                self.agent.exploration.explore1(None).run()

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

                    soko_dis1 = sokomap.bfs()
                    sokomap.move(y, x, dy, dx)
                    soko_dis2 = sokomap.bfs()

                    # see points that will no longer be accessible
                    with self.agent.env.debug_log('checking for monsters'):
                        to_visit_mask = self.agent.bfs() != -1
                        to_visit_mask[offset[0] : offset[0] + sokomap.sokomap.shape[0],
                                      offset[1] : offset[1] + sokomap.sokomap.shape[1]] &= \
                                              (soko_dis1 != -1) & (soko_dis2 == -1)

                        with self.agent.env.debug_tiles(to_visit_mask, color=(255, 0, 0, 128)):
                            while to_visit_mask.any():
                                vy, vx = list(zip(*to_visit_mask.nonzero()))[0]
                                to_visit_mask[vy, vx] = 0

                                def clear_neighbors():
                                    to_visit_mask[self.agent.blstats.y, self.agent.blstats.x] = 0
                                    to_visit_mask[utils.isin(self.agent.glyphs, G.VISIBLE_FLOOR)] = 0
                                    return not to_visit_mask[vy, vx]

                                self.agent.go_to(vy, vx, callback=clear_neighbors)

                    push_bolder(ty, tx, dy, dx)

                    possible_mimics = set()
                    last_resort_move = None

                    if not utils.isin(self.agent.current_level().objects, G.TRAPS).any():
                        return

                else:
                    sokomap.move(y, x, dy, dx)

                if (~soko_boulder_mask | mask).all():
                    if self.agent.bfs()[ty, tx] != -1 and \
                            self.agent.glyphs[ty + dy, tx + dx] in G.BOULDER and \
                            last_resort_move is None:
                        last_resort_move = (ty, tx, dy, dx)

                    if not possible_mimics:
                        for mim_y, mim_x in zip(*(~soko_boulder_mask & mask).nonzero()):
                            possible_mimics.add((mim_y + offset[0], mim_x + offset[1]))


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
                    push_bolder(ty, tx, dy, dx)
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

    @utils.debug_log('identify_items_on_altar')
    @Strategy.wrap
    def identify_items_on_altar(self):
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

    @utils.debug_log('dip_for_excalibur')
    @Strategy.wrap
    def dip_for_excalibur(self):
        if self.agent.character.alignment != Character.LAWFUL or self.agent.blstats.experience_level < 5:
            yield False
        if self.agent.current_level().dungeon_number == Level.GNOMISH_MINES and \
                (self.minetown_level is None or self.agent.current_level().key() == self.minetown_level):
            yield False

        dis = self.agent.bfs()
        mask = utils.isin(self.agent.current_level().objects, G.FOUNTAIN) & (dis != -1)
        if not mask.any():
            yield False

        candidate = None
        for item in self.agent.inventory.items:
            if item.is_unambiguous() and item.object == O.from_name('long sword'):
                if item.dmg_bonus is not None: # TODO: better condition for excalibur existance
                    yield False
                candidate = item

        if candidate is None:
            yield False
        yield True

        self.agent.go_to(*list(zip(*mask.nonzero()))[0])

        # TODO: refactor
        with self.agent.atom_operation():
            self.agent.step(A.Command.DIP)
            self.agent.type_text(self.agent.inventory.items.get_letter(candidate))
            if 'What do you want to dip ' in self.agent.message and 'into?' in self.agent.message:
                raise AgentPanic('no fountain here')

    @Strategy.wrap
    def current_strategy(self):
        yield True
        while 1:
            if self.milestone == Milestone.BE_ON_FIRST_LEVEL:
                condition = lambda: self.agent.blstats.experience_level >= 7
                level = (Level.DUNGEONS_OF_DOOM, 1)

            elif self.milestone == Milestone.FIND_SOKOBAN:
                condition = lambda: self.agent.current_level().dungeon_number == Level.SOKOBAN
                level = (Level.SOKOBAN, 4)

            elif self.milestone == Milestone.FIND_GNOMISH_MINES:
                condition = lambda: self.agent.current_level().dungeon_number == Level.GNOMISH_MINES
                level = (Level.GNOMISH_MINES, 1)

            elif self.milestone == Milestone.FIND_MINETOWN:
                condition = lambda: self.minetown_level is not None
                level = (Level.GNOMISH_MINES, 4)  # TODO

            elif self.milestone == Milestone.SOLVE_SOKOBAN:
                # TODO: fix the condition, monster can destroy doors
                condition = lambda: self.agent.current_level().key() == (Level.SOKOBAN, 1) and \
                                    not utils.isin(self.agent.current_level().objects, G.DOOR_CLOSED).any()
                level = (Level.SOKOBAN, 1)

            elif self.milestone == Milestone.FIND_MINES_END:
                condition = lambda: self.agent.current_level().key() == (Level.GNOMISH_MINES, 9)  # TODO
                level = (Level.GNOMISH_MINES, 9)  # TODO

            else:
                # TODO
                condition = lambda: False
                level = (Level.DUNGEONS_OF_DOOM, 100)

            if condition():
                self.milestone = Milestone(int(self.milestone) + 1)
                continue


            exploration_strategy = lambda level: (
                Strategy(lambda: self.agent.exploration.explore1(level, trap_search_offset=1,
                    kick_doors=self.agent.current_level().dungeon_number != Level.GNOMISH_MINES).strategy())
                .preempt(self.agent, [
                    self.identify_items_on_altar().every(100),
                    self.dip_for_excalibur().condition(lambda: self.agent.blstats.experience_level >= 7).every(10),
                ])
            )

            go_to_strategy = lambda y, x: (
                exploration_strategy(None)
                .preempt(self.agent, [
                    self.agent.exploration.go_to_strategy(y, x).preempt(self.agent, [
                        self.agent.inventory.gather_items(),
                        self.identify_items_on_altar(),
                        self.dip_for_excalibur().condition(lambda: self.agent.blstats.experience_level >= 7),
                    ])
                ])
                .until(self.agent, lambda: (self.agent.blstats.y, self.agent.blstats.x) == (y, x))
            )

            (
                self.agent.exploration.go_to_level_strategy(*level, go_to_strategy, exploration_strategy(None))
                .before(self.agent.exploration.explore1(None, trap_search_offset=1))
                .preempt(self.agent, [
                    exploration_strategy(0),
                    exploration_strategy(None)
                    .until(self.agent, lambda: self.agent.blstats.hitpoints >= 0.8 * self.agent.blstats.max_hitpoints)
                ])
                .until(self.agent, condition)
            ).run()

    def global_strategy(self):
        return (
            self.current_strategy().repeat()
            .preempt(self.agent, [
                self.solve_sokoban_strategy()
                .condition(lambda: self.milestone == Milestone.SOLVE_SOKOBAN and
                                   self.agent.current_level().dungeon_number == Level.SOKOBAN)
            ])
            .preempt(self.agent, [
                self.wait_out_unexpected_state_strategy(),
            ])
            .preempt(self.agent, [
                self.agent.eat1().every(5).condition(lambda: self.agent.blstats.hunger_state >= Hunger.NOT_HUNGRY),
                self.agent.eat_from_inventory().every(5),
            ])
            .preempt(self.agent, [
                self.agent.fight2(),
            ])
            .preempt(self.agent, [
                self.agent.engulfed_fight(),
            ])
            .preempt(self.agent, [
                self.agent.emergency_strategy(),
            ])
        )
