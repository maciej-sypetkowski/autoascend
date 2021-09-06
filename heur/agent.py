import contextlib
import re
from collections import namedtuple
from functools import partial

import nle.nethack as nh
import numpy as np
from nle.nethack import actions as A

import rl_utils
import utils
from character import Character
from exceptions import AgentPanic, AgentFinished, AgentChangeStrategy
from exploration_logic import ExplorationLogic
from global_logic import GlobalLogic
from glyph import MON, C, Hunger, G, SHOP, ALL
from item import Inventory, Item, flatten_items
from level import Level
from monster_tracker import MonsterTracker
from strategy import Strategy

BLStats = namedtuple('BLStats',
                     'x y strength_percentage strength dexterity constitution intelligence wisdom charisma score hitpoints max_hitpoints depth gold energy max_energy armor_class monster_level experience_level experience_points time hunger_state carrying_capacity dungeon_number level_number prop_mask')


class Agent:
    def __init__(self, env, seed=0, verbose=False, panic_on_errors=False,
                 rl_model_to_train=None, rl_model_training_comm=(None, None)):
        self.env = env
        self.verbose = verbose
        self.rng = np.random.RandomState(seed)
        self.panic_on_errors = panic_on_errors
        self.all_panics = []
        self.rl_model_to_train = rl_model_to_train
        self.rl_model_training_comm = rl_model_training_comm

        self.on_update = []
        self.levels = {}
        self.score = 0
        self.step_count = 0
        self._observation = None  # this should be used in additional_action_iterator generators
        # single_{message,popup} should be used in additional_action_itertator generators.
        # (non-single) message & popup contain cummulated content
        self.message = self.single_message = ''
        self.popup = self.single_popup = []
        self._message_history = []
        self.cursor_pos = (0, 0)

        self._last_pet_seen = 0

        self.inventory = Inventory(self)
        self.character = Character(self)
        self.exploration = ExplorationLogic(self)
        self.global_logic = GlobalLogic(self)
        self.monster_tracker = MonsterTracker(self)

        self.last_bfs_dis = None
        self.last_bfs_step = None
        self.last_prayer_turn = None
        # self._previous_glyphs = None
        self._last_turn = -1
        self._inactivity_counter = 0
        self._is_updating_state = False

        self._no_step_calls = False

        self.turns_in_atom_operation = None
        self._atom_operation_allow_update = None

        self._is_reading_message_or_popup = False
        self._last_terrain_check = None
        self._forbidden_engrave_position = (-1, -1)

        self._init_fight3_model()

    @property
    def has_pet(self):
        return (self.blstats.time - self._last_pet_seen) <= 16

    @property
    def in_atom_operation(self):
        return self.turns_in_atom_operation is not None

    ######## CONVENIENCE FUNCTIONS

    @contextlib.contextmanager
    def disallow_step_calling(self):
        if self._no_step_calls:
            yield
            return

        try:
            self._no_step_calls = True
            yield
        finally:
            self._no_step_calls = False

    @contextlib.contextmanager
    def atom_operation(self, allow_update=False):
        assert not self._no_step_calls
        if self.turns_in_atom_operation is not None:
            # already in an atom operation
            old_allow_update = self._atom_operation_allow_update
            if old_allow_update is None:
                self._atom_operation_allow_update = allow_update
            else:
                assert old_allow_update or not allow_update
                self._atom_operation_allow_update = old_allow_update and allow_update
            try:
                yield
            finally:
                self._atom_operation_allow_update = old_allow_update
            return

        self.turns_in_atom_operation = 0
        self._atom_operation_allow_update = allow_update
        try:
            yield
        finally:
            self.turns_in_atom_operation = None
            self._atom_operation_allow_update = None

        self.update_state()

    @contextlib.contextmanager
    def panic_if_position_changes(self):
        y, x = self.blstats.y, self.blstats.x

        def f(self):
            if (y, x) != (self.blstats.y, self.blstats.x):
                raise AgentPanic('position changed')

        fun = partial(f, self)

        self.on_update.append(fun)

        try:
            yield
        finally:
            assert fun in self.on_update
            self.on_update.pop(self.on_update.index(fun))

    @contextlib.contextmanager
    def add_on_update(self, funcs):
        self.on_update.extend(funcs)

        try:
            yield
        finally:
            for f in funcs:
                self.on_update.pop(self.on_update.index(f))

    @contextlib.contextmanager
    def context_preempt(self, conditions):
        ids = []
        id2fun = {}
        for cond in conditions:
            def f(iden, cond=cond):
                if cond():
                    raise AgentChangeStrategy(iden, cond)

            fun = partial(f, id(f))
            assert id(f) not in id2fun
            id2fun[id(f)] = fun
            ids.append(id(f))
            self.on_update.append(fun)

        outcome = None
        for i, cond in enumerate(conditions):
            if cond():
                outcome = i
                break

        def outcome_f():
            nonlocal outcome
            return outcome

        try:
            yield outcome_f

        except AgentChangeStrategy as e:
            i = e.args[0]
            if i not in id2fun:
                raise
            outcome = ids.index(i)
        finally:
            for f in id2fun.values():
                self.on_update.pop(self.on_update.index(f))

        # check if less nested ChangeStategy is present
        self.call_update_functions()

    def preempt(self, strategies, func, first_func=None, continue_after_preemption=True):
        id2fun = {}
        for strategy in strategies:
            def f(iden, strategy):
                it = strategy.strategy()
                if next(it):
                    raise AgentChangeStrategy(iden, it)

            iden = (id(f), id(strategy))
            fun = partial(f, iden, strategy)
            assert iden not in id2fun
            id2fun[iden] = fun

        last_turn = 0

        call_update = True

        val = None

        last_step = self.step_count
        inactivity_counter = 0
        is_first = True
        while 1:
            inactivity_counter += 1
            if self.step_count != last_step:
                last_step = self.step_count
                inactivity_counter = 0
            assert inactivity_counter < 5, 'cyclic preempt'

            iterator = None
            try:
                with self.add_on_update(list(id2fun.values())):
                    if call_update:
                        call_update = False
                        self.call_update_functions(list(id2fun.values()))

                    f = (first_func or func) if is_first else func
                    if isinstance(f, Strategy):
                        val = f.run()
                    else:
                        val = f()
                    break

            except AgentChangeStrategy as e:
                i = e.args[0]
                if i not in id2fun:
                    raise
                iterator = e.args[1]

            if iterator is not None:
                try:
                    next(iterator)
                    assert 0, iterator
                except StopIteration:
                    pass

                if not continue_after_preemption:
                    break

            is_first = False

        return val

    ######## UPDATE FUNCTIONS

    def on_panic(self):
        self.check_terrain(force=True)
        self.inventory.on_panic()
        self.monster_tracker.on_panic()
        self.update_state()

    @staticmethod
    def _find_marker(lines, regex=re.compile(r"(--More--|\(end\)|\(\d+ of \d+\))")):
        """ Return (line, column) of markers:
        --More-- | (end) | (X of N)
        """
        if len(regex.findall(' '.join(lines))) > 1:
            raise ValueError('Too many markers')

        result, marker_type = None, None
        for i, line in enumerate(lines):
            res = regex.findall(line)
            if res:
                assert len(res) == 1
                j = line.find(res[0])
                result, marker_type = (i, j), res[0]
                break

        if result is not None and result[1] == 1:
            result = (result[0], 0)  # e.g. for known items view
        return result, marker_type

    def get_message_and_popup(self, obs):
        """ Uses MORE action to get full popup and/or message.
        """

        message = bytes(obs['message']).decode().replace('\0', ' ').replace('\n', '').strip()
        if message.endswith('--More--'):
            # FIXME: It seems like in this case the environment doesn't expect additional input,
            #        but I'm not 100% sure, so it's too risky to change it, because it could stall everything.
            #        With the current implementation, in the worst case, we'll get "Unknown command ' '".
            message = message[:-len('--More--')]

        # assert '\n' not in message and '\r' not in message
        popup = []

        lines = [bytes(line).decode().replace('\0', ' ').replace('\n', '') for line in obs['tty_chars']]
        marker_pos, marker_type = self._find_marker(lines)

        if marker_pos is None:
            return message, popup, True

        pref = ''
        message_lines_count = 0
        if message:
            for i, line in enumerate(lines[:marker_pos[0] + 1]):
                if i == marker_pos[0]:
                    line = line[:marker_pos[1]]
                message_lines_count += 1
                pref += line.strip()

                # I'm not sure when the new line character in broken messages should be a space and when be ignored.
                # '#' character (and others) occasionally occurs at the beginning of the broken line and isn't in
                # the message. Sometimes the message on the screen lacks last '.'.
                replace_func = lambda x: ''.join((c for c in x if c.isalnum()))
                if replace_func(pref) == replace_func(message):
                    break
            else:
                if marker_pos[0] == 0:
                    elems1 = [s for s in message.split() if s]
                    elems2 = [s for s in pref.split() if s]
                    assert len(elems1) < len(elems2) and elems2[-len(elems1):] == elems1, (elems1, elems2)
                    return pref, popup, False
                raise ValueError(f"Message:\n{repr(message)}\ndoesn't match the screen:\n{repr(pref)}")

        # cut out popup
        for l in lines[message_lines_count:marker_pos[0]] + [lines[marker_pos[0]][:marker_pos[1]]]:
            l = l[marker_pos[1]:].strip()
            if l:
                popup.append(l)

        return message, popup, False

    def update_message_and_popup(self, obs):
        if self._is_reading_message_or_popup:
            message_prefix = self.message + (' ' if self.message else '')
            popup_prefix = self.popup
        else:
            message_prefix = ''
            popup_prefix = []

        self.single_message, self.single_popup, done = self.get_message_and_popup(obs)
        self.single_message = self.single_message.strip()
        self.single_popup = [p.strip() for p in self.single_popup]

        self.message = message_prefix + self.single_message
        self.popup = popup_prefix + self.single_popup
        return done

    def step(self, action, additional_action_iterator=None):
        if self._no_step_calls:
            raise ValueError("Shouldn't call step now")

        if isinstance(action, str):
            assert len(action) == 1
            action = A.ACTIONS[A.ACTIONS.index(ord(action))]
        observation, reward, done, info = self.env.step(action)
        observation = {k: v.copy() for k, v in observation.items()}
        self.step_count += 1
        self.score += reward

        self.cursor_pos = (observation['tty_cursor'][0] - 1, observation['tty_cursor'][1])

        if done:
            raise AgentFinished()

        self.update(observation, additional_action_iterator)

    def update(self, observation, additional_action_iterator=None):
        self._observation = observation
        done = self.update_message_and_popup(observation)

        self._is_reading_message_or_popup = True
        if additional_action_iterator is not None:
            is_next_action = True
            try:
                next_action = next(additional_action_iterator)
            except StopIteration:
                is_next_action = False

            if is_next_action:
                self.step(next_action, additional_action_iterator)
                return

        # FIXME: self.update_state() won't be called on all states sometimes.
        #        Otherwise there are problems with atomic operations.
        if not done or observation['misc'][2]:
            self.step(A.TextCharacters.SPACE)
            return

        if "You may wish for an object." in self.message:
            # TODO: wishing strategy
            # TODO: assume wished item as blessed
            self.step('b', iter('lessed greased +2 gray dragon scale mail\r'))
            return

        if observation['misc'][1]:  # entering text
            self.step(A.Command.ESC)
            return

        if 'Where do you want to be teleported?' in self.message:
            # TODO: teleport control
            self.step(A.Command.ESC)
            return

        if b'[yn]' in bytes(observation['tty_chars'].reshape(-1)):
            self.type_text('y')
            return

        self._is_reading_message_or_popup = False
        self._message_history.append(self.message)

        # should_update = True

        # if self.turns_in_atom_operation is not None:
        #     should_update = False
        #     # if any([(self.last_observation[key] != observation[key]).any()
        #     #         for key in ['glyphs', 'blstats', 'inv_strs', 'inv_letters', 'inv_oclasses', 'inv_glyphs']]):
        #     #     self.turns_in_atom_operation += 1
        #     # assert self.turns_in_atom_operation in [0, 1]

        self.last_observation = observation

        self.blstats = BLStats(*self.last_observation['blstats'])
        self.glyphs = self.last_observation['glyphs']

        self._inactivity_counter += 1
        if self._last_turn != self.blstats.time:
            self._last_turn = self.blstats.time
            self._inactivity_counter = 0
        assert self._inactivity_counter < 200, ('turn inactivity', sorted(set(self._message_history[-50:])))

        self.update_state(allow_update=self._atom_operation_allow_update or not self.in_atom_operation,
                          allow_callbacks=not self.in_atom_operation)

    def update_state(self, allow_update=True, allow_callbacks=True):
        assert not self._no_step_calls
        if self._is_updating_state:
            return
        self._is_updating_state = True
        message = self.message
        popup = self.popup

        try:
            if allow_update:
                # functions that are allowed to call state unchanging steps
                for func in [self.character.update, self.inventory.update, self.monster_tracker.update,
                            partial(self.check_terrain, force=False), self.update_level,
                            self.global_logic.update]:
                    func()
                    self.message = message
                    self.popup = popup

            if allow_callbacks:
                self.call_update_functions()
        finally:
            self._is_updating_state = False

    def call_update_functions(self, funcs=None):
        if funcs is None:
            funcs = self.on_update
        assert all((func in self.on_update for func in funcs))

        with self.disallow_step_calling():
            for func in funcs:
                func()

    def _update_level_items(self):
        level = self.current_level()

        level.items[self.blstats.y, self.blstats.x] = self.inventory.items_below_me
        level.item_count[self.blstats.y, self.blstats.x] = len(self.inventory.items_below_me)

        # TODO: optimize
        ignore_mask = utils.isin(self.glyphs, G.MONS, G.PETS)  # TODO: effects, etc
        item_mask = level.item_count != 0
        mask = item_mask & ~ignore_mask
        level.item_disagreement_counter[~mask] = 0
        for y, x in zip(*mask.nonzero()):
            if (level.item_count[y, x] >= 2) == ((self.last_observation['specials'][y, x] & nh.MG_OBJPILE) > 0):
                glyphs = (glyph for item in level.items[y, x] for glyph in item.display_glyphs())
                if self.glyphs[y, x] in glyphs:
                    level.item_disagreement_counter[y, x] = 0
                    continue

            level.item_disagreement_counter[y, x] += 1
            if level.item_disagreement_counter[y, x] > 3:
                level.item_disagreement_counter[y, x] = 0
                level.items[y, x] = ()
                level.item_count[y, x] = 0

    def _update_level_shops(self):
        level = self.current_level()

        shop_type = None
        matches = re.search(f"Welcome( again)? to [a-zA-Z' ]*({'|'.join(SHOP.name2id.keys())})!", self.message)
        if matches is not None:
            shop_name = matches.groups()[1]
            assert shop_name in SHOP.name2id, shop_name
            shop_type = SHOP.name2id[shop_name]

        shopkeepers = list(
            zip(*(utils.isin(self.glyphs, G.SHOPKEEPER) & self.monster_tracker.peaceful_monster_mask).nonzero()))
        for y, x in shopkeepers:
            wall_mask = utils.isin(level.objects, G.WALL)
            entry = ((utils.translate(wall_mask, 1, 0) & utils.translate(wall_mask, -1, 0)) |
                     (utils.translate(wall_mask, 0, 1) & utils.translate(wall_mask, 0, -1))) & \
                    level.walkable
            walkable = level.walkable & ~entry
            mask = utils.bfs(y, x, walkable=walkable, walkable_diagonally=walkable, can_squeeze=False) != -1
            mask = utils.dilate(mask, radius=1)

            level.shop[mask] = True
            if mask[self.blstats.y, self.blstats.x] and shop_type is not None:
                level.shop_type[mask] = shop_type
            level.shop_interior[mask & ~utils.dilate(entry, radius=1, with_diagonal=False)] = True

    def update_level(self):
        if utils.isin(self.glyphs, G.SWALLOW).any():
            return

        if utils.any_in(self.glyphs, G.PETS):
            self._last_pet_seen = self.blstats.time

        level = self.current_level()

        # if self._previous_glyphs is None or (self._previous_glyphs != self.last_observation['glyphs']).any():
        #     self._previous_glyphs = self.last_observation['glyphs']

        mask = utils.isin(self.glyphs, G.FLOOR, G.STAIR_UP, G.STAIR_DOWN, G.DOOR_OPENED, G.TRAPS,
                          G.ALTAR, G.FOUNTAIN)
        level.walkable[mask] = True
        level.seen[mask] = True
        level.objects[mask] = self.glyphs[mask]

        mask = utils.isin(self.glyphs, G.MONS, G.PETS, G.BODIES, G.OBJECTS, G.STATUES)
        level.seen[mask] = True
        level.walkable[mask & (level.objects == -1)] = True

        mask = utils.isin(self.glyphs, G.WALL, G.DOOR_CLOSED, G.BARS)
        level.seen[mask] = True
        level.objects[mask] = self.glyphs[mask]
        level.walkable[mask] = False

        self._update_level_items()
        self._update_level_shops()

        for y, x in zip(*utils.isin(level.objects, G.ALTAR).nonzero()):
            if (y, x) not in level.altars:
                level.altars[y, x] = Character.UNKNOWN

        level.was_on[self.blstats.y, self.blstats.x] = True

        for y, x in self.neighbors(self.blstats.y, self.blstats.x, shuffle=False):
            if self.glyphs[y, x] in G.STONE:
                level.seen[y, x] = True
                level.objects[y, x] = self.glyphs[y, x]
                level.walkable[y, x] = False  # necessary for the exit route from vaults

    ######## TRIVIAL HELPERS

    def current_level(self):
        key = (self.blstats.dungeon_number, self.blstats.level_number)
        if key not in self.levels:
            self.levels[key] = Level(*key)
        return self.levels[key]

    @staticmethod
    def calc_direction(from_y, from_x, to_y, to_x, allow_nonunit_distance=False):
        if allow_nonunit_distance:
            assert from_y == to_y or from_x == to_x or \
                   abs(from_y - to_y) == abs(from_x - to_x), ((from_y, from_x), (to_y, to_x))
            to_y = from_y + np.sign(to_y - from_y)
            to_x = from_x + np.sign(to_x - from_x)

        assert abs(from_y - to_y) <= 1 and abs(from_x - to_x) <= 1, ((from_y, from_x), (to_y, to_x))

        ret = ''
        if to_y == from_y + 1: ret += 's'
        if to_y == from_y - 1: ret += 'n'
        if to_x == from_x + 1: ret += 'e'
        if to_x == from_x - 1: ret += 'w'
        if ret == '': ret = '.'

        return ret

    ######## TRIVIAL ACTIONS

    def check_terrain(self, force):
        if force or self._last_terrain_check is None or self.blstats.time - self._last_terrain_check > 50:
            self._last_terrain_check = self.blstats.time
            with self.atom_operation():
                self.type_text('#te')
                self.step(A.MiscAction.MORE, iter('b'))
                self.update_level()
                self.step(A.Command.ESC)

    def wield_best_melee_weapon(self):
        # TODO: move to inventory
        item = self.inventory.get_best_melee_weapon()
        if item != self.inventory.items.main_hand:
            return self.inventory.wield(item)
        return False

    def type_text(self, text):
        with self.atom_operation():
            for char in text:
                self.step(char)

    def eat(self):  # TODO: eat what
        with self.atom_operation():
            self.step(A.Command.EAT)
            if ' eat it? [ynq]' in self.message or \
                    ' eat one? [ynq]' in self.message:
                self.type_text('y')
                return True
            if "What do you want to eat?" in self.message or \
                    "You don't have anything to eat." in self.message:
                raise AgentPanic('no food is lying here')
            assert 0, self.message

    def is_safe_to_pray(self):
        return (
                (self.last_prayer_turn is None and self.blstats.time > 300) or
                (self.last_prayer_turn is not None and self.blstats.time - self.last_prayer_turn > 900)
        )

    def pray(self):
        self.step(A.Command.PRAY)
        self.last_prayer_turn = self.blstats.time
        # TODO: return value
        return True

    def open_door(self, y, x):
        with self.panic_if_position_changes():
            assert self.glyphs[y, x] in G.DOOR_CLOSED
            self.direction(y, x)
            self.current_level().door_open_count[y, x] += 1
            return self.glyphs[y, x] not in G.DOOR_CLOSED

    def fight(self, y, x):
        with self.panic_if_position_changes():
            assert self.glyphs[y, x] in G.MONS or self.glyphs[y, x] in G.INVISIBLE_MON or \
                   self.glyphs[y, x] in G.SWALLOW
            self.direction(y, x)
        return True

    def zap(self, item, direction):
        with self.atom_operation():
            self.step(A.Command.ZAP)
            self.type_text(self.inventory.items.get_letter(item))
            self.direction(direction)
        return True

    def fire(self, item, direction):
        if self.character.prop.polymorph:
            # TODO: throwing is not possible if you don't have hands
            # it may be possible depending on creature
            return False

        with self.atom_operation():
            self.step(A.Command.THROW)
            self.type_text(self.inventory.items.get_letter(item))
            self.direction(direction)
        return True

    def kick(self, y, x=None):
        with self.panic_if_position_changes():
            with self.atom_operation():
                self.step(A.Command.KICK)
                self.direction(self.calc_direction(self.blstats.y, self.blstats.x, y, x))

    def search(self, max_count=1):
        assert max_count >= 1
        with self.panic_if_position_changes():
            with self.atom_operation():
                if max_count > 1:
                    self.type_text(str(max_count))
                self.step(A.Command.SEARCH)
                # TODO: estimate the real number of searches
                self.current_level().search_count[self.blstats.y, self.blstats.x] += max_count
                if 'You find ' in self.message:
                    self.check_terrain(force=True)
        return True

    def direction(self, y, x=None):
        if x is not None:
            dir = self.calc_direction(self.blstats.y, self.blstats.x, y, x)
        else:
            dir = y

        action = {
            'n': A.CompassDirection.N, 's': A.CompassDirection.S,
            'e': A.CompassDirection.E, 'w': A.CompassDirection.W,
            'ne': A.CompassDirection.NE, 'se': A.CompassDirection.SE,
            'nw': A.CompassDirection.NW, 'sw': A.CompassDirection.SW,
            '>': A.MiscDirection.DOWN, '<': A.MiscDirection.UP,
            '.': A.MiscDirection.WAIT,
        }[dir]

        self.step(action)
        return True

    def move(self, y, x=None):
        if x is not None:
            dir = self.calc_direction(self.blstats.y, self.blstats.x, y, x)
        else:
            dir = y

        expected_y = self.blstats.y + ('s' in dir) - ('n' in dir)
        expected_x = self.blstats.x + ('e' in dir) - ('w' in dir)

        if (expected_y != self.blstats.y or expected_x != self.blstats.x) \
                and self.monster_tracker.monster_mask[expected_y, expected_x]:
            # TODO: consider handling it in different way, since this situation is sometimes expected
            raise AgentPanic(f'Monster on a next tile when moving: ({expected_y},{expected_x})')

        # TODO: portals
        if dir in ['<', '>']:
            level = self.current_level()
            with self.atom_operation():
                self.direction(dir)
                assert self.current_level().key() != level.key(), self.message
                level.stair_destination[expected_y, expected_x] = \
                    (self.current_level().key(), (self.blstats.y, self.blstats.x))
                # TODO: one way portals (elemental and astral planes)
                self.current_level().stair_destination[
                    (self.blstats.y, self.blstats.x)] = (level.key(), (expected_y, expected_x))

        else:
            self.direction(dir)

            if self.blstats.y != expected_y or self.blstats.x != expected_x:
                raise AgentPanic(f'agent position do not match after "move": '
                                 f'expected ({expected_y}, {expected_x}), got ({self.blstats.y}, {self.blstats.x})')

    def can_engrave(self):
        return (self.blstats.y, self.blstats.x) != self._forbidden_engrave_position

    def engrave(self, text):
        assert '\r' not in text
        ret = False
        def gen():
            nonlocal ret
            if 'What do you want to write with?' not in self.single_message:
                self._forbidden_engrave_position = (self.blstats.y, self.blstats.x)
                yield A.Command.ESC
                return
            yield '-'
            if 'Do you want to add to the current engraving?' in self.single_message:
                yield 'n'
            while self._observation['misc'][2]:
                yield ' '
            if 'What do you want to write in the dust here?' not in self.single_message:
                self._forbidden_engrave_position = (self.blstats.y, self.blstats.x)
                yield A.Command.ESC
                return
            yield from text
            yield '\r'
            ret = True

        with self.atom_operation():
            self.step(A.Command.ENGRAVE, gen())
            self.inventory.get_items_below_me()
        return ret

    ######## NON-TRIVIAL HELPERS

    def neighbors(self, y, x, shuffle=True, diagonal=True):
        ret = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                if not diagonal and abs(dy) + abs(dx) > 1:
                    continue
                ny = y + dy
                nx = x + dx
                if 0 <= ny < C.SIZE_Y and 0 <= nx < C.SIZE_X:
                    ret.append((ny, nx))

        if shuffle:
            self.rng.shuffle(ret)
            pass

        return ret

    def bfs(self, y=None, x=None):
        if y is None:
            y = self.blstats.y
        if x is None:
            x = self.blstats.x

        if self.last_bfs_step == self.step_count and y == self.blstats.y and x == self.blstats.x:
            return self.last_bfs_dis.copy()

        level = self.current_level()

        walkable = level.walkable & ~utils.isin(self.glyphs, G.BOULDER) & \
                   ~self.monster_tracker.peaceful_monster_mask & \
                   ~utils.isin(level.objects, G.TRAPS)

        for my, mx in list(zip(*np.nonzero(utils.isin(self.glyphs, G.MONS)))):
            mon = MON.permonst(self.glyphs[my][mx])
            import fight_heur
            if mon.mname in fight_heur.ONLY_RANGED_SLOW_MONSTERS:
                walkable[my, mx] = False

        dis = utils.bfs(y, x,
                        walkable=walkable,
                        walkable_diagonally=walkable & ~utils.isin(level.objects, G.DOORS) & (level.objects != -1),
                        can_squeeze=self.inventory.items.total_weight <= 600 and \
                                    self.current_level().dungeon_number != Level.SOKOBAN,
                        )

        if y == self.blstats.y and x == self.blstats.x:
            self.last_bfs_dis = dis
            self.last_bfs_step = self.step_count

        return dis.copy()

    def path(self, from_y, from_x, to_y, to_x, dis=None):
        if from_y == to_y and from_x == to_x:
            return [(to_y, to_x)]

        if dis is None:
            dis = self.bfs(from_y, from_x)

        assert dis[to_y, to_x] != -1

        # FIXME: currently the path can lead through diagonally inwalkable tiles.
        #        The path is the shortest possible, so the agent is guaranteed to
        #        unstuck itself eventually (usually a few panic exceptions) if that happens

        cur_y, cur_x = to_y, to_x
        path_rev = [(cur_y, cur_x)]
        while cur_y != from_y or cur_x != from_x:
            for y, x in self.neighbors(cur_y, cur_x):
                if dis[y, x] == dis[cur_y, cur_x] - 1 and dis[y, x] >= 0:
                    path_rev.append((y, x))
                    cur_y, cur_x = y, x
                    break
            else:
                assert 0

        assert dis[cur_y, cur_x] == 0 and from_y == cur_y and from_x == cur_x
        path = path_rev[::-1]
        assert path[0] == (from_y, from_x) and path[-1] == (to_y, to_x)
        return path

    ######## NON-TRIVIAL ACTIONS

    def _fast_go_to(self, y, x):
        with self.atom_operation():
            self.step(A.Command.TRAVEL)
            py, px = self.cursor_pos
            while py != y or px != x:
                dy, dx = np.sign(y - py), np.sign(x - px)
                self.direction(self.calc_direction(py, px, py + dy, px + dx))
                py += dy
                px += dx
                assert (py, px) == self.cursor_pos
            self.direction('.')

        if (self.blstats.y, self.blstats.x) == (y, x):
            return True

    def go_to(self, y, x, stop_one_before=False, max_steps=None, debug_tiles_args=None, callback=lambda: False, fast=False):
        assert not stop_one_before or (self.blstats.y != y or self.blstats.x != x)
        assert max_steps is None or not fast

        if stop_one_before and self.bfs()[y, x] == -1:
            dis = self.bfs()
            best_p = None
            for ny, nx in self.neighbors(y, x):
                if dis[ny, nx] != -1 and (best_p is None or dis[best_p] > dis[ny, nx]):
                    best_p = ny, nx
            if best_p is None:
                assert 0, 'no achievable neighbor'
            y, x = best_p
            stop_one_before = False

        assert self.bfs()[y, x] != -1

        if callback():
            return
        steps_taken = 0
        cont = True
        while cont and (self.blstats.y, self.blstats.x) != (y, x):
            dis = self.bfs()
            if dis[y, x] == -1:
                raise AgentPanic('end point is no longer accessible')
            path = self.path(self.blstats.y, self.blstats.x, y, x)
            orig_path = path
            path = path[1:]
            if stop_one_before:
                path = path[:-1]

            if fast and len(path) > 2:
                my_y, my_x = self.blstats.y, self.blstats.x
                self._fast_go_to(*path[-1])
                if (my_y, my_x) != (self.blstats.y, self.blstats.x):
                    continue

            with self.env.debug_tiles(orig_path, **debug_tiles_args) \
                    if debug_tiles_args is not None else contextlib.suppress():
                for y, x in path:
                    if self.monster_tracker.peaceful_monster_mask[y, x]:
                        cont = True
                        break
                    if not self.current_level().walkable[y, x]:
                        cont = True
                        break
                    self.move(y, x)
                    if callback():
                        return
                    steps_taken += 1
                    if max_steps is not None and steps_taken >= max_steps:
                        cont = False
                        break
                else:
                    cont = False

    ######## LOW-LEVEL STRATEGIES

    @utils.debug_log('ranged_stance1')
    def ranged_stance1(self):
        while True:
            valid_combinations = self.inventory.get_ranged_combinations(throwing=False)

            # TODO: select best combination
            if not valid_combinations:
                self.wield_best_melee_weapon()
                return False

            # TODO: consider using monster information to select the best combination
            launcher, ammo = valid_combinations[0]
            if not launcher.equipped:
                if not self.inventory.wield(launcher):
                    return False
                continue

            for _, y, x, _, _ in self.get_visible_monsters():
                if (self.blstats.y == y or self.blstats.x == x or abs(self.blstats.y - y) == abs(self.blstats.x - x)):
                    # TODO: don't shoot pet !
                    # TODO: limited range
                    with self.env.debug_tiles([[y, x]], (0, 0, 255, 100)):
                        dir = self.calc_direction(self.blstats.y, self.blstats.x, y, x, allow_nonunit_distance=True)
                        if not self.fire(ammo, dir):
                            return False
                        break
            else:
                return False

    def get_visible_monsters(self):
        """ Returns list of tuples (distance, y, x, monster)
        """
        mask = self.monster_tracker.monster_mask & ~self.monster_tracker.peaceful_monster_mask
        if not mask.any():
            return []

        dis = self.bfs()
        ret = []
        for y, x in zip(*mask.nonzero()):
            if (dis[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2] != -1).any():
                if self.glyphs[y, x] == nh.GLYPH_INVISIBLE or \
                        not MON.is_monster(self.glyphs[y, x]):  # TODO: some ghost are not visible in glyphs (?)
                    if utils.adjacent((self.blstats.y, self.blstats.x), (y, x)):
                        class dummy_permonst:
                            mname = 'unknown'
                            mlet = '0'

                        ret.append((dis[y][x], y, x, dummy_permonst(), self.glyphs[y][x]))
                else:
                    ret.append((dis[y][x], y, x, MON.permonst(self.glyphs[y][x]), self.glyphs[y][x]))
        ret.sort()
        return ret

    def should_keep_distance(self, monsters):
        ret = np.zeros(len(monsters), dtype=bool)
        for i, (dis, y, x, mon, _) in enumerate(monsters):
            if max(abs(x - self.blstats.x), abs(y - self.blstats.y)) not in (1, 2):
                continue
            # if mon.mname == 'goblin':
            #     ret[i] = True
            if self.blstats.hitpoints <= 8:
                ret[i] = True
        return ret

    @utils.debug_log('keep_distance')
    def keep_distance(self, monsters, keep_distance):
        if not keep_distance.any():
            return False
        monsters = [m for m, k in zip(monsters, keep_distance) if k]
        bad_tiles = ~self.current_level().walkable
        for _, y, x, _, _ in monsters:
            for y1 in (y - 1, y, y + 1):
                for x1 in (x - 1, x, x + 1):
                    if 0 <= y1 <= bad_tiles.shape[0] and 0 <= x1 <= bad_tiles.shape[1]:
                        bad_tiles[y1, x1] = True

        with self.env.debug_tiles(bad_tiles, color=(255, 0, 0, 64)):
            for y1 in (y - 1, y, y + 1):
                for x1 in (x - 1, x, x + 1):
                    if 0 <= y1 <= bad_tiles.shape[0] and 0 <= x1 <= bad_tiles.shape[1]:
                        if not bad_tiles[y1][x1]:
                            with self.env.debug_tiles([[y1, x1]], color=(0, 255, 0, 64)):
                                self.move(y1, x1)
                            return True

        return False

    def _init_fight3_model(self):
        self._fight3_model = rl_utils.RLModel({
                'walkable': ((7, 7), bool),
                'monster_mask': ((7, 7), bool),
            },
            [(y, x) for y in [-1, 0, 1] for x in [-1, 0, 1]],
            train=self.rl_model_to_train == 'fight3',
            training_comm=self.rl_model_training_comm,
        )

    @utils.debug_log('fight3')
    @Strategy.wrap
    def fight3(self):
        yielded = False
        while 1:
            monsters = self.get_visible_monsters()

            # get only monsters with path to them
            monsters = [m for m in monsters if m[0] != -1]

            if not monsters or all(dis > 7 for dis, *_ in monsters):
                if not yielded:
                    yield False
                return

            if not yielded:
                yielded = True
                yield True
                self.character.parse_enhance_view()

            if self.wield_best_melee_weapon():
                continue

            dis = self.bfs()
            level = self.current_level()
            walkable = level.walkable & ~utils.isin(self.glyphs, G.BOULDER) & \
                       ~self.monster_tracker.peaceful_monster_mask & \
                       ~utils.isin(level.objects, G.TRAPS)

            radius_y = self._fight3_model.observation_def['walkable'][0][0] // 2
            radius_x = self._fight3_model.observation_def['walkable'][0][1] // 2
            y1, y2, x1, x2 = self.blstats.y - radius_y, self.blstats.y + radius_y + 1, \
                             self.blstats.x - radius_x, self.blstats.x + radius_x + 1

            observation = {
                'walkable': utils.slice_with_padding(walkable, y1, y2, x1, x2),
                'monster_mask': utils.slice_with_padding(self.monster_tracker.monster_mask, y1, y2, x1, x2),
            }
            actions = [(y, x) for y, x in self._fight3_model.action_space
                       if 0 <= self.blstats.y + y < C.SIZE_Y and 0 <= self.blstats.x + x < C.SIZE_X \
                               and level.walkable[self.blstats.y + y, self.blstats.x + x]]

            off_y, off_x = self._fight3_model.choose_action(observation, actions)

            y, x = self.blstats.y + off_y, self.blstats.x + off_x
            if self.monster_tracker.monster_mask[y, x]:
                # TODO: copied from `flight1`
                mon_glyph = self.glyphs[y, x]
                try:
                    self.fight(y, x)
                finally:  # TODO: what if panic?
                    if nh.glyph_is_body(self.glyphs[y, x]) \
                            and self.glyphs[y, x] - nh.GLYPH_BODY_OFF == nh.glyph_to_mon(mon_glyph):
                        self.current_level().corpse_age[y, x] = self.blstats.time
            else:
                self.move(y, x)

    @utils.debug_log('fight2')
    @Strategy.wrap
    def fight2(self):
        import fight_heur
        yielded = False
        wait_counter = 0
        while 1:
            monsters = self.get_visible_monsters()
            only_ranged_slow_monsters = all([monster[3].mname in fight_heur.ONLY_RANGED_SLOW_MONSTERS
                                             and not fight_heur.consider_melee_only_ranged_if_hp_full(self, monster)
                                             for monster in monsters])

            dis = self.bfs()

            if not monsters or all(dis > 7 for dis, *_ in monsters) or \
                    (only_ranged_slow_monsters and not self.inventory.get_ranged_combinations() and np.sum(
                        dis != -1) > 1):
                if wait_counter:
                    self.search()
                    wait_counter -= 1
                    continue
                if not yielded:
                    yield False
                return

            if not yielded:
                yielded = True
                yield True
                self.character.parse_enhance_view()

            move_priority_heatmap, actions = fight_heur.get_priorities(self)

            best_move_score, best_x, best_y, possible_move_to = self._fight2_get_best_move(dis, move_priority_heatmap)

            if self.character.prop.polymorph:
                actions = list(filter(lambda x: x[1] != 'ranged', actions))
            best_action = max(actions, key=lambda x: x[0]) if actions else None

            if best_y is None and best_action is None:
                assert 0, 'No possible action available during fight2'

            with self.env.debug_tiles(move_priority_heatmap, color='turbo', is_heatmap=True):
                def action_str(a):
                    if a[1] == 'pickup':
                        return f'{a[0]}{a[1][0]}:{len(a[2])}'
                    elif a[1] == 'zap':
                        wand = a[4]
                        letter = self.inventory.items.get_letter(wand)
                        return f'{a[0]}z{letter}:{a[2]},{a[3]}'
                    else:
                        return f'{a[0]}{a[1][0]}:{a[2]},{a[3]}'

                actions_str = '|'.join([action_str(a) for a in sorted(actions, key=lambda x: x[0])])
                with self.env.debug_log(actions_str + f'|{best_move_score}|' + '|'.join(map(str, possible_move_to))):
                    wait_counter = self._fight2_perform_action(best_action, best_move_score, best_x, best_y,
                                                               wait_counter)

    def _fight2_get_best_move(self, dis, move_priority_heatmap):
        mask = ~np.isnan(move_priority_heatmap)
        assert mask.any()
        move_priority_heatmap[~mask] = np.min(move_priority_heatmap[mask]) - 1
        adjacent = dis == 1
        assert np.sum(adjacent) <= 8, np.sum(adjacent)
        possible_move_to = []
        if adjacent.any():
            possible_move_to = list(
                zip(*np.nonzero((move_priority_heatmap == np.max(move_priority_heatmap[adjacent])) & adjacent)))
        assert len(possible_move_to) <= 8
        best_y, best_x = None, None
        if possible_move_to:
            best_y, best_x = possible_move_to[self.rng.randint(0, len(possible_move_to))]
            if move_priority_heatmap[best_y, best_x] < -2 ** 15:
                best_y, best_x = None, None
        best_move_score = None
        if best_y is not None:
            best_move_score = move_priority_heatmap[best_y, best_x]
        move_priority_heatmap[~mask] = float('nan')
        return best_move_score, best_x, best_y, possible_move_to

    def _fight2_perform_action(self, best_action, best_move_score, best_x, best_y, wait_counter):
        if best_action is None or (best_y is not None and best_move_score > best_action[0]):
            with self.env.debug_tiles([[self.blstats.y, self.blstats.x],
                                       [best_y, best_x]], color=(0, 255, 0), is_path=True):
                self.move(best_y, best_x)
                wait_counter = 5
                return wait_counter
        else:
            if best_action[1] == 'melee':
                _, _, target_y, target_x, monster = best_action
                if self.wield_best_melee_weapon():
                    return wait_counter
                with self.env.debug_tiles([[self.blstats.y, self.blstats.x],
                                           [target_y, target_x]], color=(255, 0, 255), is_path=True):
                    try:
                        self.fight(target_y, target_x)
                    finally:
                        self._track_hunted_corpse(monster, target_x, target_y)
                    wait_counter = 0
                    return wait_counter
            elif best_action[1] == 'ranged':
                _, _, target_y, target_x, monster = best_action
                launcher, ammo = self.inventory.get_best_ranged_set()
                assert ammo is not None
                if launcher is not None and not launcher.equipped:
                    if self.inventory.wield(launcher):
                        return wait_counter
                with self.env.debug_tiles([[target_y, target_x]], (0, 0, 255, 255), mode='frame'):
                    dir = self.calc_direction(self.blstats.y, self.blstats.x, target_y, target_x,
                                              allow_nonunit_distance=True)
                    try:
                        assert self.fire(ammo, dir)
                    finally:
                        self._track_hunted_corpse(monster, target_x, target_y)
                    return wait_counter
            elif best_action[1] == 'zap':
                _, _, dy, dx, wand, targeted_monsters = best_action
                dir = self.calc_direction(self.blstats.y, self.blstats.x, self.blstats.y + dy, self.blstats.x + dx,
                                          allow_nonunit_distance=True)

                with self.env.debug_tiles([[my, mx] for my, mx, _ in targeted_monsters],
                                          (255, 0, 255, 255), mode='frame'):
                    self.zap(wand, dir)
                    for my, mx, monster in targeted_monsters:
                        self._track_hunted_corpse(monster, mx, my)

            elif best_action[1] == 'pickup':
                _, _, items_to_pickup = best_action
                self.inventory.pickup(items_to_pickup)
            else:
                raise NotImplementedError()
        return wait_counter

    def _track_hunted_corpse(self, monster, target_x, target_y):
        _, _, _, _, mon_glyph = monster
        if nh.glyph_is_body(self.glyphs[target_y, target_x]) \
                and self.glyphs[target_y, target_x] - nh.GLYPH_BODY_OFF == nh.glyph_to_mon(mon_glyph):
            self.current_level().corpse_age[target_y, target_x] = self.blstats.time

    @utils.debug_log('engulfed_fight')
    @Strategy.wrap
    def engulfed_fight(self):
        if not utils.any_in(self.glyphs, G.SWALLOW):
            yield False
        yield True
        while True:
            mask = utils.isin(self.glyphs, G.SWALLOW)
            if not mask.any():
                break
            assert self.fight(*list(zip(*mask.nonzero()))[0])

    @utils.debug_log('fight1')
    @Strategy.wrap
    def fight1(self):
        yielded = False
        while 1:
            monsters = self.get_visible_monsters()

            # get only monsters with path to them
            monsters = [m for m in monsters if m[0] != -1]

            if not monsters or all(dis > 7 for dis, *_ in monsters):
                if not yielded:
                    yield False
                return

            if not yielded:
                yielded = True
                yield True
                self.character.parse_enhance_view()

            assert len(monsters) > 0
            dis, y, x, _, mon_glyph = monsters[0]

            def is_monster_next_to_me():
                monsters = self.get_visible_monsters()
                if not monsters:
                    return False
                for _, y, x, _, _ in monsters:
                    if utils.adjacent((y, x), (self.blstats.y, self.blstats.x)):
                        return True
                return False

            with self.context_preempt([
                is_monster_next_to_me,
            ]) as outcome:
                if outcome() is None:
                    if self.ranged_stance1():
                        continue

            keep_distance = self.should_keep_distance(monsters)
            if self.keep_distance(monsters, keep_distance):
                continue
            # else:
            #     if self.emergency_strategy().run(return_condition=True):
            #         continue

            # TODO: why is this possible
            if self.bfs()[y, x] == -1:
                continue

            if abs(self.blstats.y - y) > 1 or abs(self.blstats.x - x) > 1:
                throwable = [i for i in self.inventory.items if i.is_thrown_projectile() and not i.equipped]
                # TODO: don't shoot pet !
                # TODO: limited range
                if throwable and (self.blstats.y == y or self.blstats.x == x or abs(self.blstats.y - y) == abs(
                        self.blstats.x - x)):
                    dir = self.calc_direction(self.blstats.y, self.blstats.x, y, x, allow_nonunit_distance=True)
                    self.fire(throwable[0], dir)
                    continue

                self.go_to(y, x, stop_one_before=True, max_steps=1,
                           debug_tiles_args=dict(color=(255, 0, 0), is_path=True))
                continue

            if self.wield_best_melee_weapon():
                continue
            try:
                self.fight(y, x)
            finally:  # TODO: what if panic?
                if nh.glyph_is_body(self.glyphs[y, x]) \
                        and self.glyphs[y, x] - nh.GLYPH_BODY_OFF == nh.glyph_to_mon(mon_glyph):
                    self.current_level().corpse_age[y, x] = self.blstats.time

    @utils.debug_log('eat1')
    @Strategy.wrap
    def eat1(self):
        level = self.current_level()

        if self.character.race == Character.ORC:
            flags1 = MON.M1_ACID
        else:
            flags1 = MON.M1_ACID | MON.M1_POIS
        flags2 = MON.M2_WERE

        editable_bodies = [b for b in G.BODIES if MON.permonst(b).mflags1 & flags1 == 0 and \
                                                  MON.permonst(b).mflags2 & flags2 == 0 and \
                                                  ord(MON.permonst(b).mlet) != MON.S_COCKATRICE]
        mask = utils.isin(self.glyphs, editable_bodies) & \
               ((self.blstats.time - level.corpse_age <= 50) |
                utils.isin(self.glyphs, [MON.body_from_name('lizard'), MON.body_from_name('lichen')]))
        mask |= utils.isin(self.glyphs, G.FOOD_OBJECTS)
        mask &= ~level.shop
        if not mask.any():
            yield False

        mask &= (self.bfs() != -1)
        if not mask.any():
            yield False

        yield True

        dis = self.bfs()
        mask &= dis == dis[mask].min()

        closests_y, closests_x = mask.nonzero()
        target_y, target_x = closests_y[0], closests_x[0]

        self.go_to(target_y, target_x, debug_tiles_args=dict(color=(255, 255, 0), is_path=True))
        if not self.current_level().shop[self.blstats.y, self.blstats.x]:
            self.eat()  # TODO: what

    @utils.debug_log('emergency_strategy')
    @Strategy.wrap
    def emergency_strategy(self):
        if (
                self.is_safe_to_pray() and
                (self.blstats.hitpoints < 1 / (5 if self.blstats.experience_level < 6 else 6)
                 * self.blstats.max_hitpoints or self.blstats.hitpoints < 6
                 or self.blstats.hunger_state >= Hunger.FAINTING)
        ):
            yield True
            self.pray()
            return

        items = [item for item in flatten_items(self.inventory.items) if item.is_unambiguous() and
                item.category == nh.POTION_CLASS and item.object.name in ['healing', 'extra healing', 'full healing']]
        if (
                (self.blstats.hitpoints < 1 / 3 * self.blstats.max_hitpoints
                 or self.blstats.hitpoints < 8 or self.blstats.hunger_state >= Hunger.FAINTING) and
                items
        ):
            yield True
            self.inventory.quaff(items[0])
            return

        # if self.inventory.engraving_below_me.lower() != 'elbereth' and self.can_engrave() and \
        #         (self.blstats.hitpoints < 1 / 5 * self.blstats.max_hitpoints or self.blstats.hitpoints < 5):
        #     yield True
        #     self.engrave('Elbereth')
        #     for _ in range(8):
        #         if self.inventory.engraving_below_me.lower() != 'elbereth':
        #             break
        #         self.direction('.')
        #     return

        yield False

    @utils.debug_log('eat_from_inventory')
    @Strategy.wrap
    def eat_from_inventory(self):
        if self.blstats.hunger_state < Hunger.HUNGRY:
            yield False
        for item in flatten_items(self.inventory.items):
            if item.category == nh.FOOD_CLASS and \
                    item.objs[0].name != 'sprig of wolfsbane' and \
                    (not item.is_corpse() or
                     item.monster_id in [MON.from_name(n) - nh.GLYPH_MON_OFF for n in ['lizard', 'lichen']]):
                yield True
                self.inventory.eat(item)
                return
        yield False

    @utils.debug_log('cure_disease')
    @Strategy.wrap
    def cure_disease(self):
        if self.character.is_lycanthrope:
            # spring of wolfbane
            for item in flatten_items(self.inventory.items):
                if item.objs[0].name == 'sprig of wolfsbane':
                    yield True
                    self.inventory.eat(item)
                    return

            # holy water
            for item in flatten_items(self.inventory.items):
                if item.objs[0].name == 'water' and item.status == Item.BLESSED:
                    yield True
                    self.inventory.quaff(item)
                    return

            # pray
            if self.is_safe_to_pray():
                yield True
                self.pray()
                return

        yield False

    ####### MAIN

    def handle_exception(self, exc):
        if isinstance(exc, (KeyboardInterrupt, AgentFinished)):
            raise exc
        if isinstance(exc, BaseException):
            if not isinstance(exc, AgentPanic) and not self.panic_on_errors:
                raise exc
            self.all_panics.append(exc)
            if self.verbose:
                print(f'PANIC!!!! : {exc}')

    def main(self):
        try:
            init_finished = False
            try:
                with self.atom_operation():
                    self.step(A.Command.ESC)
                    self.step(A.Command.ESC)

                    self.current_level().stair_destination[self.blstats.y, self.blstats.x] = \
                        ((Level.PLANE, 1), (None, None))  # TODO: check level num
                    self.character.parse()
                    self.character.parse_enhance_view()
                    self.step(A.Command.AUTOPICKUP)
                    if 'Autopickup: ON' in self.message:
                        self.step(A.Command.AUTOPICKUP)
                    init_finished = True
            except BaseException as e:
                self.handle_exception(e)

            assert init_finished

            last_step = self.step_count
            inactivity_counter = 0
            while 1:
                inactivity_counter += 1
                if self.step_count != last_step:
                    inactivity_counter = 0
                assert inactivity_counter < 5, ('cyclic panic', sorted({p.args[0] for p in self.all_panics[-5:]}))

                try:
                    self.step(A.Command.ESC)
                    self.step(A.Command.ESC)
                    self.on_panic()

                    last_step = self.step_count

                    self.global_logic.global_strategy().run()
                    assert 0
                except BaseException as e:
                    self.handle_exception(e)
        except AgentFinished:
            pass
