import contextlib
import re
from collections import namedtuple
from functools import partial

import nle.nethack as nh
import numpy as np
from nle.nethack import actions as A

import objects
import utils
from character import Character
from exceptions import AgentPanic, AgentFinished, AgentChangeStrategy
from glyph import SS, MON, C
from item import Item, Inventory
from strategy import Strategy

BLStats = namedtuple('BLStats',
                     'x y strength_percentage strength dexterity constitution intelligence wisdom charisma score hitpoints max_hitpoints depth gold energy max_energy armor_class monster_level experience_level experience_points time hunger_state carrying_capacity dungeon_number level_number')


class G:  # Glyphs
    FLOOR: ['.'] = {SS.S_room, SS.S_ndoor, SS.S_darkroom}
    STONE: [' '] = {SS.S_stone}
    WALL: ['|', '-'] = {SS.S_vwall, SS.S_hwall, SS.S_tlcorn, SS.S_trcorn, SS.S_blcorn, SS.S_brcorn,
                        SS.S_crwall, SS.S_tuwall, SS.S_tdwall, SS.S_tlwall, SS.S_trwall}
    CORRIDOR: ['#'] = {SS.S_corr}
    STAIR_UP: ['<'] = {SS.S_upstair}
    STAIR_DOWN: ['>'] = {SS.S_dnstair}

    DOOR_CLOSED: ['+'] = {SS.S_vcdoor, SS.S_hcdoor}
    DOOR_OPENED: ['-', '|'] = {SS.S_vodoor, SS.S_hodoor}
    DOORS = set.union(DOOR_CLOSED, DOOR_OPENED)

    MONS = set(MON.ALL_MONS)
    PETS = set(MON.ALL_PETS)

    PEACEFUL_MONS = {i + nh.GLYPH_MON_OFF for i in range(nh.NUMMONS) if nh.permonst(i).mflags2 & MON.M2_PEACEFUL}

    BODIES = {nh.GLYPH_BODY_OFF + i for i in range(nh.NUMMONS)}
    OBJECTS = {nh.GLYPH_OBJ_OFF + i for i in range(nh.NUM_OBJECTS) if ord(nh.objclass(i).oc_class) != nh.ROCK_CLASS}
    BOULDER = {nh.GLYPH_OBJ_OFF + i for i in range(nh.NUM_OBJECTS) if ord(nh.objclass(i).oc_class) == nh.ROCK_CLASS}

    NORMAL_OBJECTS = {i for i in range(nh.MAX_GLYPH) if nh.glyph_is_normal_object(i)}
    FOOD_OBJECTS = {i for i in NORMAL_OBJECTS if ord(nh.objclass(nh.glyph_to_obj(i)).oc_class) == nh.FOOD_CLASS}

    DICT = {k: v for k, v in locals().items() if not k.startswith('_')}

    @classmethod
    def assert_map(cls, glyphs, chars):
        for glyph, char in zip(glyphs.reshape(-1), chars.reshape(-1)):
            char = bytes([char]).decode()
            for k, v in cls.__annotations__.items():
                assert glyph not in cls.DICT[k] or char in v, f'{k} {v} {glyph} {char}'


G.INV_DICT = {i: [k for k, v in G.DICT.items() if i in v]
              for i in set.union(*map(set, G.DICT.values()))}


class Hunger:
    SATIATED = 0
    NOT_HUNGRY = 1
    HUNGRY = 2
    WEAK = 3
    FAINTING = 4


class Level:
    def __init__(self):
        self.walkable = np.zeros((C.SIZE_Y, C.SIZE_X), bool)
        self.seen = np.zeros((C.SIZE_Y, C.SIZE_X), bool)
        self.objects = np.zeros((C.SIZE_Y, C.SIZE_X), np.int16)
        self.objects[:] = -1
        self.search_count = np.zeros((C.SIZE_Y, C.SIZE_X), np.int32)
        self.corpse_age = np.zeros((C.SIZE_Y, C.SIZE_X), np.int32) - 10000
        self.shop = np.zeros((C.SIZE_Y, C.SIZE_X), bool)
        self.checked_item_pile = np.zeros((C.SIZE_Y, C.SIZE_X), bool)


class Agent:
    def __init__(self, env, seed=0, verbose=False):
        self.env = env
        self.verbose = verbose
        self.rng = np.random.RandomState(seed)
        self.all_panics = []

        self.on_update = []
        self.levels = {}
        self.score = 0
        self.step_count = 0
        self.message = ''
        self.popup = []

        self.inventory = Inventory(self)
        self.character = Character(self)

        self.last_bfs_dis = None
        self.last_bfs_step = None
        self.last_prayer_turn = None
        self._previous_glyphs = None

        self._no_step_calls = False

        self.turns_in_atom_operation = None

        self._is_reading_message_or_popup = False

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
    def atom_operation(self):
        if self.turns_in_atom_operation is not None:
            # already in an atom operation
            yield
            return

        self.turns_in_atom_operation = 0
        try:
            yield
        finally:
            self.turns_in_atom_operation = None

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
            self.on_update = list(filter(lambda f: f not in id2fun.values(), self.on_update))

        # check if less nested ChangeStategy is present
        self.call_update_functions()

    def preempt(self, strategies, default):
        id2fun = {}
        for strategy in strategies:
            def f(iden, strategy=strategy):
                it = strategy.strategy()
                if next(it):
                    raise AgentChangeStrategy(iden, it)

            fun = partial(f, id(f))
            assert id(f) not in id2fun
            id2fun[id(f)] = fun

        inactivity_counter = 0
        last_step = 0

        call_update = True

        while 1:
            assert inactivity_counter < 100
            if last_step != self.step_count:
                last_step = self.step_count
                inactivity_counter = 0
            else:
                inactivity_counter += 1

            iterator = None
            try:
                with self.add_on_update(list(id2fun.values())):
                    if call_update:
                        call_update = False
                        self.call_update_functions()

                    if isinstance(default, Strategy):
                        val = default.run()
                    else:
                        val = default()
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

        return val

    ######## UPDATE FUNCTIONS

    def on_panic(self):
        self.inventory.on_panic()

    def get_message_and_popup(self, obs):
        """ Uses MORE action to get full popup and/or message.
        """

        def find_marker(lines):
            """ Return (line, column) of markers:
            --More-- | (end) | (X of N)
            """
            regex = r"(--More--|\(end\)|\(\d+ of \d+\))"
            if len(re.findall(regex, ' '.join(lines))) > 1:
                raise ValueError('Too many markers')

            result, marker_type = None, None
            for i, line in enumerate(lines):
                res = re.findall(regex, line)
                if res:
                    assert len(res) == 1
                    j = line.find(res[0])
                    result, marker_type = (i, j), res[0]
                    break
            return result, marker_type

        message = bytes(obs['message']).decode().replace('\0', ' ').replace('\n', '').strip()
        if message.endswith('--More--'):
            # FIXME: It seems like in this case the environment doesn't expect additional input,
            #        but I'm not 100% sure, so it's too risky to change it, because it could stall everything.
            #        With the current implementation, in the worst case, we'll get "Unknown command ' '".
            message = message[:-len('--More--')]

        # assert '\n' not in message and '\r' not in message
        if self._is_reading_message_or_popup:
            message_preffix = self.message + (' ' if self.message else '')
            popup = self.popup
        else:
            message_preffix = ''
            popup = []

        lines = [bytes(line).decode().replace('\0', ' ').replace('\n', '') for line in obs['tty_chars']]
        marker_pos, marker_type = find_marker(lines)

        if marker_pos is None:
            self._is_reading_message_or_popup = False
            return message_preffix + message, popup, True
        self._is_reading_message_or_popup = True

        pref = ''
        message_lines_count = 0
        if message:
            for i, line in enumerate(lines[:marker_pos[0] + 1]):
                if i == marker_pos[0]:
                    line = line[:marker_pos[1]]
                message_lines_count += 1
                pref += line.strip()

                # I'm not sure when the new line character in broken messages should be a space and when be ignored.
                # '#' character occasionally occurs at the beginning of the broken line and isn't in the message.
                if pref.replace(' ', '').replace('#', '') == message.replace(' ', '').replace('#', ''):
                    break
            else:
                if marker_pos[0] == 0:
                    elems1 = [s for s in message.split() if s]
                    elems2 = [s for s in pref.split() if s]
                    assert len(elems1) < len(elems2) and elems2[-len(elems1):] == elems1, (elems1, elems2)
                    return message_preffix + pref, popup, False
                if self.env.visualizer is not None:
                    self.env.visualizer.frame_skipping = 1
                    self.env.render()
                raise ValueError(f"Message:\n{repr(message)}\ndoesn't match the screen:\n{repr(pref)}")

        # cut out popup
        for l in lines[message_lines_count:marker_pos[0]] + [lines[marker_pos[0]][:marker_pos[1]]]:
            l = l[marker_pos[1]:].strip()
            if l:
                popup.append(l)

        return message_preffix + message, popup, False

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

        if done:
            raise AgentFinished()

        self.update(observation, additional_action_iterator)

    def update(self, observation, additional_action_iterator=None):
        self.message, self.popup, done = self.get_message_and_popup(observation)

        self.message = self.message.strip()
        self.popup = [p.strip() for p in self.popup]

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
        if not done:
            self.step(A.TextCharacters.SPACE)
            return

        if observation['misc'][1]:  # entering text
            self.step(A.Command.ESC)
            return

        if b'[yn]' in bytes(observation['tty_chars'].reshape(-1)):
            self.type_text('y')
            return

        should_update = True

        if self.turns_in_atom_operation is not None:
            should_update = False
            if any([(self.last_observation[key] != observation[key]).any()
                    for key in ['glyphs', 'blstats', 'inv_strs', 'inv_letters', 'inv_oclasses', 'inv_glyphs']]):
                self.turns_in_atom_operation += 1
            # assert self.turns_in_atom_operation in [0, 1]

        self.last_observation = observation

        self.blstats = BLStats(*self.last_observation['blstats'])
        self.glyphs = self.last_observation['glyphs']

        if should_update:
            self.update_state()

    def update_state(self):
        self.inventory.update()
        self.update_level()
        self.call_update_functions()

    def call_update_functions(self):
        with self.disallow_step_calling():
            for func in self.on_update:
                func()

    def update_level(self):
        level = self.current_level()

        if '(for sale,' in self.message:
            level.shop[self.blstats.y, self.blstats.x] = 1

        if self._previous_glyphs is None or (self._previous_glyphs != self.last_observation['glyphs']).any():
            self._previous_glyphs = self.last_observation['glyphs']

            # TODO: all statues
            mask = utils.isin(self.glyphs, G.FLOOR, G.CORRIDOR, G.STAIR_UP, G.STAIR_DOWN, G.DOOR_OPENED)
            level.walkable[mask] = True
            level.seen[mask] = True
            level.objects[mask] = self.glyphs[mask]

            mask = utils.isin(self.glyphs, G.WALL, G.DOOR_CLOSED)
            level.seen[mask] = True
            level.objects[mask] = self.glyphs[mask]

            mask = utils.isin(self.glyphs, G.MONS, G.PETS, G.BODIES, G.OBJECTS)
            level.seen[mask] = True
            level.walkable[mask] = True

        for y, x in self.neighbors(self.blstats.y, self.blstats.x, shuffle=False):
            if self.glyphs[y, x] in G.STONE:
                level.seen[y, x] = True
                level.objects[y, x] = self.glyphs[y, x]

    ######## TRIVIAL HELPERS

    def current_level(self):
        key = (self.blstats.dungeon_number, self.blstats.level_number)
        if key not in self.levels:
            self.levels[key] = Level()
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

    def wield_best_weapon(self):
        # TODO: move to inventory
        item = self.inventory.get_best_weapon()
        if item is None:
            return False
        if not item.equipped:
            self.inventory.wield(item)
            return True
        return False

    def type_text(self, text):
        with self.atom_operation():
            for char in text:
                self.step(char)

    def eat(self):  # TODO: eat what
        with self.atom_operation():
            self.step(A.Command.EAT)
            if "You don't have anything to eat." in self.message:
                return False
            self.type_text('y')
            if "You don't have that object." in self.message:
                self.step(A.Command.ESC)
                return False
        return True

    def pray(self):
        self.step(A.Command.PRAY)
        return True

    def open_door(self, y, x=None):
        with self.panic_if_position_changes():
            assert self.glyphs[y, x] in G.DOOR_CLOSED
            self.direction(y, x)
            return self.glyphs[y, x] not in G.DOOR_CLOSED

    def fight(self, y, x=None):
        with self.panic_if_position_changes():
            assert self.glyphs[y, x] in G.MONS
            self.direction(y, x)
        return True

    def fire(self, item, direction):
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

    def search(self):
        with self.panic_if_position_changes():
            self.step(A.Command.SEARCH)
            self.current_level().search_count[self.blstats.y, self.blstats.x] += 1
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

        self.direction(dir)

        if self.blstats.y != expected_y or self.blstats.x != expected_x:
            raise AgentPanic(f'agent position do not match after "move": '
                             f'expected ({expected_y}, {expected_x}), got ({self.blstats.y}, {self.blstats.x})')

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

        walkable = level.walkable & ~utils.isin(self.glyphs, G.PEACEFUL_MONS, G.BOULDER)

        dis = utils.bfs(y, x,
                        walkable=walkable,
                        walkable_diagonally=walkable & ~utils.isin(level.objects, G.DOORS) & (level.objects != -1))

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

    def go_to(self, y, x, stop_one_before=False, max_steps=None, debug_tiles_args=None):
        assert not stop_one_before or (self.blstats.y != y or self.blstats.x != x)
        assert self.bfs()[y, x] != -1

        steps_taken = 0
        cont = True
        while cont:
            dis = self.bfs()
            if dis[y, x] == -1:
                raise AgentPanic('end point is no longer accessible')
            path = self.path(self.blstats.y, self.blstats.x, y, x)

            with self.env.debug_tiles(path, **debug_tiles_args) \
                    if debug_tiles_args is not None else contextlib.suppress():
                path = path[1:]
                if stop_one_before:
                    path = path[:-1]
                for y, x in path:
                    if self.glyphs[y, x] in G.PEACEFUL_MONS:
                        cont = True
                        break
                    if not self.current_level().walkable[y, x]:
                        cont = True
                        break
                    self.move(y, x)
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
            launchers = [i for i in self.inventory.items if i.is_launcher()]
            ammo_list = [i for i in self.inventory.items if i.is_fired_projectile()]

            valid_combinations = []
            for launcher in launchers:
                for ammo in ammo_list:
                    if ammo.is_fired_projectile(launcher):
                        valid_combinations.append((launcher, ammo))

            # TODO: select best combination
            if not valid_combinations:
                self.wield_best_weapon()
                return False

            # TODO: consider using monster information to select the best combination
            launcher, ammo = valid_combinations[0]
            if not launcher.equipped:
                self.inventory.wield(launcher)
                continue

            for _, y, x, _ in self.get_visible_monsters():
                if (self.blstats.y == y or self.blstats.x == x or abs(self.blstats.y - y) == abs(self.blstats.x - x)):
                    # TODO: don't shoot pet !
                    # TODO: limited range
                    with self.env.debug_tiles([[y, x]], (0, 0, 255, 100)):
                        dir = self.calc_direction(self.blstats.y, self.blstats.x, y, x, allow_nonunit_distance=True)
                        self.fire(ammo, dir)
                        break
            else:
                return False

    def get_visible_monsters(self):
        """ Returns list of tuples (distance, y, x, monster)
        """
        dis = self.bfs()
        mask = utils.isin(self.glyphs, G.MONS - G.PEACEFUL_MONS)
        mask[self.blstats.y, self.blstats.x] = 0
        mask &= dis != -1
        # mask &= dis == dis[mask].min()
        ret = []
        for y, x in zip(*mask.nonzero()):
            ret.append((dis[y][x], y, x, nh.glyph_to_mon(self.glyphs[y][x])))
        ret.sort()
        return ret

    @utils.debug_log('fight1')
    @Strategy.wrap
    def fight1(self):
        yielded = False
        while 1:
            monsters = self.get_visible_monsters()

            # get only monsters with path to them
            monsters = [m for m in monsters if m[0] != -1]

            if not monsters:
                if not yielded:
                    yield False
                return

            if not yielded:
                yielded = True
                yield True
                self.character.parse_enhance_view()

            assert len(monsters) > 0
            dis, y, x, mon = monsters[0]

            def is_monster_next_to_me():
                monsters = self.get_visible_monsters()
                if not monsters:
                    return False
                return True

            with self.context_preempt([
                is_monster_next_to_me,
            ]) as outcome:
                if outcome() is None:
                    if self.ranged_stance1():
                        continue

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

            if self.wield_best_weapon():
                continue
            try:
                self.fight(y, x)
            finally:  # TODO: what if panic?
                if nh.glyph_is_body(self.glyphs[y, x]) and self.glyphs[y, x] - nh.GLYPH_BODY_OFF == mon:
                    self.current_level().corpse_age[y, x] = self.blstats.time

    @utils.debug_log('eat1')
    @Strategy.wrap
    def eat1(self):
        level = self.current_level()

        mask = utils.isin(self.glyphs, G.BODIES) & (self.blstats.time - level.corpse_age <= 100)
        mask |= utils.isin(self.glyphs, G.FOOD_OBJECTS)
        mask &= ~level.shop
        if not mask.any():
            yield False

        mask &= (self.bfs() != -1)
        if not mask.any():
            yield False

        yield True

        # TODO: use variables from the condition
        dis = self.bfs()
        closest = None

        level = self.current_level()
        # TODO: iter by distance
        for y in range(C.SIZE_Y):
            for x in range(C.SIZE_X):
                if dis[y, x] != -1 and (closest is None or dis[y, x] < dis[closest]) and not level.shop[y, x]:
                    if self.glyphs[y, x] in G.BODIES and self.blstats.time - level.corpse_age[y, x] <= 100:
                        closest = (y, x)
                    if nh.glyph_is_normal_object(self.glyphs[y, x]):
                        obj = nh.objclass(nh.glyph_to_obj(self.glyphs[y, x]))
                        if ord(obj.oc_class) == nh.FOOD_CLASS:
                            closest = (y, x)

        assert closest is not None
        # if closest is None:
        #    return False

        target_y, target_x = closest
        path = self.path(self.blstats.y, self.blstats.x, target_y, target_x)

        self.go_to(target_y, target_x, debug_tiles_args=dict(color=(255, 255, 0), is_path=True))
        if not self.current_level().shop[self.blstats.y, self.blstats.x]:
            self.eat()  # TODO: what

    @utils.debug_log('explore1')
    def explore1(self, search_prio_limit=0):
        # TODO: refactor entire function

        def open_neighbor_doors():
            for py, px in self.neighbors(self.blstats.y, self.blstats.x, diagonal=False):
                if self.glyphs[py, px] in G.DOOR_CLOSED:
                    with self.panic_if_position_changes():
                        if not self.open_door(py, px):
                            if not 'locked' in self.message:
                                for _ in range(6):
                                    if self.open_door(py, px):
                                        break
                                else:
                                    while self.glyphs[py, px] in G.DOOR_CLOSED:
                                        self.kick(py, px)
                            else:
                                while self.glyphs[py, px] in G.DOOR_CLOSED:
                                    self.kick(py, px)
                    break

        def to_visit_func():
            level = self.current_level()
            to_visit = np.zeros((C.SIZE_Y, C.SIZE_X), dtype=bool)
            dis = self.bfs()
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy != 0 or dx != 0:
                        to_visit |= utils.translate(~level.seen & utils.isin(self.glyphs, G.STONE), dy, dx)
                        if dx == 0 or dy == 0:
                            to_visit |= utils.translate(utils.isin(self.glyphs, G.DOOR_CLOSED), dy, dx)
            return to_visit

        def to_search_func(prio_limit=0, return_prio=False):
            level = self.current_level()
            dis = self.bfs()

            prio = np.zeros((C.SIZE_Y, C.SIZE_X), np.float32)
            prio[:] = -1
            prio -= level.search_count ** 2 * 2
            is_on_corridor = utils.isin(level.objects, G.CORRIDOR)
            is_on_door = utils.isin(level.objects, G.DOORS)

            stones = np.zeros((C.SIZE_Y, C.SIZE_X), np.int32)
            walls = np.zeros((C.SIZE_Y, C.SIZE_X), np.int32)

            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy != 0 or dx != 0:
                        stones += utils.isin(utils.translate(level.objects, dy, dx), G.STONE)
                        walls += utils.isin(utils.translate(level.objects, dy, dx), G.WALL)

            prio += (is_on_door & (stones > 3)) * 250
            prio += (np.stack([utils.translate(level.walkable, y, x).astype(np.int32)
                               for y, x in [(1, 0), (-1, 0), (0, 1), (0, -1)]]).sum(0) <= 1) * 250
            prio[(stones == 0) & (walls == 0)] = -np.inf

            prio[~level.walkable | (dis == -1)] = -np.inf

            if return_prio:
                return prio
            return prio >= prio_limit

        @Strategy.wrap
        def open_visit_search(search_prio_limit):
            yielded = False
            while 1:
                for py, px in self.neighbors(self.blstats.y, self.blstats.x, diagonal=False, shuffle=False):
                    if self.glyphs[py, px] in G.DOOR_CLOSED:
                        if not yielded:
                            yielded = True
                            yield True
                        open_neighbor_doors()
                        break

                to_visit = to_visit_func()
                to_search = to_search_func(search_prio_limit if search_prio_limit is not None else 0)

                # consider exploring tile only when there is a path to it
                dis = self.bfs()
                to_explore = (to_visit | to_search) & (dis != -1)

                dynamic_search_fallback = False
                if not to_explore.any():
                    dynamic_search_fallback = True
                else:
                    # find all closest to_explore tiles
                    nonzero_y, nonzero_x = ((dis == dis[to_explore].min()) & to_explore).nonzero()
                    if len(nonzero_y) == 0:
                        dynamic_search_fallback = True

                if dynamic_search_fallback:
                    if search_prio_limit is not None and search_prio_limit >= 0:
                        if not yielded:
                            yield False
                        return

                    search_prio = to_search_func(return_prio=True)
                    if search_prio_limit is not None:
                        search_prio[search_prio < search_prio_limit] = -np.inf
                        search_prio[search_prio < search_prio_limit] = -np.inf
                        search_prio -= dis * np.isfinite(search_prio) * 100
                    else:
                        search_prio -= dis * 4

                    to_search = np.isfinite(search_prio)
                    to_explore = (to_visit | to_search) & (dis != -1)
                    if not to_explore.any():
                        if not yielded:
                            yield False
                        return
                    nonzero_y, nonzero_x = ((search_prio == search_prio[to_explore].max()) & to_explore).nonzero()

                if not yielded:
                    yielded = True
                    yield True

                # select random closest to_explore tile
                i = self.rng.randint(len(nonzero_y))
                target_y, target_x = nonzero_y[i], nonzero_x[i]

                with self.env.debug_tiles(to_explore, color=(0, 0, 255, 64)):
                    self.go_to(target_y, target_x, debug_tiles_args=dict(
                        color=(255 * bool(to_visit[target_y, target_x]),
                               255, 255 * bool(to_search[target_y, target_x])),
                        is_path=True))
                    if to_search[target_y, target_x] and not to_visit[target_y, target_x]:
                        self.search()

            assert search_prio_limit is not None

        return open_visit_search(search_prio_limit).preempt(self, [
            self.inventory.gather_items(),
        ])

    @utils.debug_log('move_down')
    @Strategy.wrap
    def move_down(self):
        level = self.current_level()

        mask = utils.isin(level.objects, G.STAIR_DOWN)
        if not mask.any():
            yield False

        mask &= (self.bfs() != -1)
        if not mask.any():
            yield False
        yield True

        nonzero_y, nonzero_x = mask.nonzero()
        target_y, target_x = nonzero_y[0], nonzero_x[0]

        self.go_to(target_y, target_x, debug_tiles_args=dict(color=(0, 0, 255), is_path=True))
        with self.env.debug_log('waiting for a pet'):
            for _ in range(8):
                for y, x in self.neighbors(self.blstats.y, self.blstats.x):
                    if self.glyphs[y, x] in G.PETS:
                        break
                else:
                    self.direction('.')
                    continue
                break
            self.direction('>')

    @utils.debug_log('emergency_strategy')
    @Strategy.wrap
    def emergency_strategy(self):
        # TODO: to refactor
        if (
                ((self.last_prayer_turn is None and self.blstats.time > 300) or
                 (self.last_prayer_turn is not None and self.blstats.time - self.last_prayer_turn > 900)) and
                (self.blstats.hitpoints < 1/(5 if self.blstats.experience_level < 6 else 6) * self.blstats.max_hitpoints or
                 self.blstats.hitpoints < 6 or self.blstats.hunger_state >= Hunger.FAINTING)
                ):
            yield True
            self.last_prayer_turn = self.blstats.time
            self.pray()
            return

        if (
                (self.blstats.hitpoints < 1/3 * self.blstats.max_hitpoints or self.blstats.hitpoints < 8 or self.blstats.hunger_state >= Hunger.FAINTING) and
                len([s for s in map(lambda x: bytes(x).decode(), self.last_observation['inv_strs'])
                     if 'potion of healing' in s or 'potion of extra healing' in s or 'potion of full healing' in s]) > 0
                ):
            yield True
            with self.atom_operation():
                self.type_text('q')
                for letter, s in zip(self.last_observation['inv_letters'], map(lambda x: bytes(x).decode(), self.last_observation['inv_strs'])):
                    if 'potion of healing' in s or 'potion of extra healing' in s or 'potion of full healing' in s:
                        self.type_text(chr(letter))
                        break
                else:
                    assert 0
            return

        yield False

    ######## HIGH-LEVEL STRATEGIES

    @utils.debug_log('main_strategy')
    def main_strategy(self):
        @Strategy.wrap
        def eat_from_inventory():
            if not (self.blstats.hunger_state >= Hunger.WEAK and any(
                    map(lambda item: item.category == nh.FOOD_CLASS,
                        self.inventory.items))):
                yield False
            yield True
            with self.atom_operation():
                self.step(A.Command.EAT)
                for item in self.inventory.items:
                    if item.category == nh.FOOD_CLASS:
                        self.type_text(self.inventory.items.get_letter(item))
                        break
                else:
                    assert 0

        return \
            (self.explore1(0).before(self.explore1(None))).preempt(self, [
                self.move_down().condition(lambda: self.blstats.score > 550 and self.blstats.hitpoints >= 0.8 * self.blstats.max_hitpoints)
            ]).preempt(self, [
                self.eat1().condition(lambda: self.blstats.time % 3 == 0 and self.blstats.hunger_state >= Hunger.NOT_HUNGRY),
                eat_from_inventory(),
            ]).preempt(self, [
                self.fight1(),
            ]).preempt(self, [
                self.emergency_strategy(),
            ])

    ####### MAIN

    def main(self):
        self.update({k: v.copy() for k, v in self.env.reset().items()})
        self.character.parse()
        self.character.parse_enhance_view()

        try:
            self.step(A.Command.AUTOPICKUP)

            while 1:
                try:
                    self.step(A.Command.ESC)
                    self.step(A.Command.ESC)
                    self.main_strategy().run()
                    assert 0
                except AgentPanic as e:
                    self.all_panics.append(e)
                    self.on_panic()
                    if self.verbose:
                        print(f'PANIC!!!! : {e}')
        except AgentFinished:
            pass
