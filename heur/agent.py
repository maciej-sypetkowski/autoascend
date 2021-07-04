import contextlib
import re
from collections import namedtuple
from functools import partial
from itertools import chain

import nle.nethack as nh
import numpy as np
from nle.nethack import actions as A

import utils
from glyph import SS, MON, C

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

    SHOPKEEPER = {MON.fn('shopkeeper')}

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


class AgentFinished(Exception):
    pass


class AgentPanic(Exception):
    pass


class AgentChangeStrategy(Exception):
    pass


class CH:
    ARCHEOLOGIST = 0
    BARBARIAN = 1
    CAVEMAN = 2
    HEALER = 3
    KNIGHT = 4
    MONK = 5
    PRIEST = 6
    RANGER = 7
    ROGUE = 8
    SAMURAI = 9
    TOURIST = 10
    VALKYRIE = 11
    WIZARD = 12

    name_to_role = {
        'Archeologist': ARCHEOLOGIST,
        'Barbarian': BARBARIAN,
        'Caveman': CAVEMAN,
        'Cavewoman': CAVEMAN,
        'Healer': HEALER,
        'Knight': KNIGHT,
        'Monk': MONK,
        'Priest': PRIEST,
        'Priestess': PRIEST,
        'Ranger': RANGER,
        'Rogue': ROGUE,
        'Samurai': SAMURAI,
        'Tourist': TOURIST,
        'Valkyrie': VALKYRIE,
        'Wizard': WIZARD,
    }

    CHAOTIC = 0
    NEUTRAL = 1
    LAWFUL = 2

    name_to_alignment = {
        'chaotic': CHAOTIC,
        'neutral': NEUTRAL,
        'lawful': LAWFUL,
    }

    HUMAN = 0
    DWARF = 1
    ELF = 2
    GNOME = 3
    ORC = 4

    name_to_race = {
        'human': HUMAN,
        'dwarf': DWARF,
        'dwarven': DWARF,
        'elf': ELF,
        'elven': ELF,
        'gnome': GNOME,
        'gnomish': GNOME,
        'orc': ORC,
        'orcish': ORC,
    }

    MALE = 0
    FEMALE = 1

    name_to_gender = {
        'male': MALE,
        'female': FEMALE,
    }

    def __init__(self, role, alignment, race, gender):
        self.role = role
        self.alignment = alignment
        self.race = race
        self.gender = gender

    @classmethod
    def parse(cls, message):
        all = re.findall('You are a ([a-z]+) (([a-z]+) )?([a-z]+) ([A-Z][a-z]+).', message)
        if len(all) == 1:
            alignment, _, gender, race, role = all[0]
        else:
            all = re.findall(
                'You are an? ([a-zA-Z ]+), a level (\d+) (([a-z]+) )?([a-z]+) ([A-Z][a-z]+). *You are ([a-z]+)',
                message)
            assert len(all) == 1, repr(message)
            _, _, _, gender, race, role, alignment = all[0]

        if not gender:
            if role == 'Priestess':
                gender = 'female'
            elif role == 'Priest':
                gender = 'male'
            elif role == 'Caveman':
                gender = 'male'
            elif role == 'Cavewoman':
                gender = 'female'
            elif role == 'Valkyrie':
                gender = 'female'
            else:
                assert 0, repr(message)

        return cls(cls.name_to_role[role], cls.name_to_alignment[alignment],
                   cls.name_to_race[race], cls.name_to_gender[gender])

    def __str__(self):
        return '-'.join([f'{list(d.keys())[list(d.values()).index(v)][:3].lower()}'
                         for d, v in [(self.name_to_role, self.role),
                                      (self.name_to_race, self.race),
                                      (self.name_to_gender, self.gender),
                                      (self.name_to_alignment, self.alignment),
                                      ]])


class Item:
    UNKNOWN = 0
    CURSED = 1
    UNCURSED = 2
    BLESSED = 3

    def __init__(self, glyphs, count, status, modifier):
        # glyphs is a list of possible glyphs for this item
        assert isinstance(glyphs, list)
        assert all(map(nh.glyph_is_object, glyphs))
        assert len(glyphs)

        self.glyphs = glyphs
        self.count = count
        self.status = status
        self.modifier = modifier
        self.category = ord(nh.objclass(nh.glyph_to_obj(self.glyphs[0])).oc_class)

        assert all(map(lambda glyph: ord(nh.objclass(nh.glyph_to_obj(glyph)).oc_class) == self.category, self.glyphs))

    def __str__(self):
        return (f'{self.count}_'
                f'{self.status if self.status is not None else ""}_'
                f'{self.modifier if self.modifier is not None else ""}_'
                f'{",".join([nh.objdescr.from_idx(nh.glyph_to_obj(glyph)).oc_name for glyph in self.glyphs])}'
                )


class ItemManager:
    def __init__(self, agent):
        self.agent = agent

    def get_item_from_text(self, text, category):
        assert category not in [nh.BALL_CLASS, nh.ROCK_CLASS, nh.RANDOM_CLASS]

        matches = re.findall('^(a|an|\d+)( (cursed|uncursed|blessed))?( (very |thourogly )?(rustproof|poisoned|corroded|rusty|burnt))*( ([+-]\d+))? ([a-zA-z0-9- ]+)( \(([a-zA-Z0-9:; ]+)\))?$', text)
        assert len(matches) <= 1, text
        assert len(matches), text

        count, _, status, effects, _, _, _, modifier, name, _, info = matches[0]
        # TODO: effects

        count = int({'a': 1, 'an': 1}.get(count, count))
        status = {'': Item.UNKNOWN, 'cursed': Item.CURSED, 'uncursed': Item.UNCURSED, 'blessed': Item.BLESSED}[status]
        modifier = None if not modifier else {'+': 1, '-': -1}[modifier[0]] * int(modifier[1:])

        #if category not in [nh.WEAPON_CLASS, nh.ARMOR_CLASS]:
        #    return None # TODO

        if category == nh.WEAPON_CLASS:
            name_augmentation = lambda x: [x, f'{x}s']
        elif category == nh.ARMOR_CLASS:
            name_augmentation = lambda x: [x, f'pair of {x}']
        elif category in [nh.AMULET_CLASS, nh.FOOD_CLASS, nh.GEM_CLASS, nh.POTION_CLASS, nh.RING_CLASS,
                          nh.SCROLL_CLASS, nh.SPBOOK_CLASS, nh.TOOL_CLASS, nh.WAND_CLASS, nh.COIN_CLASS]:
            return Item([i + nh.GLYPH_OBJ_OFF for i in range(nh.NUM_OBJECTS) if ord(nh.objclass(i).oc_class) == category and
                                                                                nh.objdescr.from_idx(i).oc_name is not None],
                        count, status, modifier)
        else:
            assert 0, category

        ret = []

        if name == 'wakizashi':
            name = 'short sword'
        elif name == 'ninja-to':
            name = 'broadsword'

        for i in range(nh.NUM_OBJECTS):
            if ord(nh.objclass(i).oc_class) != category:
                continue

            obj_descr = nh.objdescr.from_idx(i).oc_descr
            obj_name = nh.objdescr.from_idx(i).oc_name
            if (obj_name and name in name_augmentation(obj_name)) or (obj_descr and name in name_augmentation(obj_descr)):
                ret.append(i)

        assert len(ret) == 1, (ret, name, text)
        return Item([r + nh.GLYPH_OBJ_OFF for r in ret], count, status, modifier)


class Agent:
    def __init__(self, env, seed=0, verbose=False):
        self.env = env
        self.verbose = verbose
        self.rng = np.random.RandomState(seed)
        self.all_panics = []

        self.on_update = []
        self.levels = {}
        self.last_observation = {k: v.copy() for k, v in env.reset().items()}
        self.score = 0
        self.step_count = 0

        self.last_bfs_dis = None
        self.last_bfs_step = None

        self.previous_inv_strs = None

        self.item_manager = ItemManager(self)

        self.update_map()

    def parse_character(self):
        with self.stop_updating():
            self.step(A.Command.ATTRIBUTES)
            text = self.last_observation['tty_chars']
            text = ' '.join([bytes(t).decode() for t in text])
            self.character = CH.parse(text)
            self.step(A.Command.ESC)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.step_count += 1

        self.last_observation = {k: v.copy() for k, v in obs.items()}
        self.score += reward
        if done:
            raise AgentFinished()

        self.update_map()

        return obs, reward, done, info

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
    def stop_updating(self, update_at_end=False):
        on_update = self.on_update
        self.on_update = []

        try:
            yield
        finally:
            assert self.on_update == []
            self.on_update = on_update

        if update_at_end:
            for f in self.on_update:
                f()

    @contextlib.contextmanager
    def preempt(self, conditions):
        funcs = []
        for cond in conditions:
            def f(self, f, cond):
                if cond():
                    raise AgentChangeStrategy(f)

            fun = partial(f, self, f, cond)
            funcs.append(fun)
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
            f = e.args[0]
            if f not in funcs:
                raise
            outcome = funcs.index(f)
        finally:
            self.on_update = list(filter(lambda f: f not in funcs, self.on_update))

    def get_message_and_popup(self, obs):
        """ Uses MORE action to get full popup and/or message.
        """
        # TODO: rethink recursion from calling step
        message = bytes(obs['message']).decode()
        popup = []
        is_extended_message = False
        while b'--More--' in bytes(obs['tty_chars'].reshape(-1)):
            assert not is_extended_message
            for i, line in enumerate(obs['tty_chars']):
                line = bytes(line)
                if b'--More--' not in line:
                    continue
                if i == 0:
                    message = line.decode()
                    is_extended_message = True
                else:
                    popup_start_column = line.find(b'--More--')
                    for line in obs['tty_chars'][:i]:
                        line = bytes(line)
                        # TODO: consider cutting line suffix (check if it is always of the same length)
                        popup.append(line[popup_start_column:].decode().strip())
                    break
            self.step(A.MiscAction.MORE)
            obs = self.last_observation

        if message == bytes([0] * 256).decode():
            message = ""

        if popup and message:
            raise ValueError(f'Either message ({str(message)}) or popup ({str(popup)}) should be empty')

        return message, popup

    def update_map(self):
        obs = self.last_observation

        self.blstats = BLStats(*obs['blstats'])
        self.glyphs = obs['glyphs']

        self.message, self.popup = self.get_message_and_popup(obs)

        if obs['misc'][1]: # entering text
            self.step(A.Command.ESC)
            return

        if b'[yn]' in bytes(obs['tty_chars'].reshape(-1)):
            self.enter_text('y')
            return

        self.update_level()
        self.update_inventory()

        for func in self.on_update:
            func()

    def update_inventory(self):
        if self.previous_inv_strs is not None and (self.last_observation['inv_strs'] == self.previous_inv_strs).all():
            return

        self.inventory = {}
        for item_name, category, letter in zip(
                self.last_observation['inv_strs'],
                self.last_observation['inv_oclasses'],
                self.last_observation['inv_letters']):
            item_name = bytes(item_name).decode().strip('\0')
            letter = chr(letter)
            if not item_name:
                continue
            item = self.item_manager.get_item_from_text(item_name, category)
            self.inventory[letter] = item

        self.previous_inv_strs = self.last_observation['inv_strs']

    def current_level(self):
        key = (self.blstats.dungeon_number, self.blstats.level_number)
        if key not in self.levels:
            self.levels[key] = Level()
        return self.levels[key]

    def glyphs_mask_in(self, *gset):
        gset = list(chain(*gset))
        return np.isin(self.glyphs, gset)

    def update_level(self):
        level = self.current_level()

        if '(for sale,' in self.message:
            level.shop[self.blstats.y, self.blstats.x] = 1

        mask = self.glyphs_mask_in(G.FLOOR, G.CORRIDOR, G.STAIR_UP, G.STAIR_DOWN, G.DOOR_OPENED)
        level.walkable[mask] = True
        level.seen[mask] = True
        level.objects[mask] = self.glyphs[mask]

        mask = self.glyphs_mask_in(G.WALL, G.DOOR_CLOSED)
        level.seen[mask] = True
        level.objects[mask] = self.glyphs[mask]

        mask = self.glyphs_mask_in(G.MONS, G.PETS, G.BODIES, G.OBJECTS)
        level.seen[mask] = True
        level.walkable[mask] = True

        for y, x in self.neighbors(self.blstats.y, self.blstats.x):
            if self.glyphs[y, x] in G.STONE:
                level.seen[y, x] = True
                level.objects[y, x] = self.glyphs[y, x]

    ######## TRIVIAL HELPERS

    @staticmethod
    def calc_direction(from_y, from_x, to_y, to_x):
        assert abs(from_y - to_y) <= 1 and abs(from_x - to_x) <= 1, ((from_y, from_x), (to_y, to_x))

        ret = ''
        if to_y == from_y + 1: ret += 's'
        if to_y == from_y - 1: ret += 'n'
        if to_x == from_x + 1: ret += 'e'
        if to_x == from_x - 1: ret += 'w'
        if ret == '': ret = '.'

        return ret

    ######## TRIVIAL ACTIONS

    def enter_text(self, text):
        with self.panic_if_position_changes():
            for char in text:
                char = ord(char)
                self.step(A.ACTIONS[A.ACTIONS.index(char)])

    def eat(self):  # TODO: eat what
        with self.panic_if_position_changes():  # TODO: anything, not only position
            self.step(A.Command.EAT)
            self.enter_text('y')
            self.step(A.Command.ESC)
            self.step(A.Command.ESC)
        return True  # TODO: return value

    def open_door(self, y, x=None):
        assert self.glyphs[y, x] in G.DOOR_CLOSED
        self.direction(y, x)
        return self.glyphs[y, x] not in G.DOOR_CLOSED

    def fight(self, y, x=None):
        assert self.glyphs[y, x] in G.MONS
        self.direction(y, x)
        return True

    def kick(self, y, x=None):
        with self.panic_if_position_changes():
            self.step(A.Command.KICK)
            self.direction(self.calc_direction(self.blstats.y, self.blstats.x, y, x))

    def search(self):
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
            '>': A.MiscDirection.DOWN, '<': A.MiscDirection.UP
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
            return self.last_bfs_dis

        level = self.current_level()

        walkable = level.walkable & ~self.glyphs_mask_in(G.SHOPKEEPER, G.BOULDER)

        dis = utils.bfs(y, x,
                        walkable=walkable,
                        walkable_diagonally=walkable & ~np.isin(level.objects, list(G.DOORS)) & (level.objects != -1))


        if y == self.blstats.y and x == self.blstats.x:
            self.last_bfs_dis = dis
            self.last_bfs_step = self.step_count

        return dis

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

    def is_any_mon_on_map(self):
        mask = self.glyphs_mask_in(G.MONS - G.SHOPKEEPER)
        mask[self.blstats.y, self.blstats.x] = 0
        if not mask.any():
            return False
        return (mask & (self.bfs() != -1)).any()

    def is_any_food_on_map(self):
        level = self.current_level()

        mask = self.glyphs_mask_in(G.BODIES) & (self.blstats.time - level.corpse_age <= 100)
        mask |= self.glyphs_mask_in(G.FOOD_OBJECTS)
        mask &= ~level.shop
        if not mask.any():
            return False
        return (mask & (self.bfs() != -1)).any()

    ######## NON-TRIVIAL ACTIONS

    def go_to(self, y, x, stop_one_before=False, max_steps=None, debug_path_args=None):
        assert not stop_one_before or (self.blstats.y != y or self.blstats.x != x)
        assert self.bfs()[y, x] != -1

        steps_taken = 0
        cont = True
        while cont:
            dis = self.bfs()
            if dis[y, x] == -1:
                raise AgentPanic('end point is no longer accessible')
            path = self.path(self.blstats.y, self.blstats.x, y, x)

            with self.env.debug_path(path, **debug_path_args) if debug_path_args is not None else contextlib.suppress():
                path = path[1:]
                if stop_one_before:
                    path = path[:-1]
                for y, x in path:
                    if self.glyphs[y, x] in G.SHOPKEEPER:
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

    def fight1(self):
        while 1:
            dis = self.bfs()
            closest = None

            mask = self.glyphs_mask_in(G.MONS - G.SHOPKEEPER)
            mask[self.blstats.y, self.blstats.x] = 0
            mask &= dis != -1

            if not mask.any():
                return False

            mask &= dis == dis[mask].min()
            closests_y, closests_x = mask.nonzero()

            assert len(closests_y) > 0

            y, x = closests_y[0], closests_x[0]

            # TODO: allow diagonal fight from doors

            if self.bfs()[y, x] != 1:
                self.go_to(y, x, stop_one_before=True, max_steps=1, debug_path_args=dict(color=(255, 0, 0)))
                continue

            mon = nh.glyph_to_mon(self.glyphs[y, x])

            try:
                self.fight(y, x)
            finally:  # TODO: what if panic?
                if nh.glyph_is_body(self.glyphs[y, x]) and self.glyphs[y, x] - nh.GLYPH_BODY_OFF == mon:
                    self.current_level().corpse_age[y, x] = self.blstats.time

    def eat1(self):
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

        self.go_to(target_y, target_x, debug_path_args=dict(color=(255, 255, 0)))
        if not self.current_level().shop[self.blstats.y, self.blstats.x]:
            self.eat()  # TODO: what

    def explore1(self):

        # doors
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

        level = self.current_level()
        to_explore = np.zeros((C.SIZE_Y, C.SIZE_X), dtype=bool)
        dis = self.bfs()
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy != 0 or dx != 0:
                    to_explore |= utils.translate(~level.seen & self.glyphs_mask_in(G.STONE), dy, dx)
                    if dx == 0 or dy == 0:
                        to_explore |= utils.translate(self.glyphs_mask_in(G.DOOR_CLOSED), dy, dx)

        # consider exploring tile only when there is a path to it
        to_explore &= dis != -1

        # find all closest to_explore tiles
        nonzero_y, nonzero_x = \
            (dis == (dis * (to_explore) - 1).astype(np.uint16).min() + 1).nonzero()
        nonzero = [(y, x) for y, x in zip(nonzero_y, nonzero_x) if to_explore[y, x]]
        if len(nonzero) == 0:
            return False

        # select closest to_explore tile
        nonzero_y, nonzero_x = zip(*nonzero)
        target_y, target_x = nonzero_y[0], nonzero_x[0]

        del level

        self.go_to(target_y, target_x, debug_path_args=dict(color=(255, 255, 0)))

    def search1(self):
        level = self.current_level()
        dis = self.bfs()

        prio = np.zeros((C.SIZE_Y, C.SIZE_X), np.float32)
        for y in range(C.SIZE_Y):
            for x in range(C.SIZE_X):
                if not level.walkable[y, x] or dis[y, x] == -1:
                    prio[y, x] = -np.inf
                else:
                    prio[y, x] = -20
                    prio[y, x] -= dis[y, x]
                    prio[y, x] -= level.search_count[y, x] ** 2 * 10
                    prio[y, x] += (level.objects[y, x] in G.CORRIDOR) * 15 + (level.objects[y, x] in G.DOORS) * 80
                    for py, px in self.neighbors(y, x, shuffle=False):
                        prio[y, x] += (level.objects[py, px] in G.STONE) * 40 + (level.objects[py, px] in G.WALL) * 20

        nonzero_y, nonzero_x = (prio == prio.max()).nonzero()
        assert len(nonzero_y) >= 0

        target_y, target_x = nonzero_y[0], nonzero_x[0]

        self.go_to(target_y, target_x, debug_path_args=dict(color=(0, 255, 255)))
        self.search()

    def move_down(self):
        level = self.current_level()

        pos = None
        for y in range(C.SIZE_Y):
            for x in range(C.SIZE_X):
                if level.objects[y, x] in G.STAIR_DOWN:
                    pos = (y, x)
                    break
            else:
                continue
            break

        if pos is None:
            return False

        dis = self.bfs()
        if dis[pos] == -1:
            return False

        target_y, target_x = pos

        self.go_to(target_y, target_x, debug_path_args=dict(color=(0, 0, 255)))
        self.direction('>')

    ######## HIGH-LEVEL STRATEGIES

    def main_strategy(self):
        while 1:
            with self.preempt([
                self.is_any_mon_on_map,
                lambda: self.blstats.time % 3 == 0 and self.blstats.hunger_state >= Hunger.NOT_HUNGRY and self.is_any_food_on_map(),
                lambda: self.blstats.hunger_state >= Hunger.WEAK and any(map(lambda item: item.category == nh.FOOD_CLASS,
                                                                             self.inventory.values())),
            ]) as outcome:
                if outcome() is None:
                    if self.explore1() is not False:
                        continue

                    if len(self.levels) <= 100 and self.move_down() is not False:
                        continue

                    if self.search1() is not False:
                        continue

            if outcome() == 0:
                self.fight1()
                continue

            if outcome() == 1:
                self.eat1()
                continue

            if outcome() == 2:
                # TODO: refactor
                with self.panic_if_position_changes():
                    self.step(A.Command.EAT)
                for k, item in self.inventory.items():
                    if item.category == nh.FOOD_CLASS:
                        self.enter_text(k)
                        break
                continue

            assert 0

    ####### MAIN

    def main(self):
        self.parse_character()

        try:
            try:
                self.step(A.Command.AUTOPICKUP)
            except AgentChangeStrategy:
                pass

            while 1:
                try:
                    self.main_strategy()
                except AgentPanic as e:
                    self.all_panics.append(e)
                    if self.verbose:
                        print(f'PANIC!!!! : {e}')
                except AgentChangeStrategy:
                    pass
        except AgentFinished:
            pass
