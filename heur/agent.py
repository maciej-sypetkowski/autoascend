import contextlib
import re
from collections import namedtuple
from functools import partial, wraps
from itertools import chain

import nle.nethack as nh
import numpy as np
import toolz
from nle.nethack import actions as A

import utils
from glyph import SS, MON, C, WEA

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

    WEAPONS = {nh.GLYPH_OBJ_OFF + i for i in range(nh.NUM_OBJECTS) if ord(nh.objclass(i).oc_class) == nh.WEAPON_CLASS}

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
        self.checked_item_pile = np.zeros((C.SIZE_Y, C.SIZE_X), bool)


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

    def __init__(self, glyphs, count=1, status=UNKNOWN, modifier=None, worn=False, at_ready=False):
        # glyphs is a list of possible glyphs for this item
        assert isinstance(glyphs, list)
        assert all(map(nh.glyph_is_object, glyphs))
        assert len(glyphs)

        self.glyphs = glyphs
        self.count = count
        self.status = status
        self.modifier = modifier
        self.worn = worn
        self.at_ready = at_ready
        self.category = ord(nh.objclass(nh.glyph_to_obj(self.glyphs[0])).oc_class)

        assert all(map(lambda glyph: ord(nh.objclass(nh.glyph_to_obj(glyph)).oc_class) == self.category, self.glyphs))

    def base_cost(self):
        assert len(self.glyphs) == 1, 'TODO: what in this case?'
        return nh.objclass(nh.glyph_to_obj(self.glyphs[0])).oc_cost

    ######## WEAPON

    def is_weapon(self):
        assert self.category != nh.WEAPON_CLASS or len(self.glyphs) == 1, self.glyphs
        return self.category == nh.WEAPON_CLASS

    def get_dps(self, big_monster):
        assert self.is_weapon()
        return WEA.get_dps(self.glyphs[0], big_monster) + (self.modifier if self.modifier is not None else 0)

    def is_launcher(self):
        if not self.is_weapon():
            return False

        return nh.objdescr.from_idx(nh.glyph_to_obj(self.glyphs[0])).oc_name in \
                ['bow', 'elven bow', 'orcish bow', 'yumi', 'crossbow', 'sling']

    def is_thrown_projectile(self):
        if not self.is_weapon():
            return False

        # TODO: boomerang
        return nh.objdescr.from_idx(nh.glyph_to_obj(self.glyphs[0])).oc_name in \
                ['orcish dagger', 'dagger silver', 'athame dagger', 'elven dagger',
                 'worm tooth', 'knife', 'stiletto', 'scalpel', 'crysknife',
                 'dart', 'shuriken', ]

    def __str__(self):
        return (f'{self.count}_'
                f'{self.status if self.status is not None else ""}_'
                f'{self.modifier if self.modifier is not None else ""}_'
                f'{",".join([nh.objdescr.from_idx(nh.glyph_to_obj(glyph)).oc_name for glyph in self.glyphs])}'
                )


class ItemManager:
    def __init__(self, agent):
        self.agent = agent

    def get_item_from_text(self, text, category=None, glyph=None):
        # TODO: there are some problems with 'inv_glyphs' and the environment gives incorrect inventory glyphs.
        #       I'm ignoring them for now
        glyph = None

        assert category is not None or glyph is not None
        assert glyph is None or nh.glyph_is_normal_object(glyph)

        if category is None:
            category = ord(nh.objclass(nh.glyph_to_obj(glyph)).oc_class)
        else:
            assert glyph is None or category == ord(nh.objclass(nh.glyph_to_obj(glyph)).oc_class)

        assert category not in [nh.BALL_CLASS, nh.ROCK_CLASS, nh.RANDOM_CLASS]

        matches = re.findall(
            '^(a|an|\d+)( (cursed|uncursed|blessed))?( (very |thoroughly )?(rustproof|poisoned|corroded|rusty|burnt|rotted))*( ([+-]\d+))? ([a-zA-z0-9- ]+)( \(([0-9]+:[0-9]+)\))?( \(([a-zA-Z0-9; ]+)\))?$',
            text)
        assert len(matches) <= 1, text
        assert len(matches), text

        count, _, status, effects, _, _, _, modifier, name, _, uses, _, info = matches[0]
        # TODO: effects, uses

        if info in {'weapon in paw', 'weapon in hand', 'weapon in paws', 'weapon in hands', 'being worn', 'being worn; slippery', 'wielded'}:
            worn = True
            at_ready = False
        elif info in {'at the ready', 'in quiver', 'in quiver pouch'}:
            worn = False
            at_ready = True
        elif info in {'', 'alternate weapon; not wielded'}:
            worn = False
            at_ready = False
        else:
            assert 0, info

        count = int({'a': 1, 'an': 1}.get(count, count))
        status = {'': Item.UNKNOWN, 'cursed': Item.CURSED, 'uncursed': Item.UNCURSED, 'blessed': Item.BLESSED}[status]
        modifier = None if not modifier else {'+': 1, '-': -1}[modifier[0]] * int(modifier[1:])

        if category == nh.WEAPON_CLASS:
            name_augmentation = lambda x: [x, f'{x}s']
        elif category == nh.ARMOR_CLASS:
            name_augmentation = lambda x: [x, f'pair of {x}']
        elif category in [nh.AMULET_CLASS, nh.FOOD_CLASS, nh.GEM_CLASS, nh.POTION_CLASS, nh.RING_CLASS,
                          nh.SCROLL_CLASS, nh.SPBOOK_CLASS, nh.TOOL_CLASS, nh.WAND_CLASS, nh.COIN_CLASS]:
            return Item(
                [i + nh.GLYPH_OBJ_OFF for i in range(nh.NUM_OBJECTS) if ord(nh.objclass(i).oc_class) == category and
                 nh.objdescr.from_idx(i).oc_name is not None],
                count, status, modifier, worn, at_ready)
        else:
            assert 0, category

        ret = []

        if name == 'wakizashi':
            name = 'short sword'
        elif name == 'ninja-to':
            name = 'broadsword'
        elif name == 'nunchaku':
            name == 'flail'
        elif name == 'shito':
            name == 'knife'
        elif name == 'naginata':
            name == 'glaive'

        for i in range(nh.NUM_OBJECTS):
            if ord(nh.objclass(i).oc_class) != category:
                continue

            obj_descr = nh.objdescr.from_idx(i).oc_descr
            obj_name = nh.objdescr.from_idx(i).oc_name
            if (obj_name and name in name_augmentation(obj_name)) or (
                    obj_descr and name in name_augmentation(obj_descr)):
                ret.append(i)

        assert len(ret) == 1, (ret, name, text)
        assert glyph is None or ret[0] == nh.glyph_to_obj(glyph), \
               ((ret[0], nh.objdescr.from_idx(ret[0])), (nh.glyph_to_obj(glyph), nh.objdescr.from_idx(nh.glyph_to_obj(glyph))))
        return Item([r + nh.GLYPH_OBJ_OFF for r in ret], count, status, modifier, worn, at_ready)


@toolz.curry
def debug_log(txt, fun, color=(255, 255, 255)):
    @wraps(fun)
    def wrapper(self, *args, **kwargs):
        with self.env.debug_log(txt=txt, color=color):
            return fun(self, *args, **kwargs)

    return wrapper


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

        self.inventory = {}
        self.character = 'x-x-x-x'

        self.last_bfs_dis = None
        self.last_bfs_step = None

        self.previous_inv_strs = None
        self.turns_in_atom_operation = None

        self.item_manager = ItemManager(self)

        self._is_reading_message_or_popup = False

    ######## CONVENIENCE FUNCTIONS

    @contextlib.contextmanager
    def atom_operation(self, max_different_turns=1):
        if self.turns_in_atom_operation is not None:
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

    #@contextlib.contextmanager
    #def stop_updating(self, update_at_end=False):
    #    on_update = self.on_update
    #    self.on_update = []

    #    try:
    #        yield
    #    finally:
    #        assert self.on_update == []
    #        self.on_update = on_update

    #    if update_at_end:
    #        for f in self.on_update:
    #            f()

    @contextlib.contextmanager
    def preempt(self, conditions):
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
        for fun in self.on_update:
            fun()

    ######## UPDATE FUNCTIONS

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
            message_preffix = self.message + ' '
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

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.step_count += 1
        observation = {k: v.copy() for k, v in observation.items()}
        self.score += reward

        if done:
            raise AgentFinished()

        self.update(observation)

    def update(self, observation):
        should_update = True

        self.message, self.popup, done = self.get_message_and_popup(observation)
        if not done:
            self.step(A.TextCharacters.SPACE)
            return

        if observation['misc'][1]:  # entering text
            self.step(A.Command.ESC)
            return

        if b'[yn]' in bytes(observation['tty_chars'].reshape(-1)):
            self.enter_text('y')
            return

        # FIXME: self.update_state() won't be called on all states sometimes.
        #        Otherwise there are problems with atomic operations.

        if self.turns_in_atom_operation is not None:
            should_update = False
            if any([(self.last_observation[key] != observation[key]).any()
                    for key in ['glyphs', 'blstats', 'inv_strs', 'inv_letters', 'inv_oclasses', 'inv_glyphs']]):
                self.turns_in_atom_operation += 1
            assert self.turns_in_atom_operation in [0, 1]

        self.last_observation = observation

        self.blstats = BLStats(*self.last_observation['blstats'])
        self.glyphs = self.last_observation['glyphs']

        if should_update:
            self.update_state()

    def update_state(self):
        self.update_level()
        self.update_inventory()

        for func in self.on_update:
            func()

    def update_inventory(self):
        if self.previous_inv_strs is not None and (self.last_observation['inv_strs'] == self.previous_inv_strs).all():
            return

        self.inventory = {}
        for item_name, category, glyph, letter in zip(
                self.last_observation['inv_strs'],
                self.last_observation['inv_oclasses'],
                self.last_observation['inv_glyphs'],
                self.last_observation['inv_letters']):
            item_name = bytes(item_name).decode().strip('\0')
            letter = chr(letter)
            if not item_name:
                continue
            item = self.item_manager.get_item_from_text(item_name, category=category, glyph=glyph)
            self.inventory[letter] = item

        self.previous_inv_strs = self.last_observation['inv_strs']

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

    def glyphs_mask_in(self, *gset):
        gset = list(chain(*gset))
        return np.isin(self.glyphs, gset)

    @staticmethod
    def calc_direction(from_y, from_x, to_y, to_x, allow_nonunit_distance=False):
        if allow_nonunit_distance:
            assert from_y == to_y or from_x == to_x or abs(from_y - to_y) == abs(from_x - to_x), ((from_y, from_x), (to_y, to_x))
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

    def parse_character(self):
        with self.atom_operation():
            self.step(A.Command.ATTRIBUTES)
            text = ' '.join(self.popup)
            self.character = CH.parse(text)

    def enter_text(self, text):
        with self.atom_operation():
            for char in text:
                char = ord(char)
                self.step(A.ACTIONS[A.ACTIONS.index(char)])

    def eat(self):  # TODO: eat what
        with self.atom_operation():
            self.step(A.Command.EAT)
            if "You don't have anything to eat." in self.message:
                return False
            self.enter_text('y')
            if "You don't have that object." in self.message:
                self.step(A.Command.ESC)
                return False
        return True

    def take_item(self):  # TODO: take what
        with self.atom_operation():
            self.step(A.Command.PICKUP)
            self.step(A.Command.ESC)
        return True

    def wield(self, letter):
        with self.atom_operation():
            self.step(A.Command.WIELD)
            self.enter_text(letter)
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

    def fire(self, letter, direction):
        assert letter in self.inventory
        with self.atom_operation():
            self.step(A.Command.THROW)
            self.enter_text(letter)
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

        walkable = level.walkable & ~self.glyphs_mask_in(G.SHOPKEEPER, G.BOULDER)

        dis = utils.bfs(y, x,
                        walkable=walkable,
                        walkable_diagonally=walkable & ~np.isin(level.objects, list(G.DOORS)) & (level.objects != -1))

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

    @debug_log('fight1')
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

            if abs(self.blstats.y - y) > 1 or abs(self.blstats.x - x) > 1:
                throwable = {k: v for k, v in self.inventory.items() if v.is_thrown_projectile() and not v.worn}
                if throwable and (self.blstats.y == y or self.blstats.x == x or abs(self.blstats.y - y) == abs(self.blstats.x - x)):
                    dir = self.calc_direction(self.blstats.y, self.blstats.x, y, x, allow_nonunit_distance=True)
                    self.fire(list(throwable.keys())[0], dir)
                    continue

                self.go_to(y, x, stop_one_before=True, max_steps=1,
                           debug_tiles_args=dict(color=(255, 0, 0), is_path=True))
                continue

            mon = nh.glyph_to_mon(self.glyphs[y, x])

            try:
                self.fight(y, x)
            finally:  # TODO: what if panic?
                if nh.glyph_is_body(self.glyphs[y, x]) and self.glyphs[y, x] - nh.GLYPH_BODY_OFF == mon:
                    self.current_level().corpse_age[y, x] = self.blstats.time

    @debug_log('eat1')
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

        self.go_to(target_y, target_x, debug_tiles_args=dict(color=(255, 255, 0), is_path=True))
        if not self.current_level().shop[self.blstats.y, self.blstats.x]:
            self.eat()  # TODO: what

    @debug_log('explore1')
    def explore1(self, search_prio_limit=0):
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
                        to_visit |= utils.translate(~level.seen & self.glyphs_mask_in(G.STONE), dy, dx)
                        if dx == 0 or dy == 0:
                            to_visit |= utils.translate(self.glyphs_mask_in(G.DOOR_CLOSED), dy, dx)
            return to_visit

        def to_search_func(prio_limit=0, return_prio=False):
            level = self.current_level()
            dis = self.bfs()

            prio = np.zeros((C.SIZE_Y, C.SIZE_X), np.float32)
            prio[:] = -1
            prio -= level.search_count ** 2 * 2
            is_on_corridor = np.isin(level.objects, list(G.CORRIDOR))
            is_on_door = np.isin(level.objects, list(G.DOORS))

            stones = np.zeros((C.SIZE_Y, C.SIZE_X), np.int32)
            walls = np.zeros((C.SIZE_Y, C.SIZE_X), np.int32)

            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy != 0 or dx != 0:
                        stones += np.isin(utils.translate(level.objects, dy, dx), list(G.STONE))
                        walls += np.isin(utils.translate(level.objects, dy, dx), list(G.WALL))

            prio += (is_on_door & (stones > 3)) * 250
            prio += (np.stack([utils.translate(level.walkable, y, x).astype(np.int32)
                               for y, x in [(1, 0), (-1, 0), (0, 1), (0, -1)]]).sum(0) <= 1) * 250
            prio[(stones == 0) & (walls == 0)] = -np.inf

            prio[~level.walkable | (dis == -1)] = -np.inf

            if return_prio:
                return prio
            return prio >= prio_limit

        def open_visit_search(search_prio_limit):
            while 1:
                open_neighbor_doors()
                to_visit  = to_visit_func()
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
                        return
                    nonzero_y, nonzero_x = ((search_prio == search_prio[to_explore].max()) & to_explore).nonzero()

                # select random closest to_explore tile
                i = self.rng.randint(len(nonzero_y))
                target_y, target_x = nonzero_y[i], nonzero_x[i]

                with self.env.debug_tiles(to_explore, color=(0, 0, 255, 64)):
                    self.go_to(target_y, target_x, debug_tiles_args=dict(
                        color=(255 * bool(to_visit[target_y, target_x]), 255, 255 * bool(to_search[target_y, target_x])),
                        is_path=True))
                    if to_search[target_y, target_x] and not to_visit[target_y, target_x]:
                        self.search()

        def check_item_pile():
            mask = (((self.last_observation['specials'] & nh.MG_OBJPILE) > 0) & (self.bfs() != -1) & 
                    ~self.current_level().checked_item_pile)

            dis = self.bfs()
            nonzero_y, nonzero_x = (mask & (dis == dis[mask].min())).nonzero()
            i = self.rng.randint(len(nonzero_y))
            target_y, target_x = nonzero_y[i], nonzero_x[i]

            #with self.env.debug_tiles(mask, color=(255, 0, 0, 128)):
            #    # TODO: search for traps before stepping in
            #    self.go_to(target_y, target_x, debug_tiles_args=dict(color=(255, 0, 0), is_path=True))

            self.current_level().checked_item_pile[target_y, target_x] = True

        def take_item(only_check=False):
            return False # TODO: ignore for now

            if self.character.role == CH.MONK:
                return False
            for item in self.inventory.values():
                if item.is_weapon() and item.worn:
                    if item.status == Item.CURSED:
                        return
                    current_weapon_dps = item.get_dps(big_monster=False) # TODO: what about monster size
                    current_weapon = item
                    break
            else:
                current_weapon_dps = 0
                current_weapon = 'fists'

            current_weapon_dps += 2  # take only relatively better items than yours

            mask = ((self.last_observation['specials'] & nh.MG_OBJPILE) == 0) & ~self.current_level().shop & (self.bfs() != -1) & self.glyphs_mask_in(G.WEAPONS)
            nonzero_y, nonzero_x = mask.nonzero()

            best_item = None
            best_item_dps = None
            for y, x in zip(nonzero_y, nonzero_x):
                glyph = self.glyphs[y, x]
                dps = Item([self.glyphs[y, x]]).get_dps(big_monster=False)
                if dps > current_weapon_dps:
                    best_item_dps = dps
                    best_item = (y, x)

            if best_item is None:
                return False

            if only_check:
                return True

            with self.env.debug_log(f'going for {Item([self.glyphs[best_item]])}'):
                target_y, target_x = best_item
                self.go_to(target_y, target_x, debug_tiles_args=dict(color=(255, 0, 255), is_path=True))
                if self.current_level().shop[target_y, target_x]:
                    return
                self.take_item()

                # select the best
                best_item = None
                best_dps = None
                for letter, item in self.inventory.items():
                    if item.is_weapon():
                        dps = item.get_dps(big_monster=False) # TODO: what about monster size
                        if best_dps is None or best_dps < dps:
                            best_dps = dps
                            best_item = letter

                self.wield(letter)
                if self.blstats.carrying_capacity != 0:
                    print(self.blstats.carrying_capacity)


        while 1:
            with self.preempt([
                lambda: (((self.last_observation['specials'] & nh.MG_OBJPILE) > 0) & (self.bfs() != -1) & 
                         ~self.current_level().checked_item_pile).any(),
                lambda: take_item(only_check=True)
            ]) as outcome:
                if outcome() is None:
                    open_visit_search(search_prio_limit)
                    break

            if outcome() == 0:
                check_item_pile()
                continue

            if outcome() == 1:
                take_item()
                continue

            assert 0, outcome()


    @debug_log('move_down')
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

        assert pos is not None

        dis = self.bfs()
        if dis[pos] == -1:
            return False

        target_y, target_x = pos

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

    ######## HIGH-LEVEL STRATEGIES

    @debug_log('main_strategy')
    def main_strategy(self):
        while 1:
            with self.preempt([
                self.is_any_mon_on_map,
            ]) as outcome1:
                if outcome1() is None:
                    with self.preempt([
                        lambda: self.blstats.time % 3 == 0 and self.blstats.hunger_state >= Hunger.NOT_HUNGRY and \
                                self.is_any_food_on_map(),
                        lambda: self.blstats.hunger_state >= Hunger.WEAK and any(
                            map(lambda item: item.category == nh.FOOD_CLASS,
                                self.inventory.values())),
                    ]) as outcome2:
                        if outcome2() is None:

                            self.explore1(0)

                            with self.preempt([
                                # TODO: implement it better
                                lambda: np.isin(self.current_level().objects, list(G.STAIR_DOWN)).any() and
                                        (np.isin(self.current_level().objects, list(G.STAIR_DOWN)) & (self.bfs() != -1)).any() and self.blstats.hitpoints >= 0.8 * self.blstats.max_hitpoints,
                            ]) as outcome3:
                                if outcome3() is None:

                                    self.explore1(None)

                            if outcome3() == 0:
                                self.move_down()
                                continue

                            assert 0, outcome3()

                    if outcome2() == 0:
                        self.eat1()
                        continue

                    if outcome2() == 1:
                        # TODO: refactor
                        with self.atom_operation():
                            self.step(A.Command.EAT)
                            for k, item in self.inventory.items():
                                if item.category == nh.FOOD_CLASS:
                                    self.enter_text(k)
                                    break
                            else:
                                assert 0
                        continue

                    assert 0, outcome2()

            if outcome1() == 0:
                self.fight1()
                continue

            assert 0, outcome()

    ####### MAIN

    def main(self):
        self.update({k: v.copy() for k, v in self.env.reset().items()})
        self.parse_character()

        try:
            self.step(A.Command.AUTOPICKUP)

            while 1:
                try:
                    self.main_strategy()
                except AgentPanic as e:
                    self.all_panics.append(e)
                    if self.verbose:
                        print(f'PANIC!!!! : {e}')
        except AgentFinished:
            pass
