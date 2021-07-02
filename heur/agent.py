import numpy as np
from collections import namedtuple
from nle.nethack import actions as A
import nle.nethack as nh
from glyph import SS, MON, C, ALL
import operator
import contextlib
from functools import partial


BLStats = namedtuple('BLStats', 'x y strength_percentage strength dexterity constitution intelligence wisdom charisma score hitpoints max_hitpoints depth gold energy max_energy armor_class monster_level experience_level experience_points time hunger_state carrying_capacity dungeon_number level_number')


class G: # Glyphs
    FLOOR : ['.'] = {SS.S_room, SS.S_ndoor, SS.S_darkroom}
    STONE : [' '] = {SS.S_stone}
    WALL : ['|', '-'] = {SS.S_vwall, SS.S_hwall, SS.S_tlcorn, SS.S_trcorn, SS.S_blcorn, SS.S_brcorn,
                         SS.S_crwall, SS.S_tuwall, SS.S_tdwall, SS.S_tlwall, SS.S_trwall}
    CORRIDOR : ['#'] = {SS.S_corr}
    STAIR_UP : ['<'] = {SS.S_upstair}
    STAIR_DOWN : ['>'] = {SS.S_dnstair}

    DOOR_CLOSED : ['+'] = {SS.S_vcdoor, SS.S_hcdoor}
    DOOR_OPENED : ['-', '|'] = {SS.S_vodoor, SS.S_hodoor}
    DOORS = set.union(DOOR_CLOSED, DOOR_OPENED)


    MONS = set(MON.ALL_MONS)
    PETS = set(MON.ALL_PETS)

    SHOPKEEPER = {MON.fn('shopkeeper')}

    BODIES = {nh.GLYPH_BODY_OFF + i for i in range(nh.NUMMONS)}
    OBJECTS = {nh.GLYPH_OBJ_OFF + i for i in range(nh.NUM_OBJECTS) if ord(nh.objclass(i).oc_class) != nh.ROCK_CLASS}
    BIG_OBJECTS = {nh.GLYPH_OBJ_OFF + i for i in range(nh.NUM_OBJECTS) if ord(nh.objclass(i).oc_class) == nh.ROCK_CLASS}


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

class Agent:
    def __init__(self, env, seed=0, verbose=False):
        self.env = env
        self.verbose = verbose
        self.rng = np.random.RandomState(seed)
        self.all_panics = []

        self.on_update = []
        self.levels = {}
        self.last_observation = env.reset()
        self.score = 0
        self.step_count = 0

        self.update_map()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.step_count += 1

        self.last_observation = obs
        self.score += reward
        if done:
            raise AgentFinished()

        self.update_map()

        return obs, reward, done, info

    @contextlib.contextmanager
    def stop_updating(self):
        on_update = self.on_update
        self.on_update = []

        try:
            yield
        finally:
            assert self.on_update == []
            self.on_update = on_update

    @contextlib.contextmanager
    def preempt(self, conditions):
        funcs = []
        for cond in conditions:
            def f(self):
                if cond():
                    raise AgentChangeStrategy(f)
            f = partial(f, self)
            funcs.append(f)
            self.on_update.append(f)

        outcome = None
        for i, f in enumerate(funcs):
            if f():
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

    def update_map(self):
        obs = self.last_observation

        self.blstats = BLStats(*obs['blstats'])
        self.glyphs = obs['glyphs']
        self.message = bytes(obs['message']).decode()

        if b'--More--' in bytes(obs['tty_chars'].reshape(-1)):
            self.step(A.Command.ESC)
            return

        if b'[yn]' in bytes(obs['tty_chars'].reshape(-1)):
            self.enter_text('y')
            return

        self.update_level()

        for func in self.on_update:
            func()

    def current_level(self):
        key = (self.blstats.dungeon_number, self.blstats.level_number)
        if key not in self.levels:
            self.levels[key] = Level()
        return self.levels[key]

    def update_level(self):
        level = self.current_level()

        if '(for sale,' in self.message:
            level.shop[self.blstats.y, self.blstats.x] = 1

        for y in range(C.SIZE_Y):
            for x in range(C.SIZE_X):
                if any(map(lambda s: operator.contains(s, self.glyphs[y, x]),
                           [G.FLOOR, G.CORRIDOR, G.STAIR_UP, G.STAIR_DOWN, G.DOOR_OPENED])):
                    level.walkable[y, x] = True
                    level.seen[y, x] = True
                    level.objects[y, x] = self.glyphs[y, x]
                elif any(map(lambda s: operator.contains(s, self.glyphs[y, x]),
                             [G.WALL, G.DOOR_CLOSED])):
                    level.seen[y, x] = True
                    level.objects[y, x] = self.glyphs[y, x]
                elif any(map(lambda s: operator.contains(s, self.glyphs[y, x]),
                             [G.MONS, G.PETS, G.BODIES, G.OBJECTS])):
                    level.seen[y, x] = True
                    level.walkable[y, x] = True

        for y, x in self.neighbors(self.blstats.y, self.blstats.x):
            if self.glyphs[y, x] in G.STONE:
                level.seen[y, x] = True
                level.objects[y, x] = self.glyphs[y, x]

    ######## TRIVIAL ACTIONS AND HELPERS

    @staticmethod
    def calc_direction(from_y, from_x, to_y, to_x):
        assert abs(from_y - to_y) <= 1 and abs(from_x - to_x) <= 1

        ret = ''
        if to_y == from_y + 1: ret += 's'
        if to_y == from_y - 1: ret += 'n'
        if to_x == from_x + 1: ret += 'e'
        if to_x == from_x - 1: ret += 'w'
        if ret == '': ret = '.'

        return ret

    def enter_text(self, text):
        for char in text:
            char = ord(char)
            self.step(A.ACTIONS[A.ACTIONS.index(char)])

    def eat(self): # TODO: eat what
        self.step(A.Command.EAT)
        self.enter_text('y')
        self.step(A.Command.ESC)
        self.step(A.Command.ESC)
        return True # TODO: return value

    def open_door(self, y, x=None):
        assert self.glyphs[y, x] in G.DOOR_CLOSED
        self.direction(y, x)
        return self.glyphs[y, x] not in G.DOOR_CLOSED

    def fight(self, y, x=None):
        assert self.glyphs[y, x] in G.MONS
        self.direction(y, x)
        return True

    def kick(self, y, x=None):
        with self.stop_updating():
            self.step(A.Command.KICK)
            self.move(y, x)

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

        return ret

    def bfs(self, y=None, x=None):
        if y is None:
            y = self.blstats.y
        if x is None:
            x = self.blstats.x

        level = self.current_level()

        dis = np.zeros((C.SIZE_Y, C.SIZE_X), dtype=np.int16)
        dis[:] = -1
        dis[y, x] = 0

        buf = np.zeros((C.SIZE_Y * C.SIZE_X, 2), dtype=np.uint16)
        index = 0
        buf[index] = (y, x)
        size = 1
        while index < size:
            y, x = buf[index]
            index += 1

            # TODO: handle situations
            # dir: SE
            # @|
            # -.
            # TODO: debug diagonal moving into and from doors
            for py, px in self.neighbors(y, x):
                if (level.walkable[py, px] and self.glyphs[py, px] not in G.SHOPKEEPER and
                    (abs(py - y) + abs(px - x) <= 1 or
                     (level.objects[py, px] not in G.DOORS and
                      level.objects[y, x] not in G.DOORS))):
                    if dis[py, px] == -1:
                        dis[py, px] = dis[y, x] + 1
                        buf[size] = (py, px)
                        size += 1

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
                if dis[y, x] < dis[cur_y, cur_x] and dis[y, x] >= 0:
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
        dis = self.bfs()
        for y in range(C.SIZE_Y):
            for x in range(C.SIZE_X):
                if y != self.blstats.y or x != self.blstats.x:
                    if dis[y, x] != -1:
                        if self.glyphs[y, x] in G.MONS and self.glyphs[y, x] not in G.SHOPKEEPER:
                            return True
        return False

    def is_any_food_on_map(self):
        level = self.current_level()
        dis = self.bfs()
        for y in range(C.SIZE_Y):
            for x in range(C.SIZE_X):
                if dis[y, x] != -1 and not level.shop[y, x]:
                    if self.glyphs[y, x] in G.BODIES and self.blstats.time - level.corpse_age[y, x] <= 20:
                        return True
                    if nh.glyph_is_normal_object(self.glyphs[y, x]):
                        obj = nh.objclass(nh.glyph_to_obj(self.glyphs[y, x]))
                        if ord(obj.oc_class) == nh.FOOD_CLASS:
                            return True
        return False


    ######## NON-TRIVIAL ACTIONS

    ######## LOW-LEVEL STRATEGIES

    def fight1(self):
        dis = self.bfs()
        closest = None

        # TODO: iter by distance
        for y in range(C.SIZE_Y):
            for x in range(C.SIZE_X):
                if y != self.blstats.y or x != self.blstats.x:
                    if self.glyphs[y, x] in G.MONS and self.glyphs[y, x] not in G.SHOPKEEPER:
                        if dis[y, x] != -1 and (closest is None or dis[y, x] < dis[closest]):
                            closest = (y, x)

        assert closest is not None
        #if closest is None:
        #    return False

        y, x = closest
        path = self.path(self.blstats.y, self.blstats.x, y, x)[1:] # TODO: allow diagonal fight from doors

        if len(path) == 1:
            y, x = path[0]
            mon = nh.glyph_to_mon(self.glyphs[y, x])
            try:
                self.fight(y, x)
            finally: # TODO: what if panic?
                if nh.glyph_is_body(self.glyphs[y, x]) and self.glyphs[y, x] - nh.GLYPH_BODY_OFF == mon:
                    self.current_level().corpse_age[y, x] = self.blstats.time

        else:
            self.move(*path[0])

    def eat1(self):
        dis = self.bfs()
        closest = None

        level = self.current_level()
        # TODO: iter by distance
        for y in range(C.SIZE_Y):
            for x in range(C.SIZE_X):
                if dis[y, x] != -1 and (closest is None or dis[y, x] < dis[closest]) and not level.shop[y, x]:
                    if self.glyphs[y, x] in G.BODIES and self.blstats.time - level.corpse_age[y, x] <= 20:
                        closest = (y, x)
                    if nh.glyph_is_normal_object(self.glyphs[y, x]):
                        obj = nh.objclass(nh.glyph_to_obj(self.glyphs[y, x]))
                        if ord(obj.oc_class) == nh.FOOD_CLASS:
                            closest = (y, x)

        assert closest is not None
        #if closest is None:
        #    return False

        ty, tx = closest
        path = self.path(self.blstats.y, self.blstats.x, ty, tx)

        for y, x in path[1:]:
            if self.glyphs[y, x] in G.SHOPKEEPER:
                return
            self.move(y, x)
        if not self.current_level().shop[self.blstats.y, self.blstats.x]:
            self.eat() # TODO: what

    def explore1(self):
        for py, px in self.neighbors(self.blstats.y, self.blstats.x, diagonal=False):
            if self.glyphs[py, px] in G.DOOR_CLOSED:
                if not self.open_door(py, px):
                    while self.glyphs[py, px] in G.DOOR_CLOSED:
                        self.kick(py, px)
                break

        level = self.current_level()
        to_explore = np.zeros((C.SIZE_Y, C.SIZE_X), dtype=bool)
        dis = self.bfs()
        for y in range(C.SIZE_Y):
            for x in range(C.SIZE_X):
                if dis[y, x] != -1:
                    for py, px in self.neighbors(y, x):
                        if not level.seen[py, px] and self.glyphs[py, px] in G.STONE:
                            to_explore[y, x] = True
                            break
                    for py, px in self.neighbors(y, x, diagonal=False):
                        if self.glyphs[py, px] in G.DOOR_CLOSED:
                            to_explore[y, x] = True
                            break

        nonzero_y, nonzero_x = \
                (dis == (dis * (to_explore) - 1).astype(np.uint16).min() + 1).nonzero()
        nonzero = [(y, x) for y, x in zip(nonzero_y, nonzero_x) if to_explore[y, x]]
        if len(nonzero) == 0:
            return False

        nonzero_y, nonzero_x = zip(*nonzero)
        ty, tx = nonzero_y[0], nonzero_x[0]

        del level

        path = self.path(self.blstats.y, self.blstats.x, ty, tx, dis=dis)
        for y, x in path[1:]:
            if not self.current_level().walkable[y, x]:
                return
            if self.glyphs[y, x] in G.SHOPKEEPER:
                return
            self.move(y, x)

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

        ty, tx = nonzero_y[0], nonzero_x[0]
        path = self.path(self.blstats.y, self.blstats.x, ty, tx, dis=dis)
        for y, x in path[1:]:
            if not self.current_level().walkable[y, x]:
                return
            if self.glyphs[y, x] in G.SHOPKEEPER:
                return
            self.move(y, x)
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

        ty, tx = pos

        path = self.path(self.blstats.y, self.blstats.x, ty, tx, dis=dis)
        for y, x in path[1:]:
            if not self.current_level().walkable[y, x]:
                return
            self.move(y, x)

        self.direction('>')


    ######## HIGH-LEVEL STRATEGIES

    def main_strategy(self):
        while 1:
            with self.preempt([
                        self.is_any_mon_on_map,
                        lambda: self.blstats.time % 3 == 0 and self.blstats.hunger_state >= Hunger.NOT_HUNGRY and self.is_any_food_on_map(),
                    ]) as outcome:
                if outcome() is None:
                    if self.explore1() is not False:
                        continue

                    if self.move_down() is not False:
                        continue

                    if self.search1() is not False:
                        continue

            if outcome() == 0:
                self.fight1()
                continue

            if outcome() == 1:
                self.eat1()
                continue

            assert 0


    ####### MAIN

    def main(self):
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
