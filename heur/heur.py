import numpy as np
from collections import namedtuple
import gym
import nle
from nle.nethack import actions as A
import nle.nethack as nh
from glyph import SS, MON, C, ALL
from itertools import chain
import operator
from functools import partial
from pprint import pprint
import time


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


    DICT = {k: v for k, v in locals().items() if not k.startswith('_')}

    @classmethod
    def assert_map(cls, glyphs, chars):
        for glyph, char in zip(glyphs.reshape(-1), chars.reshape(-1)):
            char = bytes([char]).decode()
            for k, v in cls.__annotations__.items():
                assert glyph not in cls.DICT[k] or char in v, f'{k} {v} {glyph} {char}'

G.INV_DICT = {i: [k for k, v in G.DICT.items() if i in v]
              for i in set.union(*map(set, G.DICT.values()))}


class AgentFinished(Exception):
    pass

class AgentPanic(Exception):
    pass



class Agent:
    def __init__(self, env, seed=0):
        self.env = env
        self.last_observation = env.reset()
        self.rng = np.random.RandomState(seed)

        self.score = 0
        self.update_map()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.last_observation = obs
        self.score += reward
        if done:
            raise AgentFinished()

        self.update_map()

        return obs, reward, done, info

    def update_map(self, obs=None):
        if obs is None:
            obs = self.last_observation

        self.blstats = BLStats(*obs['blstats'])
        self.glyphs = obs['glyphs']

        self.update_level()

        if b'--More--' in bytes(obs['tty_chars'].reshape(-1)):
            self.step(A.Command.ESC)


    ######## TRIVIAL ACTIONS

    def direction(self, from_y, from_x, to_y, to_x):
        assert abs(from_y - to_y) <= 1 and abs(from_x - to_x) <= 1

        ret = ''
        if to_y == from_y + 1: ret += 's'
        if to_y == from_y - 1: ret += 'n'
        if to_x == from_x + 1: ret += 'e'
        if to_x == from_x - 1: ret += 'w'
        if ret == '': ret = '.'

        return ret

    def move(self, y, x=None):
        if y is not None:
            # point
            y = self.direction(self.blstats.y, self.blstats.x, y, x)
            assert y != '.'

        # direction
        action = {
            'n': A.CompassDirection.N, 's': A.CompassDirection.S,
            'e': A.CompassDirection.E, 'w': A.CompassDirection.W,
            'ne': A.CompassDirection.NE, 'se': A.CompassDirection.SE,
            'nw': A.CompassDirection.NW, 'sw': A.CompassDirection.SW,
        }[y]

        expected_y = self.blstats.y + ('e' in y) - ('w' in y)
        expected_x = self.blstats.x + ('s' in y) - ('n' in y)

        self.step(action)

        if self.blstats.y != expected_y or self.blstats.x != expected_x:
            raise AgentPanic()

    ########

    def neighbors(self, y, x, shuffle=True):
        ret = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny = y + dy
                nx = x + dx
                if 0 <= ny < C.SIZE_Y and 0 <= nx < C.SIZE_X:
                    ret.append((ny, nx))

        if shuffle:
            self.rng.shuffle(ret)

        return ret

    class Level:
        def __init__(self):
            self.walkable = np.zeros((C.SIZE_Y, C.SIZE_X), bool)
            self.seen = np.zeros((C.SIZE_Y, C.SIZE_X), bool)
            self.objects = np.zeros((C.SIZE_Y, C.SIZE_X), np.int16)
            self.objects[:] = -1

    levels = {}

    def current_level(self):
        key = (self.blstats.dungeon_number, self.blstats.level_number)
        if key not in self.levels:
            self.levels[key] = self.Level()
        return self.levels[key]

    def update_level(self):
        level = self.current_level()

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
                             [G.MONS, G.PETS])):
                    level.seen[y, x] = True
                    level.walkable[y, x] = True

        for y, x in self.neighbors(self.blstats.y, self.blstats.x):
            if self.glyphs[y, x] in G.STONE:
                level.seen[y, x] = True
                level.objects[y, x] = self.glyphs[y, x]

    def bfs(self, y, x):
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

            for py, px in self.neighbors(y, x):
                if (level.walkable[py, px] and
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


    ######## STRATEGIES ACTIONS

    def explore1(self):
        level = self.current_level()
        to_explore = np.zeros((C.SIZE_Y, C.SIZE_X), dtype=bool)
        dis = self.bfs(self.blstats.y, self.blstats.x)
        for y in range(C.SIZE_Y):
            for x in range(C.SIZE_X):
                if dis[y, x] != -1:
                    for py, px in self.neighbors(y, x):
                        if not level.seen[py, px] and self.glyphs[py, px] in G.STONE:
                            to_explore[y, x] = True
                            break

        nonzero_y, nonzero_x = \
                (dis == (dis * (to_explore) - 1).astype(np.uint16).min() + 1).nonzero()
        nonzero = [(y, x) for y, x in zip(nonzero_y, nonzero_x) if to_explore[y, x]]
        if len(nonzero) == 0:
            # TODO
            while 1:
                self.step(A.Command.ESC)
            return

        nonzero_y, nonzero_x = zip(*nonzero)
        ty, tx = nonzero_y[0], nonzero_x[0]

        #for asd in to_explore:
        #    print(str(asd.astype(np.int8).tolist()).replace(',', '').replace(' ', '').replace('-1', 'x')[1:-1])
        del level


        path = self.path(self.blstats.y, self.blstats.x, ty, tx, dis=dis)
        for y, x in path[1:]:
            if not self.current_level().walkable[y, x]:
                return
            self.move(y, x)

    def select_strategy(self):
        self.explore1()


    ####### MAIN

    def main(self):
        try:
            while 1:
                try:
                    self.select_strategy()
                except AgentPanic:
                    pass
        except AgentFinished:
            pass


class EnvWrapper:
    def __init__(self, env):
        self.env = env

    def reset(self):
        print('\n' * 100)
        obs = self.env.reset()
        self.render(obs)

        G.assert_map(obs['glyphs'], obs['chars'])

        blstats = BLStats(*obs['blstats'])
        assert obs['chars'][blstats.y, blstats.x] == ord('@')

        return obs

    def render(self, obs):
        print(bytes(obs['message']).decode())
        print()
        print(BLStats(*obs['blstats']))
        print()
        for letter, text in zip(obs['inv_letters'], obs['inv_strs']):
            if (text != 0).any():
                print(chr(letter), '->', bytes(text).decode())
        print('-' * 20)
        self.env.render()
        print('-' * 20)

    def print_help(self):
        scene_glyphs = set(self.env.last_observation[0].reshape(-1))
        obj_classes = {getattr(nh, x): x for x in dir(nh) if x.endswith('_CLASS')}
        glyph_classes = sorted((getattr(nh, x), x) for x in dir(nh) if x.endswith('_OFF'))

        texts = []
        for i in range(nh.MAX_GLYPH):
            desc = ''
            if glyph_classes and i == glyph_classes[0][0]:
                cls = glyph_classes.pop(0)[1]

            if nh.glyph_is_monster(i):
                desc = f': "{nh.permonst(nh.glyph_to_mon(i)).mname}"'

            if nh.glyph_is_normal_object(i):
                obj = nh.objclass(nh.glyph_to_obj(i))
                appearance = nh.OBJ_DESCR(obj) or nh.OBJ_NAME(obj)
                oclass = ord(obj.oc_class)
                desc = f': {obj_classes[oclass]}: "{appearance}"'

            desc2 = 'Labels: '
            if i in G.INV_DICT:
                desc2 += ','.join(G.INV_DICT[i])

            if i in scene_glyphs:
                pos = (self.env.last_observation[0].reshape(-1) == i).nonzero()[0]
                count = len(pos)
                pos = pos[0]
                char = bytes([self.env.last_observation[1].reshape(-1)[pos]])
                texts.append((-count, f'{" " if i in G.INV_DICT else "U"} Glyph {i:4d} -> '
                                      f'Char: {char} Count: {count:4d} '
                                      f'Type: {cls.replace("_OFF",""):11s} {desc:30s} '
                                      f'{ALL.find(i) if ALL.find(i) is not None else "":20} '
                                      f'{desc2}'))
        for _, t in sorted(texts):
            print(t)

    def get_action(self):
        while 1:
            key = os.read(sys.stdin.fileno(), 3)
            if len(key) != 1:
                print('wrong key', key)
                continue
            key = key[0]
            if key == 63: # '?"
                self.print_help()
                continue
            elif key == 10:
                return None
            else:
                actions = [a for a in self.env._actions if int(a) == key]
                assert len(actions) < 2
                if len(actions) == 0:
                    print('wrong key', key)
                    continue

                action = actions[0]
                return action

    def step(self, agent_action):
        print()
        print('agent_action:', agent_action)
        print()

        action = self.get_action()
        if action is None:
            action = agent_action
        print('\n' * 100)
        print('action:', action)
        print()

        obs, reward, done, info = self.env.step(self.env._actions.index(action))
        self.render(obs)
        G.assert_map(obs['glyphs'], obs['chars'])
        return obs, reward, done, info


class EnvLimitWrapper:
    def __init__(self, env, step_limit):
        self.env = env
        self.cur_step = 0
        self.step_limit = step_limit

    def reset(self):
        self.cur_step = 0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(self.env._actions.index(action))
        self.cur_step += 1
        if self.cur_step == self.step_limit + 1:
            return obs, reward, True, info
        elif self.cur_step > self.step_limit + 1:
            assert 0
        return obs, reward, done, info


if __name__ == '__main__':
    import sys, tty, os, termios

    if len(sys.argv) > 1:
        games = int(sys.argv[1])

        from multiprocessing import Pool, Process, Queue
        from matplotlib import pyplot as plt
        import seaborn as sns
        sns.set()


        def single_simulation(_=None):
            start_time = time.time()
            env = EnvLimitWrapper(gym.make('NetHackChallenge-v0'), 500)
            agent = Agent(env)
            agent.main()
            end_time = time.time()
            return {
                'score': agent.score,
                'steps': env.env._steps,
                'turns': env.env._turns,
                'duration': end_time - start_time,
            }

        with Pool(16) as pool:
            start_time = time.time()

            plot_queue = Queue()
            def plot_thread_func():
                fig = plt.figure()
                plt.show(block=False)
                while 1:
                    try:
                        funcs = plot_queue.get(block=False)
                    except:
                        plt.pause(0.01)
                        continue

                    fig.clear()
                    for func in funcs:
                        func()
                    plt.show(block=False)


            plt_process = Process(target=plot_thread_func)
            plt_process.start()

            all_res = {}
            for single_res in pool.imap(single_simulation, [()] * games):
                if not all_res:
                    all_res = {key: [] for key in single_res}
                assert all_res.keys() == single_res.keys()

                for k, v in single_res.items():
                    all_res[k].append(v)


                plot_queue.put([f for i, k in enumerate(sorted(all_res)) for f in [
                    partial(plt.subplot, len(all_res), 1, i + 1),
                    partial(plt.xlabel, k),
                    partial(sns.histplot, all_res[k]),
                ]])


                total_duration = time.time() - start_time

                print(f'time_per_simulation           : {np.mean(all_res["duration"])}')
                print(f'time_per_turn                 : {np.sum(all_res["duration"]) / np.sum(all_res["turns"])}')
                print(f'turns_per_second              : {np.sum(all_res["turns"]) / np.sum(all_res["duration"])}')
                print(f'turns_per_second(multithread) : {np.sum(all_res["turns"]) / total_duration}')
                print()

    else:
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        try:
            env = EnvWrapper(gym.make('NetHackChallenge-v0'))

            agent = Agent(env)
            agent.main()

        finally:
            os.system('stty sane')
