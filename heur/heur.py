import numpy as np
from collections import namedtuple
import gym
import nle
from nle.nethack import actions as A
import nle.nethack as nh


BLStats = namedtuple('BLStats', 'x y strength_percentage strength dexterity constitution intelligence wisdom charisma score hitpoints max_hitpoints depth gold energy max_energy armor_class monster_level experience_level experience_points time hunger_state carrying_capacity dungeon_number level_number')


class Glyph:
    # TODO: use all constants from display.h, rm.h, etc

    FLOOR : ['.'] = [2371, 2378, 2379]
    UNKNOWN : [' '] = [2359]
    WALL : ['|', '-'] = [2360, 2361, 2362, 2363, 2364, 2365]
    CORRIDOR : ['#'] = [2380]
    STAIR_UP : ['<'] = [2382]

    DOOR_CLOSED : ['+'] = [2374]
    DOOR_OPENED : ['-'] = [2372]

    DICT, INV_DICT = (
        {k: v for k, v in locals().items() if not k.startswith('_')},
        {i: k for k, v in locals().items() for i in v if not k.startswith('_')},
    )

    assert sum([len(v) for v in DICT.values()]) == len(set.union(*map(set, DICT.values())))

    @classmethod
    def assert_map(cls, glyphs, chars):
        for glyph, char in zip(glyphs.reshape(-1), chars.reshape(-1)):
            char = bytes([char]).decode()
            for k, v in cls.__annotations__.items():
                assert glyph not in cls.DICT[k] or char in v, f'{k} {v} {glyph} {char}'


class AgentFinished(Exception):
    pass


class Agent:
    def __init__(self, env):
        self.env = env
        self.last_observation = env.reset()

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

    def move(self, x):
        action = {
            'n': A.CompassDirection.N, 's': A.CompassDirection.S,
            'e': A.CompassDirection.E, 'w': A.CompassDirection.W,
            'ne': A.CompassDirection.NE, 'se': A.CompassDirection.SE,
            'nw': A.CompassDirection.NW, 'sw': A.CompassDirection.SW,
        }[x]

        self.env.step(action)

    def main(self):
        try:
            while 1:
                self.move('n')
        except AgentFinished:
            pass


class EnvWrapper:
    def __init__(self, env):
        self.env = env

    def reset(self):
        print('\n' * 100)
        obs = self.env.reset()
        self.render(obs)
        Glyph.assert_map(obs['glyphs'], obs['chars'])
        return obs

    def render(self, obs):
        print(bytes(obs['message']).decode())
        print(obs.keys())
        print()
        print(BLStats(*obs['blstats']))
        print()
        for letter, text in zip(obs['inv_letters'], obs['inv_strs']):
            if (text != 0).any():
                print(chr(letter), '->', bytes(text).decode())
        print('-' * 20)
        self.env.render()
        print('-' * 20)


    def get_action(self):
        while 1:
            key = os.read(sys.stdin.fileno(), 3)
            if len(key) != 1:
                print('wrong key', key)
                continue
            key = key[0]
            if key == 63: # '?"
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

                    if i in scene_glyphs:
                        pos = (self.env.last_observation[0].reshape(-1) == i).nonzero()[0]
                        count = len(pos)
                        pos = pos[0]
                        char = bytes([self.env.last_observation[1].reshape(-1)[pos]])
                        texts.append((-count, f'{" " if i in Glyph.INV_DICT else "U"} Glyph {i:4d} -> Char: {char} Count: {count} Type: {cls.replace("_OFF","")} {desc}'))
                for _, t in sorted(texts):
                    print(t)

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
        Glyph.assert_map(obs['glyphs'], obs['chars'])
        return obs, reward, done, info


if __name__ == '__main__':
    import sys, tty, os, termios
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        env = EnvWrapper(gym.make('NetHackChallenge-v0'))

        agent = Agent(env)
        agent.main()

    finally:
        os.system('stty sane')
