import gym
import json
import multiprocessing.pool
import nle
import nle.nethack as nh
import numpy as np
import os
import sys
import sys
import termios
import time
import traceback
import tty
from collections import Counter
from nle.nethack import actions as A
from pathlib import Path

from agent import Agent, BLStats, G
from glyph import ALL


class EnvWrapper:
    def __init__(self, env, skip_to=0):
        self.env = env
        self.skip_to = skip_to

    def reset(self):
        print('\n' * 100)
        obs = self.env.reset()
        self.score = 0
        self.step_count = 0
        self.render(obs)

        G.assert_map(obs['glyphs'], obs['chars'])

        blstats = BLStats(*obs['blstats'])
        assert obs['chars'][blstats.y, blstats.x] == ord('@')

        return obs

    def render(self, obs):
        print(bytes(obs['message']).decode())
        print()
        print(BLStats(*obs['blstats']))
        print('Score:', self.score)
        print('Steps:', self.env._steps)
        print('Turns:', self.env._turns)
        print('Seed :', self.env.get_seeds())
        print()
        for letter, text in zip(obs['inv_letters'], obs['inv_strs']):
            if (text != 0).any():
                print(chr(letter), '->', bytes(text).decode())
        print('-' * 20)
        self.env.render()
        print('-' * 20)
        print()

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
                                      f'Type: {cls.replace("_OFF", ""):11s} {desc:30s} '
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
            if key == 10:
                key = 13
            if key == 63:  # '?"
                self.print_help()
                continue
            elif key == 127:  # Backspace
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

        if self.step_count < self.skip_to:
            action = None
        else:
            action = self.get_action()

        if action is None:
            action = agent_action

        print('\n' * 10)
        print('action:', action)
        print()

        obs, reward, done, info = self.env.step(self.env._actions.index(action))
        self.score += reward
        self.step_count += 1
        self.render(obs)
        if not done:
            G.assert_map(obs['glyphs'], obs['chars'])
        return obs, reward, done, info


class EnvLimitWrapper:
    def __init__(self, env, step_limit):
        self.env = env
        self.step_limit = step_limit

    def reset(self):
        self.cur_step = 0
        self.last_turn = 0
        self.score = 0
        self.levels = set()
        self.end_reason = ''
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(self.env._actions.index(action))
        self.score += reward
        blstats = BLStats(*obs['blstats'])
        if self.levels.add((blstats.dungeon_number, blstats.level_number)) != (0, 0):
            self.levels.add((blstats.dungeon_number, blstats.level_number))
        self.cur_step += 1
        self.last_turn = max(self.last_turn, self.env._turns)
        if done:
            end_reason = bytes(obs['tty_chars'].reshape(-1)).decode().replace('You made the top ten list!', '').split()
            if end_reason[7].startswith('Agent'):
                self.score = int(end_reason[6])
                end_reason = ' '.join(end_reason[8:-2])
            else:
                assert self.score == 0
                end_reason = ' '.join(end_reason[7:-2])
            self.end_reason = '.'.join(end_reason.split('.')[1:]).strip()
        if self.cur_step == self.step_limit + 1:
            self.end_reason = self.end_reason or 'steplimit'
            return obs, reward, True, info
        elif self.cur_step > self.step_limit + 1:
            assert 0
        return obs, reward, done, info


def single_simulation(seed):
    start_time = time.time()
    env = EnvLimitWrapper(gym.make('NetHackChallenge-v0'), 10000)
    env.env.seed(seed, seed)
    agent = Agent(env, verbose=False)

    pool = multiprocessing.pool.ThreadPool(processes=1)
    try:
        pool.apply_async(agent.main).get(60)
    except multiprocessing.context.TimeoutError:
        env.end_reason = 'timeout'
    except BaseException as e:
        env.end_reason = f'exception: {"".join(traceback.format_exception(None, e, e.__traceback__))}'

    end_time = time.time()
    return {
        'score': env.score,
        'steps': env.env._steps,
        'turns': env.last_turn,
        'duration': end_time - start_time,
        'level_num': len(env.levels),
        'end_reason': env.end_reason,
        'seed': seed,
    }


def main():
    if len(sys.argv) <= 1:
        run_simulations()
    elif sys.argv[1] == 'profile':
        run_profiling()
    else:
        seed = int(sys.argv[1])
        skip_to = int(sys.argv[2])
        run_single_interactive_game(seed, skip_to)


def run_single_interactive_game(seed, skip_to):
    termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        env = EnvWrapper(gym.make('NetHackChallenge-v0'), skip_to=skip_to)
        env.env.seed(seed, seed)

        agent = Agent(env, verbose=True)
        agent.main()

    finally:
        os.system('stty sane')


def run_profiling():
    import cProfile, pstats
    games = int(sys.argv[2])
    pr = cProfile.Profile()
    start_time = time.time()
    pr.enable()
    res = []
    for i in range(games):
        res.append(single_simulation(i))
    stats = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(30)
    stats = pstats.Stats(pr).sort_stats(pstats.SortKey.TIME)
    stats.print_stats(20)
    pr.disable()
    duration = time.time() - start_time
    print('turns_per_second:', sum([r['turns'] for r in res]) / duration)
    print('steps_per_second:', sum([r['steps'] for r in res]) / duration)
    print('games_per_hour  :', len(res) / duration * 3600)


def run_simulations():
    from multiprocessing import Pool, Process, Queue
    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.set()
    result_queue = Queue()

    def single_simulation_add_result_to_queue(seed):
        r = single_simulation(seed)
        result_queue.put(r)

    start_time = time.time()
    plot_queue = Queue()

    def plot_thread_func():
        fig = plt.figure()
        plt.show(block=False)
        while 1:
            res = None
            try:
                while 1:
                    res = plot_queue.get(block=False)
            except:
                plt.pause(0.1)
                if res is None:
                    continue

            fig.clear()
            spec = fig.add_gridspec(len(res), 2)
            for i, k in enumerate(sorted(res)):
                ax = fig.add_subplot(spec[i, 0])
                ax.set_title(k)
                if isinstance(res[k][0], str):
                    counter = Counter(res[k])
                    # sns.barplot(x=[k for k, v in counter.most_common()], y=[v for k, v in counter.most_common()])
                else:
                    sns.histplot(res[k], kde=np.var(res[k]) > 1e-6, bins=len(res[k]) // 5 + 1, ax=ax)

            ax = fig.add_subplot(spec[:len(res) // 2, 1])
            sns.scatterplot(x='turns', y='steps', data=res, ax=ax)

            ax = fig.add_subplot(spec[len(res) // 2:, 1])
            sns.scatterplot(x='turns', y='score', data=res, ax=ax)

            plt.show(block=False)

    plt_process = Process(target=plot_thread_func)
    plt_process.start()
    all_res = {}
    last_seed = np.random.randint(0, 2 ** 30)
    simulation_processes = []
    for _ in range(16):
        simulation_processes.append(Process(target=single_simulation_add_result_to_queue, args=(last_seed,)))
        simulation_processes[-1].start()
        last_seed += 1
    count = 0
    while True:
        simulation_processes = [p for p in simulation_processes if p.is_alive() or (p.close() and False)]
        single_res = result_queue.get()

        simulation_processes.append(Process(target=single_simulation_add_result_to_queue, args=(last_seed,)))
        simulation_processes[-1].start()
        last_seed += 1

        if not all_res:
            all_res = {key: [] for key in single_res}
        assert all_res.keys() == single_res.keys()

        count += 1
        for k, v in single_res.items():
            all_res[k].append(v if not hasattr(v, 'item') else v.item())

        plot_queue.put(all_res)

        total_duration = time.time() - start_time

        # print(list(zip(all_res['seed'], all_res['score'], all_res['turns'], all_res['steps'], all_res['end_reason'])))
        print(f'count                         : {count}')
        print(f'time_per_simulation           : {np.mean(all_res["duration"])}')
        print(f'simulations_per_hour          : {3600 / np.mean(all_res["duration"])}')
        print(f'time_per_turn                 : {np.sum(all_res["duration"]) / np.sum(all_res["turns"])}')
        print(f'turns_per_second              : {np.sum(all_res["turns"]) / np.sum(all_res["duration"])}')
        print(f'turns_per_second(multithread) : {np.sum(all_res["turns"]) / total_duration}')
        print(f'score_mean                    : {np.mean(all_res["score"])}')
        print(f'score_median                  : {np.median(all_res["score"])}')
        print(f'score_05-95                   : {np.quantile(all_res["score"], 0.05)} '
              f'{np.quantile(all_res["score"], 0.95)}')
        print(f'score_25-75                   : {np.quantile(all_res["score"], 0.25)} '
              f'{np.quantile(all_res["score"], 0.75)}')
        print(f'exceptions                    : {sum([r.startswith("exception:") for r in all_res["end_reason"]])}')
        print(f'steplimit                     : {sum([r.startswith("steplimit") for r in all_res["end_reason"]])}')
        print(f'timeout                       : {sum([r.startswith("timeout") for r in all_res["end_reason"]])}')
        print()

        with Path('/tmp/nh_sim.json').open('w') as f:
            json.dump(all_res, f)


if __name__ == '__main__':
    main()
