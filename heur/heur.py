import contextlib
import json
import multiprocessing.pool
import os
import subprocess
import sys
import termios
import time
import traceback
import tty
from collections import Counter
from pathlib import Path
from pprint import pprint

import gym
import nle.nethack as nh
import numpy as np

import visualize
from agent import Agent, BLStats, G
from glyph import ALL


class EnvWrapper:
    def __init__(self, env, skip_to=0, visualizer=False, step_limit=None):
        self.env = env
        self.skip_to = skip_to
        self.step_limit = step_limit
        self.visualizer = None
        if visualizer:
            self.visualizer = visualize.Visualizer(self)
        self.last_observation = None
        self.agent = None

        self.draw_walkable = False
        self.draw_seen = False

    def set_agent(self, agent):
        self.agent = agent

    def reset(self):
        obs = self.env.reset()
        self.score = 0
        self.step_count = 0
        self.end_reason = ''
        self.last_observation = obs
        self.render()

        G.assert_map(obs['glyphs'], obs['chars'])

        blstats = BLStats(*obs['blstats'])
        assert obs['chars'][blstats.y, blstats.x] == ord('@')


        return obs

    def render(self):
        if self.visualizer is not None:
            if self.agent is not None:
                print('Message:', self.agent.message)
                print('Pop-up :', self.agent.popup)
            print()
            print(BLStats(*self.last_observation['blstats']))
            if self.agent is not None:
                print('Character:', self.agent.character)
            print('Misc :', self.last_observation['misc'])
            print('Score:', self.score)
            print('Steps:', self.env._steps)
            print('Turns:', self.env._turns)
            print('Seed :', self.env.get_seeds())
            print()
            obj_classes = {getattr(nh, x): x[:-len('_CLASS')] for x in dir(nh) if x.endswith('_CLASS')}
            for letter, text, cls in zip(self.last_observation['inv_letters'],
                                         self.last_observation['inv_strs'],
                                         self.last_observation['inv_oclasses']):
                if (text != 0).any():
                    print(obj_classes[cls].ljust(7), chr(letter), '->', bytes(text).decode())
            print('-' * 20)
            self.env.render()
            print('-' * 20)
            print()
            with self.debug_tiles(self.agent.current_level().walkable, color=(0, 255, 0, 128)) \
                    if self.draw_walkable else contextlib.suppress():
                with self.debug_tiles(~self.agent.current_level().seen, color=(255, 0, 0, 128)) \
                        if self.draw_seen else contextlib.suppress():
                    with self.debug_tiles((self.last_observation['specials'] & nh.MG_OBJPILE) > 0, color=(0, 255, 255, 128)):
                        self.visualizer.step(self.last_observation)
                        self.visualizer.render()

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

            if key == b'\x1bOP':  # F1
                self.draw_walkable = not self.draw_walkable
                self.render()
                continue
            elif key == b'\x1bOQ':  # F2
                self.draw_seen = not self.draw_seen
                self.render()
                continue


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
        if self.visualizer is not None:
            self.render()

            print()
            print('agent_action:', agent_action, repr(chr(int(agent_action))))
            print()

            if self.step_count < self.skip_to:
                self.visualizer.frame_skipping = visualize.FAST_FRAME_SKIPPING
                action = None
            else:
                old_frame_skipping = self.visualizer.frame_skipping
                self.visualizer.frame_skipping = 1
                if old_frame_skipping > 1:
                    self.render()
                action = self.get_action()

            if action is None:
                action = agent_action

            print('\n' * 10)
            print('action:', action)
            print()
        else:
            action = agent_action

        obs, reward, done, info = self.env.step(self.env._actions.index(action))
        self.score += reward
        self.step_count += 1
        # if not done:
        #     G.assert_map(obs['glyphs'], obs['chars'])

        if done:
            end_reason = bytes(obs['tty_chars'].reshape(-1)).decode().replace('You made the top ten list!', '').split()
            if end_reason[7].startswith('Agent'):
                self.score = int(end_reason[6])
                end_reason = ' '.join(end_reason[8:-2])
            else:
                assert self.score == 0, end_reason
                end_reason = ' '.join(end_reason[7:-2])
            first_sentence = end_reason.split('.')[0].split()
            self.end_reason = info['end_status'].name + ': ' + \
                              (' '.join(first_sentence[:first_sentence.index('in')]) + '. ' + \
                               '.'.join(end_reason.split('.')[1:]).strip()).strip()
        if self.step_limit is not None and self.step_count == self.step_limit + 1:
            self.end_reason = self.end_reason or 'steplimit'
            done = True
        elif self.step_limit is not None and self.step_count > self.step_limit + 1:
            assert 0

        self.last_observation = obs

        if done:
            if self.visualizer is not None:
                self.render()
                print('Summary:')
                pprint(self.get_summary())

        return obs, reward, done, info

    def debug_tiles(self, *args, **kwargs):
        if self.visualizer is not None:
            return self.visualizer.debug_tiles(*args, **kwargs)
        return contextlib.suppress()

    def debug_log(self, txt, color=(255, 255, 255)):
        if self.visualizer is not None:
            return self.visualizer.debug_log(txt, color)
        return contextlib.suppress()

    def get_summary(self):
        return {
            'score': self.score,
            'steps': self.env._steps,
            'turns': self.agent.blstats.time,
            'level_num': len(self.agent.levels),
            'character': str(self.agent.character),
            'end_reason': self.end_reason,
            'seed': self.env.get_seeds(),
        }


def single_simulation(seed, timeout=360, step_limit=20000):
    start_time = time.time()
    env = EnvWrapper(gym.make('NetHackChallenge-v0'), step_limit=step_limit)
    env.env.seed(seed, seed)
    agent = Agent(env, verbose=False)
    env.set_agent(agent)

    if timeout is not None:
        pool = multiprocessing.pool.ThreadPool(processes=1)
    try:
        if timeout is not None:
            pool.apply_async(agent.main).get(60)
        else:
            agent.main()
    except multiprocessing.context.TimeoutError:
        env.end_reason = 'timeout'
    except BaseException as e:
        env.end_reason = f'exception: {"".join(traceback.format_exception(None, e, e.__traceback__))}'
        print(f'Seed {seed}, step {env.step_count}:', env.end_reason)

    end_time = time.time()
    summary = env.get_summary()
    summary['duration'] = end_time - start_time
    return summary


def run_single_interactive_game(seed, skip_to):
    termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        env = EnvWrapper(gym.make('NetHackChallenge-v0'), skip_to=skip_to, visualizer=True)
        env.env.seed(seed, seed)

        agent = Agent(env, verbose=True)
        env.set_agent(agent)

        agent.main()

    finally:
        os.system('stty sane')


def run_profiling(games):
    import cProfile, pstats
    pr = cProfile.Profile()

    start_time = time.time()
    pr.enable()
    res = []
    for i in range(games):
        res.append(single_simulation(i, timeout=None))
    pr.disable()

    duration = time.time() - start_time

    stats = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(30)
    stats = pstats.Stats(pr).sort_stats(pstats.SortKey.TIME)
    stats.print_stats(20)
    stats.dump_stats('/tmp/nethack_stats.profile')

    print('turns_per_second:', sum([r['turns'] for r in res]) / duration)
    print('steps_per_second:', sum([r['steps'] for r in res]) / duration)
    print('games_per_hour  :', len(res) / duration * 3600)

    subprocess.run('gprof2dot -f pstats /tmp/nethack_stats.profile -o /tmp/calling_graph.dot'.split())
    subprocess.run('xdot /tmp/calling_graph.dot'.split())


def run_simulations(games, first_seed):
    from multiprocessing import Process, Queue
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
                plt.pause(0.5)
                if res is None:
                    continue

            fig.clear()

            histogram_keys = ['score', 'steps', 'turns', 'level_num']
            spec = fig.add_gridspec(len(histogram_keys) + 2, 2)

            for i, k in enumerate(histogram_keys):
                ax = fig.add_subplot(spec[i, 0])
                ax.set_title(k)
                if isinstance(res[k][0], str):
                    counter = Counter(res[k])
                    sns.barplot(x=[k for k, v in counter.most_common()], y=[v for k, v in counter.most_common()])
                else:
                    if k == 'level_num':
                        bins = [b + 0.5 for b in range(max(res[k]) + 1)]
                    else:
                        bins = np.quantile(res[k], np.linspace(0, 1, min(len(res[k]) // (20 + len(res[k]) // 50) + 2, 50)))
                    sns.histplot(res[k], bins=bins, kde=np.var(res[k]) > 1e-6, stat='density', ax=ax)

            ax = fig.add_subplot(spec[:len(histogram_keys) // 2, 1])
            sns.scatterplot(x='turns', y='steps', data=res, ax=ax)

            ax = fig.add_subplot(spec[len(histogram_keys) // 2: -2, 1])
            sns.scatterplot(x='turns', y='score', data=res, ax=ax)

            ax = fig.add_subplot(spec[-2:, :])
            res['role'] = [h.split('-')[0] for h in res['character']]
            res['race'] = [h.split('-')[1] for h in res['character']]
            res['gender'] = [h.split('-')[2] for h in res['character']]
            res['alignment'] = [h.split('-')[3] for h in res['character']]
            res['race-alignment'] = [f'{r}-{a}' for r, a in zip(res['race'], res['alignment'])]
            sns.violinplot(x='role', y='score', color='white', hue='gender',
                           hue_order=sorted(set(res['gender'])), split=len(set(res['gender'])) == 2,
                           order=sorted(set(res['role'])), inner='quartile', bw=10 / len(res['role']) ** 0.7,
                           data=res, ax=ax)

            palette = ['#ff7043', '#cc3311', '#ee3377', '#0077bb', '#33bbee', '#009988', '#bbbbbb']
            sns.swarmplot(x='role', y='score', hue='race-alignment', hue_order=sorted(set(res['race-alignment'])),
                          order=sorted(set(res['role'])),
                          data=res, ax=ax, palette=palette)

            fig.tight_layout()
            plt.show(block=False)

    plt_process = Process(target=plot_thread_func)
    plt_process.start()
    all_res = {}
    last_seed = first_seed
    simulation_processes = []
    remaining_games = games
    for _ in range(min(16, remaining_games)):
        simulation_processes.append(Process(target=single_simulation_add_result_to_queue, args=(last_seed,)))
        simulation_processes[-1].start()
        last_seed += 1
        remaining_games -= 1

    count = 0
    while count < games:
        simulation_processes = [p for p in simulation_processes if p.is_alive() or (p.close() and False)]
        single_res = result_queue.get()

        if remaining_games:
            simulation_processes.append(Process(target=single_simulation_add_result_to_queue, args=(last_seed,)))
            simulation_processes[-1].start()
            last_seed += 1
            remaining_games -= 1

        if not all_res:
            all_res = {key: [] for key in single_res}
        assert all_res.keys() == single_res.keys()

        count += 1
        for k, v in single_res.items():
            all_res[k].append(v if not hasattr(v, 'item') else v.item())

        if not result_queue.empty():
            continue

        plot_queue.put(all_res)

        total_duration = time.time() - start_time

        text = []
        text.append(f'count                         : {count}')
        text.append(f'time_per_simulation           : {np.mean(all_res["duration"])}')
        text.append(f'simulations_per_hour          : {3600 / np.mean(all_res["duration"])}')
        text.append(f'time_per_turn                 : {np.sum(all_res["duration"]) / np.sum(all_res["turns"])}')
        text.append(f'turns_per_second              : {np.sum(all_res["turns"]) / np.sum(all_res["duration"])}')
        text.append(f'turns_per_second(multithread) : {np.sum(all_res["turns"]) / total_duration}')
        text.append(f'score_median                  : {np.median(all_res["score"]):.1f} +/- '
                    f'{np.std([np.median(np.random.choice(all_res["score"], size=max(1, len(all_res["score"]) // 2))) for _ in range(1024)]):.1f}')
        text.append(f'score_mean                    : {np.mean(all_res["score"]):.1f} +/- '
                    f'{np.std(all_res["score"]) / (len(all_res["score"]) ** 0.5):.1f}')
        text.append(f'score_05-95                   : {np.quantile(all_res["score"], 0.05)} '
                    f'{np.quantile(all_res["score"], 0.95)}')
        text.append(f'score_25-75                   : {np.quantile(all_res["score"], 0.25)} '
                    f'{np.quantile(all_res["score"], 0.75)}')
        text.append(f'exceptions                    : {sum([r.startswith("exception:") for r in all_res["end_reason"]])}')
        text.append(f'steplimit                     : {sum([r.startswith("steplimit") or r.startswith("ABORT") for r in all_res["end_reason"]])}')
        text.append(f'timeout                       : {sum([r.startswith("timeout") for r in all_res["end_reason"]])}')
        print('\n'.join(text) + '\n')

        with Path('/tmp/nh_sim.json').open('w') as f:
            json.dump(all_res, f)

    for p in simulation_processes:
        p.join()
        p.close()

    print('DONE!')


def main():
    if sys.argv[1] == 'simulate':
        games = int(sys.argv[2])
        seed = int(sys.argv[3])
        run_simulations(games, seed)
    elif sys.argv[1] == 'profile':
        games = int(sys.argv[2])
        run_profiling(games)
    elif sys.argv[1] == 'single':
        seed = int(sys.argv[2])
        skip_to = int(sys.argv[3])
        run_single_interactive_game(seed, skip_to)


if __name__ == '__main__':
    main()
