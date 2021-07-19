import atexit
import contextlib
import gc
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import termios
import time
import traceback
import tty
import warnings
from argparse import ArgumentParser
from collections import Counter
from multiprocessing import Process, Queue
from multiprocessing.pool import ThreadPool
from pathlib import Path
from pprint import pprint

import gym
import nle.nethack as nh
import numpy as np

import agent as agent_lib
import visualize


def fork_with_nethack_env(env):
    tmpdir = tempfile.mkdtemp(prefix='nlecopy_')
    shutil.copytree(env.env._vardir, tmpdir, dirs_exist_ok=True)
    env.env._tempdir = None  # it has to be done before the fork to avoid removing the same directory two times
    gc.collect()

    pid = os.fork()

    env.env._tempdir = tempfile.TemporaryDirectory(prefix='nlefork_')
    shutil.copytree(tmpdir, env.env._tempdir.name, dirs_exist_ok=True)
    env.env._vardir = env.env._tempdir.name
    os.chdir(env.env._vardir)
    return pid


def reload_agent(base_path=str(Path(__file__).parent.absolute())):
    global visualize, agent_lib
    visualize = agent_lib = None
    modules_to_remove = []
    for k, m in sys.modules.items():
        if hasattr(m, '__file__') and m.__file__ and m.__file__.startswith(base_path):
            modules_to_remove.append(k)
    del m

    gc.collect()
    while modules_to_remove:
        for k in modules_to_remove:
            assert sys.getrefcount(sys.modules[k]) >= 2
            if sys.getrefcount(sys.modules[k]) == 2:
                sys.modules.pop(k)
                modules_to_remove.remove(k)
                gc.collect()
                break
        else:
            assert 0, ('cannot unload agent library',
                       {k: sys.getrefcount(sys.modules[k]) for k in modules_to_remove})

    import agent as agent_lib
    import visualize


class ReloadAgent(KeyboardInterrupt):
    # it inherits from KeyboardInterrupt as the agent doesn't catch that exception
    pass


class EnvWrapper:
    def __init__(self, env, to_skip=0, visualizer_args=dict(), step_limit=None, agent_args={}, interactive=False):
        self.env = env
        self.agent_args = agent_args
        self.interactive = interactive
        self.to_skip = to_skip
        self.step_limit = step_limit
        self.visualizer = None
        if visualizer_args['enable']:
            visualizer_args.pop('enable')
            self.visualizer = visualize.Visualizer(self, **visualizer_args)
        self.last_observation = None
        self.agent = None

        self.draw_walkable = False
        self.draw_seen = False

    def main(self):
        self.reset()
        while 1:
            try:
                self.agent = agent_lib.Agent(self, **self.agent_args)
                self.agent.main()
                break
            except ReloadAgent:
                pass
            finally:
                self.render()

            self.agent = None
            reload_agent()

    def set_agent(self, agent):
        self.agent = agent

    def reset(self):
        obs = self.env.reset()
        self.score = 0
        self.step_count = 0
        self.end_reason = ''
        self.last_observation = obs

        if self.agent is not None:
            self.render()

        agent_lib.G.assert_map(obs['glyphs'], obs['chars'])

        blstats = agent_lib.BLStats(*obs['blstats'])
        assert obs['chars'][blstats.y, blstats.x] == ord('@')

        return obs

    def fork(self):
        fork_again = True
        while fork_again:
            if (pid := fork_with_nethack_env(self.env)) != 0:
                # parent
                print('freezing parent')
                while 1:
                    try:
                        os.waitpid(pid, 0)
                        break
                    except KeyboardInterrupt:
                        pass
                self.visualizer.force_next_frame()
                self.visualizer.render()
                while 1:
                    try:
                        fork_again = input('fork again [yn]: ')
                        if fork_again == 'y':
                            fork_again = True
                            break
                        elif fork_again == 'n':
                            fork_again = False
                            break
                    except KeyboardInterrupt:
                        pass

                termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            else:
                # child
                atexit.unregister(multiprocessing.util._exit_function)
                self.visualizer.force_next_frame()
                self.visualizer.render()
                break

    def render(self, force=False):
        if self.visualizer is not None:
            with self.debug_tiles(self.agent.current_level().walkable, color=(0, 255, 0, 128)) \
                    if self.draw_walkable else contextlib.suppress():
                with self.debug_tiles(~self.agent.current_level().seen, color=(255, 0, 0, 128)) \
                        if self.draw_seen else contextlib.suppress():
                    with self.debug_tiles((self.last_observation['specials'] & nh.MG_OBJPILE) > 0,
                                          color=(0, 255, 255, 128)):
                        self.visualizer.step(self.last_observation)
                        if force:
                            self.visualizer.force_next_frame()
                        rendered = self.visualizer.render()

            if not force and (not self.interactive or not rendered):
                return

            if self.agent is not None:
                print('Message:', self.agent.message)
                print('Pop-up :', self.agent.popup)
            print()
            if self.agent is not None and hasattr(self.agent, 'blstats'):
                print(agent_lib.BLStats(*self.last_observation['blstats']))
                print(f'Carrying: {self.agent.inventory.items.total_weight} / {self.agent.character.carrying_capacity}')
                print('Character:', self.agent.character)
            print('Misc :', self.last_observation['misc'])
            print('Score:', self.score)
            print('Steps:', self.env._steps)
            print('Turns:', self.env._turns)
            print('Seed :', self.env.get_seeds())
            print('Items below me :', self.agent.inventory.items_below_me)
            print()
            print(self.agent.inventory.items)
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
            if i in agent_lib.G.INV_DICT:
                desc2 += ','.join(agent_lib.G.INV_DICT[i])

            if i in scene_glyphs:
                pos = (self.env.last_observation[0].reshape(-1) == i).nonzero()[0]
                count = len(pos)
                pos = pos[0]
                char = bytes([self.env.last_observation[1].reshape(-1)[pos]])
                texts.append((-count, f'{" " if i in agent_lib.G.INV_DICT else "U"} Glyph {i:4d} -> '
                                      f'Char: {char} Count: {count:4d} '
                                      f'Type: {cls.replace("_OFF", ""):11s} {desc:30s} '
                                      f'{agent_lib.ALL.find(i) if agent_lib.ALL.find(i) is not None else "":20} '
                                      f'{desc2}'))
        for _, t in sorted(texts):
            print(t)

    def get_action(self):
        while 1:
            key = os.read(sys.stdin.fileno(), 5)

            if key == b'\x1bOP':  # F1
                self.draw_walkable = not self.draw_walkable
                self.render()
                continue
            elif key == b'\x1bOQ':  # F2
                self.draw_seen = not self.draw_seen
                self.render()
                continue

            if key == b'\x1bOS':  # F4
                raise ReloadAgent()

            if key == b'\x1b[15~':  # F5
                self.fork()
                continue

            elif key == b'\x1b[3~':  # Delete
                self.to_skip = 16
                return None

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

            if self.interactive:
                print()
                print('agent_action:', agent_action, repr(chr(int(agent_action))))
                print()

            if self.to_skip > 0:
                self.to_skip -= 1
                if self.interactive and self.to_skip == 0:
                    self.visualizer.force_next_frame()
                action = None
            else:
                self.visualizer.force_next_frame()
                self.render()
                action = self.get_action()

            if action is None:
                action = agent_action

            if self.interactive:
                print('action:', action)
                print()
        else:
            action = agent_action

        obs, reward, done, info = self.env.step(self.env._actions.index(action))
        self.score += reward
        self.step_count += 1
        # if not done:
        #     agent_lib.G.assert_map(obs['glyphs'], obs['chars'])

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
            if self.interactive:
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
            'panic_num': len(self.agent.all_panics),
            'character': str(self.agent.character).split()[0],
            'end_reason': self.end_reason,
            'seed': self.env.get_seeds(),
        }


def prepare_env(args, seed, step_limit=None):
    seed += args.seed

    if args.role is not None:
        while 1:
            env = gym.make('NetHackChallenge-v0')
            env.seed(seed, seed)
            obs = env.reset()
            blstats = agent_lib.BLStats(*obs['blstats'])
            character_glyph = obs['glyphs'][blstats.y, blstats.x]
            if nh.permonst(nh.glyph_to_mon(character_glyph)).mname.startswith(args.role):
                break
            seed += 10 ** 9
            env.close()

    if args.visualize_ends is not None:
        assert args.mode == 'simulate'
        args.skip_to = 2 ** 32
    visualizer_args = dict(enable=args.mode == 'run' or args.visualize_ends is not None,
                           start_visualize=args.visualize_ends[seed] if args.visualize_ends is not None else None,
                           show=args.mode == 'run',
                           output_dir=Path('/workspace/vis/') / str(seed),
                           frame_skipping=None if args.visualize_ends is None else 1)
    env = EnvWrapper(gym.make('NetHackChallenge-v0', no_progress_timeout=200),
                     to_skip=args.skip_to, visualizer_args=visualizer_args,
                     agent_args=dict(panic_on_errors=args.panic_on_errors,
                                     verbose=args.mode == 'run'),
                     interactive=args.mode == 'run')
    env.env.seed(seed, seed)
    return env


def single_simulation(args, seed_offset, timeout=360):
    start_time = time.time()
    env = prepare_env(args, seed_offset)

    if timeout is not None:
        pool = ThreadPool(processes=1)
    try:
        if timeout is not None:
            pool.apply_async(env.main).get(timeout)
        else:
            env.main()
    except multiprocessing.context.TimeoutError:
        env.end_reason = f'timeout'
    except BaseException as e:
        env.end_reason = f'exception: {"".join(traceback.format_exception(None, e, e.__traceback__))}'
        print(f'Seed {env.env.get_seeds()}, step {env.step_count}:', env.end_reason)

    end_time = time.time()
    summary = env.get_summary()
    summary['duration'] = end_time - start_time

    if args.visualize_ends is not None:
        env.visualizer.save_end_history()

    env.env.close()
    return summary


def run_single_interactive_game(args):
    termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        summary = single_simulation(args, 0, timeout=None)
        pprint(summary)
    finally:
        os.system('stty sane')


def run_profiling(args):
    if args.profiler == 'cProfile':
        import cProfile, pstats
    elif args.profiler == 'pyinstrument':
        from pyinstrument import Profiler
    else:
        assert 0

    if args.profiler == 'cProfile':
        pr = cProfile.Profile()
    elif args.profiler == 'pyinstrument':
        profiler = Profiler()
    else:
        assert 0

    if args.profiler == 'cProfile':
        pr.enable()
    elif args.profiler == 'pyinstrument':
        profiler.start()
    else:
        assert 0

    start_time = time.time()
    res = []
    for i in range(args.episodes):
        print(f'starting {i + 1} game...')
        res.append(single_simulation(args, i, timeout=None))
    duration = time.time() - start_time


    if args.profiler == 'cProfile':
        pr.disable()
    elif args.profiler == 'pyinstrument':
        session = profiler.stop()
    else:
        assert 0

    print()
    print('turns_per_second :', sum([r['turns'] for r in res]) / duration)
    print('steps_per_second :', sum([r['steps'] for r in res]) / duration)
    print('episodes_per_hour:', len(res) / duration * 3600)
    print()

    if args.profiler == 'cProfile':
        stats = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
        stats.print_stats(30)
        stats = pstats.Stats(pr).sort_stats(pstats.SortKey.TIME)
        stats.print_stats(30)
        stats.dump_stats('/tmp/nethack_stats.profile')

        subprocess.run('gprof2dot -f pstats /tmp/nethack_stats.profile -o /tmp/calling_graph.dot'.split())
        subprocess.run('xdot /tmp/calling_graph.dot'.split())
    elif args.profiler == 'pyinstrument':
        frame_records = session.frame_records

        new_records = []
        for record in frame_records:
            ret_frames = []
            for frame in record[0][1:][::-1]:
                func, module, line = frame.split('\0')
                if func in ['f', 'run']:
                    continue
                ret_frames.append(frame)
                if module.endswith('agent.py') and func in ['step', 'preempt', 'call_update_functions']:
                    break
            ret_frames.append(record[0][0])
            new_records.append((ret_frames[::-1], record[1] / session.duration * 100))
        session.frame_records = new_records
        session.start_call_stack = [session.start_call_stack[0]]

        print('Cumulative time:')
        profiler.last_session = session
        print(profiler.output_text(unicode=True, color=True))

        new_records = []
        for record in frame_records:
            new_records.append([record[0][1:][-1:], record[1] / session.duration * 100])
            new_records[-1][0] = record[0][:1] + new_records[-1][0]
        session.frame_records = new_records
        session.start_call_stack = [session.start_call_stack[0]]

        print('Total time:')
        profiler.last_session = session
        print(profiler.output_text(unicode=True, color=True, show_all=True))
    else:
        assert 0


def run_simulations(args):
    import ray
    ray.init(address='auto')

    start_time = time.time()
    plot_queue = Queue()

    def plot_thread_func():
        from matplotlib import pyplot as plt
        import seaborn as sns

        warnings.filterwarnings('ignore')
        sns.set()

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
                        bins = np.quantile(res[k],
                                           np.linspace(0, 1, min(len(res[k]) // (20 + len(res[k]) // 50) + 2, 50)))
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

    if not args.no_plot:
        plt_process = Process(target=plot_thread_func)
        plt_process.start()

    all_res = {}
    refs = []

    @ray.remote
    def remote_simulation(args, seed_offset):
        # I think there is some nondeterminism in nle environment when playing
        # multiple episodes (maybe bones?). That should do the trick
        q = Queue()

        def sim():
            q.put(single_simulation(args, seed_offset))

        p = Process(target=sim)
        p.start()
        p.join()
        return q.get()

    for seed_offset in range(args.episodes):
        if args.visualize_ends is None or seed_offset in args.visualize_ends:
            refs.append(remote_simulation.remote(args, seed_offset))

    count = 0
    for handle in refs:
        ref, refs = ray.wait(refs, num_returns=1, timeout=None)
        single_res = ray.get(ref[0])

        if not all_res:
            all_res = {key: [] for key in single_res}
        assert all_res.keys() == single_res.keys()

        count += 1
        for k, v in single_res.items():
            all_res[k].append(v if not hasattr(v, 'item') else v.item())

        plot_queue.put(all_res)

        total_duration = time.time() - start_time

        median_score_std = np.std([np.median(np.random.choice(all_res["score"],
                                                              size=max(1, len(all_res["score"]) // 2)))
                                   for _ in range(1024)])

        text = []
        text.append(f'count                         : {count}')
        text.append(f'time_per_simulation           : {np.mean(all_res["duration"])}')
        text.append(f'simulations_per_hour          : {3600 / np.mean(all_res["duration"])}')
        text.append(f'simulations_per_hour(multi)   : {3600 * count / total_duration}')
        text.append(f'time_per_turn                 : {np.sum(all_res["duration"]) / np.sum(all_res["turns"])}')
        text.append(f'turns_per_second              : {np.sum(all_res["turns"]) / np.sum(all_res["duration"])}')
        text.append(f'turns_per_second(multi)       : {np.sum(all_res["turns"]) / total_duration}')
        text.append(f'panic_num_per_game(median)    : {np.median(all_res["panic_num"])}')
        text.append(f'panic_num_per_game(mean)      : {np.sum(all_res["panic_num"]) / count}')
        text.append(f'score_median                  : {np.median(all_res["score"]):.1f} +/- '
                    f'{median_score_std:.1f}')
        text.append(f'score_mean                    : {np.mean(all_res["score"]):.1f} +/- '
                    f'{np.std(all_res["score"]) / (len(all_res["score"]) ** 0.5):.1f}')
        text.append(f'score_05-95                   : {np.quantile(all_res["score"], 0.05)} '
                    f'{np.quantile(all_res["score"], 0.95)}')
        text.append(f'score_25-75                   : {np.quantile(all_res["score"], 0.25)} '
                    f'{np.quantile(all_res["score"], 0.75)}')
        text.append(f'exceptions                    : '
                    f'{sum([r.startswith("exception:") for r in all_res["end_reason"]])}')
        text.append(f'steplimit                     : '
                    f'{sum([r.startswith("steplimit") or r.startswith("ABORT") for r in all_res["end_reason"]])}')
        text.append(f'timeout                       : '
                    f'{sum([r.startswith("timeout") for r in all_res["end_reason"]])}')
        print('\n'.join(text) + '\n')

        if args.visualize_ends is None:
            with Path('/tmp/nh_sim.json').open('w') as f:
                json.dump(all_res, f)

    print('DONE!')
    ray.shutdown()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('mode', choices=('simulate', 'run', 'profile'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--skip-to', type=int, default=0)
    parser.add_argument('-n', '--episodes', type=int, default=1)
    parser.add_argument('--role', choices=('arc', 'bar', 'cav', 'hea', 'kni',
                                           'mon', 'pri', 'ran', 'rog', 'sam',
                                           'tou', 'val', 'wiz'))
    parser.add_argument('--panic-on-errors', action='store_true')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--visualize-ends', type=Path, default=None,
                        help='Path to json file with dict: seed -> visualization_start_step')
    parser.add_argument('--profiler', choices=('cProfile', 'pyinstrument'), default='pyinstrument')

    args = parser.parse_args()
    if args.seed is None:
        args.seed = np.random.randint(0, 2 ** 30)

    if args.visualize_ends is not None:
        with args.visualize_ends.open('r') as f:
            args.visualize_ends = {int(k): int(v) for k, v in json.load(f).items()}

    print('ARGS:', args)
    return args


def main():
    args = parse_args()
    if args.mode == 'simulate':
        run_simulations(args)
    elif args.mode == 'profile':
        run_profiling(args)
    elif args.mode == 'run':
        run_single_interactive_game(args)


if __name__ == '__main__':
    main()
