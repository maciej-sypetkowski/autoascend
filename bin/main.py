import json
import multiprocessing
import os
import subprocess
import sys
import termios
import time
import traceback
import tty
import warnings
from argparse import ArgumentParser
from multiprocessing import Process, Queue
from multiprocessing.pool import ThreadPool
from pathlib import Path
from pprint import pprint

import gym
import nle.nethack as nh
import numpy as np

from autoascend import agent as agent_lib
from autoascend.env_wrapper import EnvWrapper
from autoascend.utils import plot_dashboard


def prepare_env(args, seed):
    seed += args.seed

    if args.role:
        while 1:
            env = gym.make('NetHackChallenge-v0')
            env.seed(seed, seed)
            obs = env.reset()
            blstats = agent_lib.BLStats(*obs['blstats'])
            character_glyph = obs['glyphs'][blstats.y, blstats.x]
            if any([nh.permonst(nh.glyph_to_mon(character_glyph)).mname.startswith(role) for role in args.role]):
                break
            seed += 10 ** 9
            env.close()

    if args.visualize_ends is not None:
        assert args.mode == 'simulate'
        args.skip_to = 2 ** 32

    visualize_with_simulate = args.visualize_ends is not None or args.output_video_dir is not None
    visualizer_args = dict(enable=args.mode == 'run' or visualize_with_simulate,
                           start_visualize=args.visualize_ends[seed] if args.visualize_ends is not None else None,
                           show=args.mode == 'run',
                           output_dir=Path('/tmp/vis/') / str(seed),
                           frame_skipping=None if not visualize_with_simulate else 1,
                           output_video_path=(args.output_video_dir / f'{seed}.mp4'
                                              if args.output_video_dir is not None else None))
    env = EnvWrapper(gym.make('NetHackChallenge-v0', no_progress_timeout=1000),
                     to_skip=args.skip_to, visualizer_args=visualizer_args,
                     agent_args=dict(panic_on_errors=args.panic_on_errors,
                                     verbose=args.mode == 'run'),
                     interactive=args.mode == 'run')
    env.env.seed(seed, seed)
    return env


def single_simulation(args, seed_offset, timeout=720):
    start_time = time.time()
    env = prepare_env(args, seed_offset)

    try:
        if timeout is not None:
            with ThreadPool(1) as pool:
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

    if env.visualizer is not None and env.visualizer.video_writer is not None:
        env.visualizer.video_writer.close()
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
    elif args.profiler == 'none':
        pass
    else:
        assert 0

    if args.profiler == 'cProfile':
        pr = cProfile.Profile()
    elif args.profiler == 'pyinstrument':
        profiler = Profiler()
    elif args.profiler == 'none':
        pass
    else:
        assert 0

    if args.profiler == 'cProfile':
        pr.enable()
    elif args.profiler == 'pyinstrument':
        profiler.start()
    elif args.profiler == 'none':
        pass
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
    elif args.profiler == 'none':
        pass
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
                if func in ['f', 'f2', 'run', 'wrapper']:
                    continue
                ret_frames.append(frame)
                if module.endswith('agent.py') and func in ['step', 'preempt', 'call_update_functions']:
                    break
            ret_frames.append(record[0][0])
            new_records.append((ret_frames[::-1], record[1] / session.duration * 100))
        session.frame_records = new_records
        session.start_call_stack = [session.start_call_stack[0]]

        print('Cumulative time:')
        profiler._last_session = session
        print(profiler.output_text(unicode=True, color=True))

        new_records = []
        for record in frame_records:
            ret_frames = []
            for frame in record[0][1:][::-1]:
                func, module, line = frame.split('\0')
                ret_frames.append(frame)
                if str(Path(module).absolute()).startswith(str(Path(__file__).parent.absolute())):
                    break
            ret_frames.append(record[0][0])
            new_records.append((ret_frames[::-1], record[1] / session.duration * 100))
        session.frame_records = new_records
        session.start_call_stack = [session.start_call_stack[0]]

        print('Total time:')
        profiler._last_session = session
        print(profiler.output_text(unicode=True, color=True, show_all=True))
    elif args.profiler == 'none':
        pass
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
            plot_dashboard(fig, res)
            fig.tight_layout()
            plt.show(block=False)

    if not args.no_plot:
        plt_process = Process(target=plot_thread_func)
        plt_process.start()

    refs = []

    @ray.remote(num_gpus=1 / 4 if args.with_gpu else 0)
    def remote_simulation(args, seed_offset, timeout=500):
        # I think there is some nondeterminism in nle environment when playing
        # multiple episodes (maybe bones?). That should do the trick
        q = Queue()

        if args.output_video_dir is not None:
            timeout = 4 * 24 * 60 * 60

        def sim():
            q.put(single_simulation(args, seed_offset, timeout=timeout))

        try:
            p = Process(target=sim, daemon=False)
            p.start()
            return q.get()
        finally:
            p.terminate()
            p.join()

        # uncomment to debug why join doesn't work properly
        # from multiprocessing.pool import ThreadPool
        # with ThreadPool(1) as thrpool:
        #     def fun():
        #         import time
        #         while True:
        #             time.sleep(1)
        #             print(p.pid, p.is_alive(), p.exitcode, p)
        #     thrpool.apply_async(fun)
        # p.join(timeout=timeout + 1)
        # assert not q.empty()

    try:
        with args.simulation_results.open('r') as f:
            all_res = json.load(f)
        print('Continue running: ', (len(all_res['seed'])))
    except FileNotFoundError:
        all_res = {}

    done_seeds = set()
    if 'seed' in all_res:
        done_seeds = set(s[0] for s in all_res['seed'])

    # remove runs finished with exceptions if rerunning with --panic-on-errors
    if args.panic_on_errors and all_res:
        idx_to_repeat = set()
        for i, (seed, reason) in enumerate(zip(all_res['seed'], all_res['end_reason'])):
            if reason.startswith('exception'):
                idx_to_repeat.add(i)
                done_seeds.remove(seed[0])
        print('Repeating idx:', idx_to_repeat)
        for k, v in all_res.items():
            all_res[k] = [v for i, v in enumerate(v) if i not in idx_to_repeat]

    print('skipping seeds', done_seeds)
    for seed_offset in range(args.episodes):
        seed = args.seed + seed_offset
        if seed in done_seeds:
            continue
        if args.seeds and seed not in args.seeds:
            continue
        if args.visualize_ends is None or seed_offset in [k % 10 ** 9 for k in args.visualize_ends]:
            refs.append(remote_simulation.remote(args, seed_offset))

    count = len(done_seeds)
    initial_count = count
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
        text.append(f'simulations_per_hour(multi)   : {3600 * (count - initial_count) / total_duration}')
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
            with args.simulation_results.open('w') as f:
                json.dump(all_res, f)

    print('DONE!')
    ray.shutdown()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('mode', choices=('simulate', 'run', 'profile'))
    parser.add_argument('--seed', type=int, help='Starting random seed')
    parser.add_argument('--seeds', nargs="*", type=int,
                        help='Run only these specific seeds (only relevant in simulate mode)')
    parser.add_argument('--skip-to', type=int, default=0)
    parser.add_argument('-n', '--episodes', type=int, default=1)
    parser.add_argument('--role', choices=('arc', 'bar', 'cav', 'hea', 'kni',
                                           'mon', 'pri', 'ran', 'rog', 'sam',
                                           'tou', 'val', 'wiz'),
                        action='append')
    parser.add_argument('--panic-on-errors', action='store_true')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--visualize-ends', type=Path, default=None,
                        help='Path to json file with dict: seed -> visualization_start_step.'
                        'THIS IS AN UNMAINTAINED FEATURE.'
                        'It was used to save some visualizer frames before agent deathto conveniently browse them.')
    parser.add_argument('--output-video-dir', type=Path, default=None,
                        help="Episode visualization video directory -- valid only with 'simulate' mode")
    parser.add_argument('--profiler', choices=('cProfile', 'pyinstrument', 'none'), default='pyinstrument')
    parser.add_argument('--with-gpu', action='store_true')
    parser.add_argument('--simulation-results', default='nh_sim.json', type=Path,
                        help='path to simulation results json. Only for simulation mode')

    args = parser.parse_args()
    if args.seed is None:
        args.seed = np.random.randint(0, 2 ** 30)

    if args.visualize_ends is not None:
        with args.visualize_ends.open('r') as f:
            args.visualize_ends = {int(k): int(v) for k, v in json.load(f).items()}

    if args.output_video_dir is not None:
        assert args.mode == 'simulate', "Video output only valid in 'simulate' mode"

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
    else:
        assert 0


if __name__ == '__main__':
    main()
