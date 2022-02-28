import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HEADER = '-' * 50


def extract_last_part_from_exception(text):
    return text[text.rfind('  File "'):]


def extract_last_place_from_exception(text):
    return text[text.rfind('  File "'):].splitlines()[0]


def load_df(filepath):
    with Path(filepath).open() as f:
        df = json.load(f)
    df = pd.DataFrame.from_dict(df)
    for k in df.keys():
        df[k] = [tuple(v) if isinstance(v, list) else v for v in df[k]]
    return df


def give_examples(df, ref_df):
    return f'{len(df)}x ({len(df) / len(ref_df) * 100:.1f}%) ({sorted([(t.seed, t.steps, t.score) for t in df.itertuples()][:5], key=lambda x: x[1])})'


def print_exceptions(df, ref_df):
    print(HEADER, 'EXCEPTIONS:')
    counter = Counter([extract_last_place_from_exception(r) for r in df.end_reason if r.startswith('exception:')])
    for k, v in counter.most_common():
        d = df[[r.startswith('exception:') and k == extract_last_place_from_exception(r) for r in df.end_reason]]
        print(k, '\n', extract_last_part_from_exception(d.end_reason.iloc[0]), ':', give_examples(d, df))
        print()
    print()
    print()


def get_group_from_end_reason(text):
    if text.startswith('exception:'):
        return 'exception'
    if 'starved' in text or 'while fainted from lack of food' in text:
        return 'food'
    if 'the shopkeeper' in text:
        return 'peaceful_mon'
    if 'was poisoned' in text:
        if 'corpse' in text or 'glob' in text:
            return 'poisoned_food'
        else:
            return 'poisoned_other'
    if 'turned to stone' in text:
        return 'stone'
    if 'frozen by a monster' in text:
        return 'frozen'
    if 'while sleeping' in text:
        return 'sleeping'

    return 'other'


def print_end_reasons(df, ref_df):
    print(HEADER, 'END REASONS:')
    for end_reason_group, d in sorted(df.groupby('end_reason_group'), key=lambda x: -len(x[1])):
        print(' ', 'GROUP:', end_reason_group, ':', give_examples(d, ref_df))
        counter = Counter([r for r in d.end_reason if not r.startswith('exception:')])
        for k, v in counter.most_common():
            d2 = d[[not r.startswith('exception:') and k == r for r in d.end_reason]]
            print('   ', k, ':', give_examples(d2, d))
        print()
    print()
    print()


def print_summary(comment, df, ref_df, indent=0):
    indent_chars = '  ' * indent

    print(indent_chars + HEADER, f'SUMMARY ({comment}):')

    print(indent_chars + ' ', '*' * 8, 'stats')
    for stat_name, stat_values in [
        ('score        ', df.score),
        *[(f'score-{role}    ', df[df.role == role].score) for role in sorted(df.role.unique())],
        *[(f'score-mile-{milestone} ', df[df.milestone == milestone].score) for milestone in
          sorted(df.milestone.unique())],
        ('exp_level    ', df.experience_level),
        ('dung_level   ', df.level_num),
        ('runtime_dur  ', df.duration),
    ]:
        mean = np.mean(stat_values)
        std = np.std(stat_values)
        quantiles = np.quantile(stat_values, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
        quantiles = ' '.join((f'{q:6.0f}' for q in quantiles))
        with np.printoptions(precision=3, suppress=True):
            print(indent_chars + '     ', stat_name, ':',
                  f'{mean:6.0f} +/- {std:6.0f}  [{quantiles}] ({len(stat_values)}x)')

    print(indent_chars + ' ', '*' * 8, 'end_reasons', give_examples(df, ref_df))
    for end_reason_group, d in sorted(df.groupby('end_reason_group'), key=lambda x: -len(x[1])):
        print(indent_chars + '   ', end_reason_group, ':', give_examples(d, df))

    print()


def main(filepath):
    df = load_df(filepath)
    df.seed = [s[0] for s in df.seed]
    df['end_reason_group'] = [get_group_from_end_reason(e) for e in df.end_reason]
    df['role'] = [c[:3] for c in df.character]
    median = np.median(df.score)

    print_exceptions(df, df)
    print_end_reasons(df, df)

    print(HEADER, 'SORTED BY SCORE:')
    print(df.sort_values('score'))
    print()

    print_summary('all', df, df)
    print_summary('score >= median', df[df.score >= median], df)
    print_summary('score < median', df[df.score < median], df)

    print(HEADER, 'BY ROLE:')
    for k, d in df.groupby('role'):
        print_summary(k, d, df, indent=1)

    print(HEADER, 'BY MILESTONE:')
    for k, d in df.groupby('milestone'):
        print_summary(f'milestone-{k}', d, df, indent=1)

    print(HEADER, 'TO PASTE:')
    std_median = np.std([np.median(np.random.choice(df.score, size=max(1, len(df) // 2))) for _ in range(1024)])
    print(f'median    : {np.median(df.score):.1f} +/- {std_median:.1f}')
    print(f'mean      : {np.mean(df.score):.1f} +/- {np.std(df.score) / (len(df) ** 0.5):.1f}')
    for role in sorted(df.role.unique()):
        s = df[df.role == role].score
        print(f'{role}:{s.median():.0f},{s.mean():.0f}', end='')
        if role != 'wiz':
            print('|', end='')
        if role == 'mon':
            print()
    print()
    print(f'exceptions: {sum((r.startswith("exception:") for r in df.end_reason)) / len(df) * 100:.1f}%')
    print(f'avg_turns : {np.mean(df.turns):.1f} +/- {np.std(df.turns) / (len(df) ** 0.5):.1f}')
    print(f'avg_steps : {np.mean(df.steps):.1f} +/- {np.std(df.steps) / (len(df) ** 0.5):.1f}')


if __name__ == '__main__':
    pd.set_option('display.min_rows', 30)
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)

    filepath = '/workspace/nh_sim.json' if len(sys.argv) <= 1 else sys.argv[1]
    main(filepath)
