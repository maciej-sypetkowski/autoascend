import json

import pandas as pd


def interesting_reason(txt):
    return ('food' not in txt
            and 'shop' not in txt
            and 'falling rock' not in txt
            and 'Error' not in txt
            and 'starved' not in txt
            and 'timeout' not in txt
            and 'sleeping' not in txt
            and 'wand' not in txt
            and 'bolt' not in txt
            and 'missile' not in txt
            and 'rotted' not in txt
            and 'invisible' not in txt)


def process(path='/tmp/nh_sim.json'):
    ret = dict()

    with open(path, 'r') as f:
        df = pd.DataFrame(json.load(f))
    df['role'] = [ch[:3] for ch in df.character]

    for row in df.itertuples():
        if interesting_reason(row.end_reason):
            print(row.seed[0], row.end_reason)
            ret[row.seed[0]] = row.steps - 64

    with open('filtered.json', 'w') as f:
        json.dump(ret, f)


process('/tmp/nh_sim.json')
# process('/workspace/nh_sim_fight2.json')

