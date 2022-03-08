"""
Script for extracting features stats for featurization for RL experiment.
Uncomment code fragment in autoascend/combat/rl_scoring.py to generate the observations.txt file.
"""

import base64
import json
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


with open('/tmp/vis/observations.txt') as f:
    observations = defaultdict(list)
    for line in f.readlines():
        observation = pickle.loads(base64.b64decode(line))
        for k, v in observation.items():
            observations[k].append(v)


def plot_column(df, column):
    if df[column].dtype == object:
        plt.xticks(rotation=45)
    try:
        sns.histplot(df[column])
    except ValueError:
        print(f'ValueError when plotting {column}')
    except IndexError:
        print(f'IndexError when plotting {column}')


def plot_df(df_name, df, max_plots_in_row=10):
    nrows = (len(df.columns) + max_plots_in_row - 1) // max_plots_in_row
    fig = plt.figure(figsize=(8, 2 * nrows), dpi=80)
    fig.suptitle(df_name, fontsize=100)
    gridspec = fig.add_gridspec(nrows=nrows, ncols=max_plots_in_row)
    for i, c in enumerate(df.columns):
        row = i // max_plots_in_row
        col = i % max_plots_in_row
        ax = fig.add_subplot(gridspec[i:i + 1])
        ax.title.set_text(c)
        plt.sca(ax)
        plot_column(df, c)

    plt.tight_layout()
    plt.show()


stats = defaultdict(dict)


for k, v in observations.items():
    v = np.array(v)
    print('----------------------', k, v.shape)
    if len(v.shape) > 2:
        v = v.transpose((0, 2, 3, 1))
        v = v.reshape(-1, v.shape[-1])
    print(k, v.shape)
    print([np.mean(np.isnan(v[:, i])) for i in range(v.shape[1])])
    # plot_df(k, pd.DataFrame(v).sample(10000), 5)
    mean, std = np.nanmean(v, axis=0), np.nanstd(v, axis=0)
    v_normalized = (v - mean) / std
    minv = np.nanmin(v_normalized, axis=0)
    print(mean)
    print(std)
    print(minv)

    stats[k]['mean'] = mean.tolist()
    stats[k]['std'] = mean.tolist()
    stats[k]['min'] = mean.tolist()

    if k == 'heur_action_priorities':
        for i in range(v_normalized.shape[1]):
            v_normalized[:, i][np.isnan(v_normalized[:, i])] = minv[i]
    else:
        v_normalized[np.isnan(v_normalized)] = 0

    # plot_df(k + ' normalized', pd.DataFrame(v_normalized).sample(10000), 5)
    print()


with open('/workspace/muzero/rl_features_stats.json', 'w') as f:
    json.dump(stats, f, indent=4)
