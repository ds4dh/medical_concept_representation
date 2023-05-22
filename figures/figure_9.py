import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import rankdata

do_bis_figure = True
models = ['word2vec', 'fasttext', 'glove']
model_titles = {'word2vec': 'word2vec', 'fasttext': 'fastText', 'glove': 'GloVe'}
all_perfs = {'auroc': {}, 'auprc': {}}
categories = ['DIA_', 'PRO_', 'MED_']
conds_auroc = ['AUROC-%s' % cond for cond in ['1L', '2L', '3L', '4L', 'EM']]
conds_auprc = ['AUPRC-%s' % cond for cond in ['1L', '2L', '3L', '4L', 'EM']]
result_dir = os.path.join(os.getcwd(), 'logs_04_03', 'full_whole05_shuffle')

x_data = ['1L', '2L', '3L', '4L', 'EM']
desc = {'DIA_': 'ICD10-CM', 'PRO_': 'ICD10-PCS', 'MED_': 'ATC'}
y_lims = {
    'auroc': {'DIA_': [0.0, 1.0], 'PRO_': [0.0, 1.0], 'MED_': [0.0, 1.0]},
    'auprc': {'DIA_': [0.0, 1.0], 'PRO_': [0.0, 1.0], 'MED_': [0.0, 1.0]},
    # 'auprc': {'DIA_': [0.0, 0.25], 'PRO_': [0.0, 0.65], 'MED_': [0.0, 0.25]},
}
marker_specs = {'marker': 'o', 'markersize': 3}
perf_colors = {'word2vec': 'tab:green', 'fasttext': 'tab:blue', 'glove': 'tab:red'}
rand_colors = {'auroc': 'k', 'auprc': 'k'}
medal_colors = {1: 'goldenrod', 2: 'dimgrey', 3: 'firebrick'}

perfs_rand = {
    'auroc': {
      'DIA_': [[0.5] * 5],
      'PRO_': [[0.5] * 5],
      'MED_': [[0.5] * 5],  
    },
    'auprc': {  # simulated, but corresponds to n_positives / n_total
        'DIA_': [[0.048,0.006,0.001,0.000,0.000]],
        'PRO_': [[0.272,0.022,0.004,0.001,0.000]],
        'MED_': [[0.098,0.071,0.015,0.007,0.001]],
    },
}

for model in models:
    model_dir = next((os.path.join(result_dir, item) for item in os.listdir(result_dir) if model in item))
    traj_file = next((os.path.join(model_dir, item) for item in os.listdir(model_dir) if 'trajectorization' in item))
    pred_df = pd.read_csv(traj_file)
    perfs_auroc = {cat: pred_df[pred_df['Category'] == cat][conds_auroc].values.tolist() for cat in categories}
    perfs_auprc = {cat: pred_df[pred_df['Category'] == cat][conds_auprc].values.tolist() for cat in categories}
    all_perfs['auroc'].update({model: perfs_auroc})
    all_perfs['auprc'].update({model: perfs_auprc})

if not do_bis_figure:
    for measure in ['auroc', 'auprc']:
        fig, axs = plt.subplots(3, 3, figsize=(10, 6))
        for i, (ax, cat) in enumerate(zip(axs, desc.keys())):
            p_rand = np.array(perfs_rand[measure][cat]).mean(axis=0)

            mean_perf = {}
            for model, perfs in all_perfs[measure].items():
                mean_perf[model] = np.array(perfs[cat]).mean(axis=0)
            medals = []
            for i, x in enumerate(x_data):
                mean_perfs = [mean_perf[m][i] for m in models]
                ranks = len(mean_perfs) - rankdata(mean_perfs, method='ordinal') + 1
                medals.append(ranks)
            medals = {m: r for m, r in zip(models, zip(*medals))}

            for j, (model, perfs) in enumerate(all_perfs[measure].items()):
                for k, perf in enumerate(perfs[cat]):
                    ax[j].plot(x_data, perf, **marker_specs,
                            c=perf_colors[model], alpha=1.0, label='model')
                ax[j].plot(x_data, p_rand, '--', **marker_specs,
                        c=rand_colors[measure], label='random')
                ax[j].set_ylim(y_lims[measure][cat])
                if measure == 'auprc':
                    ax[j].yaxis.set_major_locator(ticker.MultipleLocator(0.1))
                ax[j].legend()
                ax[j].grid()
                if j == 0: ax[j].set_ylabel('%s - %s' % (measure.upper(), desc[cat]), fontsize='large')
                if i == 0: ax[j].set_title(model_titles[model], fontsize='large')

                for i, x in enumerate(x_data):
                    ax[j].get_xticklabels()[i].set_color(medal_colors[medals[model][i]])
                    ax[j].get_xticklabels()[i].set_fontweight('bold')
            
        plt.tight_layout()
        plt.savefig('figures/figure_9_%s.png' % measure, dpi=300)

else:
    for measure in ['auroc', 'auprc']:
        fig, axs = plt.subplots(1, 3, figsize=(9, 2.5))
        for i, (ax, cat) in enumerate(zip(axs, desc.keys())):
            p_rand = np.array(perfs_rand[measure][cat]).mean(axis=0)
            for j, (model, perfs) in enumerate(all_perfs[measure].items()):
                for k, perf in enumerate(perfs[cat]):
                    ax.plot(x_data, perf, **marker_specs, c=perf_colors[model], label=model_titles[model])
                    if j == len(all_perfs[measure].items()) - 1:
                        ax.plot(x_data, p_rand, '--', **marker_specs, c=rand_colors[measure], label='random')
                    ax.set_ylim(y_lims[measure][cat])
                    ax.legend(ncol=(2 if measure == 'auroc' else 1),
                              loc=('lower center' if measure == 'auroc' else 'best'),
                              fontsize=('small' if measure == 'auroc' else 'medium'))
                    ax.grid()
                    if i == 0: ax.set_ylabel(measure.upper(), fontsize='large')
                    ax.set_title(desc[cat], fontsize='large')

        plt.tight_layout()
        plt.savefig('figures/figure_9_bis_%s.png' % measure, dpi=300)
