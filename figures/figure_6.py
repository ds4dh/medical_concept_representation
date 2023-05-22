import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import rankdata

do_bis_figure = False
models = ['word2vec', 'fasttext', 'glove']
model_titles = {'word2vec': 'word2vec', 'fasttext': 'fastText', 'glove': 'GloVe'}
categories = ['DIA_', 'PRO_', 'MED_']
cat_mappings = {'DIA_': 'MOR_', 'PRO_': 'REA_', 'MED_': 'LOS_'}
measure_lbls = ['AUROC-BI', 'AUPRC-BI']
result_dir = os.path.join(os.getcwd(), 'logs_04_03', 'full_whole05_shuffle')

alphas = [0.2, 0.4, 0.6, 0.8, 1.0]
partials = ['P = 0.0', 'P = 0.1', 'P = 0.3', 'P = 0.6', 'P = 1.0']
x_data = {'AUROC': 0, 'AUPRC': 1}
desc = {'LOS_': 'Length-of-stay', 'REA_': 'Readmission', 'MOR_': 'Mortality'}
y_lims = {'LOS_': [0.0, 1.0], 'REA_': [0.0, 1.0], 'MOR_': [0.0, 1.0]}
bar_specs = {'width': 0.12, 'edgecolor': 'black', 'linewidth': 0.7, 'zorder': 10}
marker_specs = {'marker': 'o', 'markersize': 3}
perf_colors = {'word2vec': 'tab:green', 'fasttext': 'tab:blue', 'glove': 'tab:red'}
rand_colors = {'auroc': 'k', 'auprc': 'k'}
medal_colors = {1: 'goldenrod', 2: 'dimgrey', 3: 'firebrick'}
perfs_rand = {
    'LOS_': [[0.500, 0.163]] * 5,
    'REA_': [[0.500, 0.197]] * 5,
    'MOR_': [[0.500, 0.042]] * 5,
}

all_perfs = {}
for model in models:
    model_dir = next((os.path.join(result_dir, item) for item in os.listdir(result_dir) if model in item))
    pred_file = next((os.path.join(model_dir, item) for item in os.listdir(model_dir) if 'prediction' in item))
    pred_df = pd.read_csv(pred_file)
    perfs = {cat_mappings[cat]: pred_df[pred_df['Category'] == cat][measure_lbls].values.tolist() for cat in categories}
    all_perfs.update({model: perfs})

if not do_bis_figure:
    fig, axs = plt.subplots(3, 3, figsize=(10, 8))
    for i, (ax, cat) in enumerate(zip(axs, desc.keys())):
        p_rand = np.array(perfs_rand[cat]).mean(axis=0)

        # mean_perf = {}
        # for model, perfs in all_perfs.items():
        #     mean_perf[model] = np.array(perfs[cat]).mean(axis=0)
        # medals = []
        # for i, x in enumerate(x_data):
        #     mean_perfs = [mean_perf[m][i] for m in models]
        #     ranks = len(mean_perfs) - rankdata(mean_perfs, method='ordinal') + 1
        #     medals.append(ranks)
        # medals = {m: r for m, r in zip(models, zip(*medals))}

        for j, (model, perfs) in enumerate(all_perfs.items()):
            for k, (perf, prt, a) in enumerate(zip(perfs[cat], partials, alphas)):
                facecolor =  mcolors.to_rgb(perf_colors[model]) + (a,)
                ax[j].bar(np.arange(len(x_data)) + k*bar_specs['width'], perf,
                        **bar_specs, facecolor=facecolor, label=prt)
            
            rand_xs = [[x - bar_specs['width'], x + bar_specs['width'] * len(perfs[cat])] for x in x_data.values()]
            ax[j].plot(rand_xs[0], [p_rand[0], p_rand[0]], '--', c='k',label='random', zorder=20)
            ax[j].plot(rand_xs[1], [p_rand[1], p_rand[1]], '--', c='k',label='_no_legend_', zorder=20)
            ax[j].set_xticks([x + (len(perfs[cat]) // 2) * bar_specs['width'] for x in x_data.values()])
            ax[j].set_xticklabels(list(x_data.keys()), fontsize='large')
            ax[j].tick_params(axis='both', which='both',length=0)
            ax[j].set_ylim(y_lims[cat])
            ax[j].legend(labelspacing=0.2)
            ax[j].grid()
            if j == 0: ax[j].set_ylabel('AUPRC - %s' % desc[cat], fontsize='large')
            if i == 0: ax[j].set_title(model_titles[model], fontsize='large')
            
            # for i, x in enumerate(x_data):
            #     ax[j].get_xticklabels()[i].set_color(medal_colors[medals[model][i]])
            #     ax[j].get_xticklabels()[i].set_fontweight('bold')
            
    plt.tight_layout()
    plt.savefig('figures/figure_6.png', dpi=300)

else:
    bar_specs = {'width': 0.18, 'edgecolor': 'black', 'linewidth': 0.7, 'zorder': 10}
    fig, axs = plt.subplots(2, 3, figsize=(11, 6))
    for i, (measure, ax) in enumerate(zip(['AUROC', 'AUPRC'], axs)):
        for j, cat in enumerate(desc.keys()):
            
            for m, (model, perfs) in enumerate(all_perfs.items()):
                for k, (perf, prt, a) in enumerate(zip(perfs[cat], partials, alphas)):
                    facecolor = mcolors.to_rgb(perf_colors[model])  # + (a,)
                    lbl = model_titles[model] if k == 0 else '_no_legend_'
                    ax[j].bar(k + m*bar_specs['width'], perf[i], **bar_specs, facecolor=facecolor, label=lbl)
                    if m == 0:
                        p_rand = np.array(perfs_rand[cat]).mean(axis=0)
                        rand_xs = [k - bar_specs['width'], k + bar_specs['width'] * len(models)]
                        lbl = 'random' if k == 0 else '_no_legend_'
                        ax[j].plot(rand_xs, [p_rand[i], p_rand[i]], '--', c='k',label=lbl, zorder=20)
            
            ax[j].set_xticks([x + (len(models) // 2) * bar_specs['width'] for x in range(len(partials))])
            ax[j].set_xticklabels([p.replace(' ', '') for p in partials], fontsize='medium')
            ax[j].tick_params(axis='both', which='both', color='white')  # length=0)
            ax[j].set_ylim([0.0, 1.0] if measure == 'AUROC' else [0.0, 0.6])
            legend = ax[j].legend(ncols=2, loc=('lower center' if measure == 'AUROC' else 'upper center'),
                                  fontsize='medium', labelspacing=0.3, framealpha=1.0)
            legend.set_zorder(30)
            ax[j].grid()
            if j == 0: ax[j].set_ylabel(measure, fontsize='large')
            if i == 0: ax[j].set_title(desc[cat], fontsize='large')
            
    plt.tight_layout()
    plt.savefig('figures/figure_6_bis.png', dpi=300)
