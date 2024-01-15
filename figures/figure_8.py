import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


MODELS = ['word2vec', 'fasttext', 'glove']
MODEL_TITLES = {'word2vec': 'word2vec', 'fasttext': 'fastText', 'glove': 'GloVe'}
RESULT_DIR = os.path.join(os.getcwd(), 'logs_true_latest', 'full_whole05_shuffle')
CATEGORIES = ['DIA_', 'PRO_', 'MED_']
CONDS_AUROC = ['AUROC-%s' % cond for cond in ['1L', '2L', '3L', '4L', 'EM']]
CONDS_AUPRC = ['AUPRC-%s' % cond for cond in ['1L', '2L', '3L', '4L', 'EM']]
MEASURE_SUFFIXES = ["", "-STD", "-STE"]
PARTIALS = ['P = 0.0', 'P = 0.1', 'P = 0.3', 'P = 0.6', 'P = 1.0']
X_DATA = ['1L', '2L', '3L', '4L', 'EM']
DESCRS = {'DIA_': 'ICD10-CM', 'PRO_': 'ICD10-PCS', 'MED_': 'ATC'}
Y_LIMS = {
    'auroc': {'DIA_': [0.0, 1.0], 'PRO_': [0.0, 1.0], 'MED_': [0.0, 1.0]},
    'auprc': {'DIA_': [0.0, 1.0], 'PRO_': [0.0, 1.0], 'MED_': [0.0, 1.0]},
    # 'auprc': {'DIA_': [0.0, 0.5], 'PRO_': [0.0, 0.7], 'MED_': [0.0, 0.85]},
}
MARKER_SPECS = {'marker': '.', 'markersize': 2}
PERF_COLORS = {'word2vec': 'tab:green', 'fasttext': 'tab:blue', 'glove': 'tab:red'}
RAND_COLORS = {'auroc': 'k', 'auprc': 'k'}
PERF_RAND = {
    'auroc': {
      'DIA_': [[0.5] * 5] * 5,
      'PRO_': [[0.5] * 5] * 5,
      'MED_': [[0.5] * 5] * 5,
    },
    'auprc': {  # simulated, but corresponds to n_positives / n_total
        'DIA_': [[0.265, 0.053, 0.011, 0.003, 0.001]] * 5,
        'PRO_': [[0.331, 0.044, 0.010, 0.002, 0.001]] * 5,
        'MED_': [[0.539, 0.507, 0.209, 0.123, 0.018]] * 5,
    },
}


def main():
    """ Load diagnosis / procedure / medication tokens prediction performance
        of all models and plots them in figure 8
    """
    perf_dict = load_data()
    generate_figure_8(perf_dict)


def load_data() -> dict:
    """ Load AUROC and AUPRC performance of all models as a dictionary
    """
    perf_dict = {'auroc': {}, 'auprc': {}}
    for model in MODELS:
        
        # Load prediction performance dictionary from model directory
        model_dir = next((
            os.path.join(RESULT_DIR, item) for item in os.listdir(RESULT_DIR)
            if model in item
        ))
        model_file = next((
            os.path.join(model_dir, item) for item in os.listdir(model_dir)
            if 'prediction' in item
        ))
        model_df = pd.read_csv(model_file)
        
        # Fill dictionary with AUROC and AUPRC performance
        perfs_auroc = load_perf(model_df, CONDS_AUROC)
        perfs_auprc = load_perf(model_df, CONDS_AUPRC)
        perf_dict['auroc'].update({model: perfs_auroc})
        perf_dict['auprc'].update({model: perfs_auprc})
        
    return perf_dict


def load_perf(model_df: pd.DataFrame, conds: list[str]) -> dict:
    """ Load model prediction performance from dataframe as a dictionary
    """
    perf_dict = {}
    for cat in CATEGORIES:
        cat_perf_df = model_df[model_df["Category"] == cat]
        cat_perf_dict = {}
        
        for suffix in MEASURE_SUFFIXES:
            labels = [cond + suffix for cond in conds]
            cat_perf_dict[suffix] = cat_perf_df[labels].values.tolist()
        
        perf_dict[cat] = cat_perf_dict
        
    return perf_dict


def generate_figure_8(perf_dict: dict) -> None:
    """ Plot model AUROC and AUPRC performance in a png file
    """
    # One figure per measurement
    for measure in ['auroc', 'auprc']:
        _, axs = plt.subplots(5, 3, figsize=(9, 10.5))
        
        # Go through each category and get random performance
        for i, cat in enumerate(DESCRS.keys()):
            p_rand = np.array(PERF_RAND[measure][cat]).mean(axis=0)
            
            # Go through each model and loop over conditions
            for j, (model, perfs) in enumerate(perf_dict[measure].items()):
                for k, (means, stds, stes, prt) in enumerate(
                    zip(
                        perfs[cat][""],
                        perfs[cat]["-STD"],
                        perfs[cat]["-STE"],
                        PARTIALS,
                )):
                    
                    # Plot model performance
                    axs[k, i].plot(
                        X_DATA, means, **MARKER_SPECS,
                        c=PERF_COLORS[model], label=MODEL_TITLES[model],
                    )
                    
                    # Plot model performance with error bars
                    errs = [s / 2 for s in stds]  # one on each side
                    axs[k, i].errorbar(
                        X_DATA, means, yerr=errs, color=PERF_COLORS[model],
                        ecolor=PERF_COLORS[model], elinewidth=1, capsize=3,
                    )
                    
                    # Plot random performance
                    if j == len(perf_dict[measure].items()) - 1:
                        axs[k, i].plot(
                            X_DATA, p_rand, '--', **MARKER_SPECS,
                            c=RAND_COLORS[measure], label='random',
                        )
                    
                    # Polish figure
                    axs[k, i].set_ylim(Y_LIMS[measure][cat])
                    axs[k, i].legend(
                        ncol=(2 if measure == 'auroc' else 1),
                        loc=('lower center' if measure == 'auroc' else 'best'),
                        fontsize=('small' if measure == 'auroc' else 'medium'),
                        columnspacing=0.5,
                    )
                    axs[k, i].grid()
                    axs[k, i].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
                    if i == 0:
                        axs[k, i].set_ylabel(
                            '%s - %s' % (measure.upper(), prt),
                            fontsize='large',
                        )
                    if k == 0:
                        axs[k, i].set_title(DESCRS[cat], fontsize='large')

        plt.tight_layout()
        plt.savefig('figures/figure_8_%s.png' % measure, dpi=300)


if __name__ == "__main__":
    main()
    