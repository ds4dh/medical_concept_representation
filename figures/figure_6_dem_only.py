import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


MODELS = ["word2vec", "fasttext", "glove"]
MODEL_TITLES = {"word2vec": "word2vec", "fasttext": "fastText", "glove": "GloVe"}
LOG_DIRS = ["logs_dem_only", "logs"]
LOG_SUBDIR = "full_whole05_shuffle"
RESULT_DIRS = [os.path.join(os.getcwd(), d, LOG_SUBDIR) for d in LOG_DIRS]
CATEGORIES = ["DIA_", "PRO_", "MED_"]
CAT_MAPPINGS = {"DIA_": "MOR_", "PRO_": "REA_", "MED_": "LOS_"}
MEASURE_LABELS = ["AUROC-BI", "AUPRC-BI"]
MEASURE_SUFFIXES = ["", "-STD", "-STE"]
ALPHAS = [0.2, 0.8]
PARTIALS = ["DEM ONLY", "P = 0.0"]
X_DATA = {"AUROC": 0, "AUPRC": 1}
DESCRS = {"LOS_": "Length-of-stay", "REA_": "Readmission", "MOR_": "Mortality"}
Y_LIMS = {"LOS_": [0.0, 1.0], "REA_": [0.0, 1.0], "MOR_": [0.0, 1.0]}
BAR_SPECS = {"width": 0.12, "edgecolor": "black", "linewidth": 0.7, "zorder": 10}
MARKER_SPECS = {"marker": "o", "markersize": 3}
PERF_COLORS = {"word2vec": "tab:green", "fasttext": "tab:blue", "glove": "tab:red"}
RAND_COLORS = {"auroc": "k", "auprc": "k"}
RANDOM_PERFS = {
    "LOS_": [[0.500, 0.163]] * len(PARTIALS),
    "REA_": [[0.500, 0.197]] * len(PARTIALS),
    "MOR_": [[0.500, 0.042]] * len(PARTIALS),
}


def main():
    """ Load binary outcome prediction performance of all models and plots them
        in figure 6
    """
    all_perf_dict = {}
    for model in MODELS:
        for result_dir in RESULT_DIRS:
            
            model_perf_df = load_data(result_dir, model, "prediction")
            model_perf_dict = load_perf(model_perf_df)
            if model not in all_perf_dict:
                all_perf_dict.update({model: model_perf_dict})
            
            else:
                for cat in all_perf_dict[model]:
                    for suffix in MEASURE_SUFFIXES:
                        appended = model_perf_dict[cat][suffix][0]
                        all_perf_dict[model][cat][suffix].append(appended)
                    
    generate_figure_6(all_perf_dict)
    

def load_data(dir, model, key):
    """ Load data from the first directory including a given model name in its
        name, opening the first file that includes a given key in its name
    """
    model_dir = next((
        os.path.join(dir, item)
        for item in os.listdir(dir)
        if model in item
    ))
    pred_file = next((
        os.path.join(model_dir, item)
        for item in os.listdir(model_dir)
        if key in item
    ))
    return pd.read_csv(pred_file)


def load_perf(model_df: pd.DataFrame) -> dict:
    """ Load model prediction performance from dataframe as a dictionary
    """
    perf_dict = {}
    for cat in CATEGORIES:
        cat_perf_df = model_df[model_df["Category"] == cat]
        cat_perf_dict = {}
        
        for suffix in MEASURE_SUFFIXES:
            labels = [label + suffix for label in MEASURE_LABELS]
            cat_perf_dict[suffix] = cat_perf_df[labels].values.tolist()
            
        perf_dict[CAT_MAPPINGS[cat]] = cat_perf_dict
        
    return perf_dict


def generate_figure_6(perf_dict: dict) -> None:
    """ Draw figure from data dictionary
    """
    # Go through all categories
    _, axs = plt.subplots(3, 3, figsize=(10, 8))
    for i, (ax, cat) in enumerate(zip(axs, DESCRS.keys())):
        p_rand = np.array(RANDOM_PERFS[cat]).mean(axis=0)
        
        # Go through all models
        for j, (model, perfs) in enumerate(perf_dict.items()):
            perf_means = perfs[cat][MEASURE_SUFFIXES[0]]
            perf_stds = perfs[cat][MEASURE_SUFFIXES[1]]
            perf_stes = perfs[cat][MEASURE_SUFFIXES[2]]
            for k, (means, stds, stes, partial, alpha) in enumerate(
                zip(perf_means, perf_stds, perf_stes, PARTIALS, ALPHAS)
            ):
                
                # Draw coloured bars
                facecolor =  mcolors.to_rgb(PERF_COLORS[model]) + (alpha,)
                ax[j].bar(
                    np.arange(len(X_DATA)) + k * BAR_SPECS["width"],
                    means, **BAR_SPECS, facecolor=facecolor, label=partial,
                )
                
                # Draw error bars on top of the coloured bars
                errs = [s / 2 for s in stds]  # one on each side
                ax[j].errorbar(
                    np.arange(len(X_DATA)) + k * BAR_SPECS["width"], means, errs,
                    fmt='none', ecolor='k', alpha=0.8, capsize=3, zorder=20,
                )
            
            # Draw random performance as a dashed line
            rand_xs = [
                [x - BAR_SPECS["width"], x + BAR_SPECS["width"] * len(PARTIALS)]
                for x in X_DATA.values()
            ]
            ax[j].plot(
                rand_xs[0], [p_rand[0], p_rand[0]], "--",
                c="k", label="random", zorder=20,
            )
            ax[j].plot(
                rand_xs[1], [p_rand[1], p_rand[1]], "--",
                c="k",label="_no_legend_", zorder=20,
            )
            
            # Polish figure
            to_remove = 0.5 * int(len(PARTIALS) % 2 == 0)
            ax[j].set_xticks([
                x + (len(PARTIALS) / 2 - to_remove) * BAR_SPECS["width"]
                for x in X_DATA.values()
            ])
            ax[j].set_xticklabels(list(X_DATA.keys()), fontsize="large")
            ax[j].tick_params(axis="both", which="both",length=0)
            ax[j].set_ylim(Y_LIMS[cat])
            ax[j].legend(labelspacing=0.2)
            ax[j].grid()
            if j == 0: ax[j].set_ylabel("AUPRC - %s" % DESCRS[cat], fontsize="large")
            if i == 0: ax[j].set_title(MODEL_TITLES[model], fontsize="large")
    
    # Save figure to a png file
    plt.tight_layout()
    plt.savefig("figures/figure_6_dem_only.png", dpi=300)


if __name__ == "__main__":
    main()
    