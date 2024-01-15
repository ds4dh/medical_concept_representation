import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


MODELS = ["word2vec", "fasttext", "glove"]
MODEL_TITLES = {"word2vec": "word2vec", "fasttext": "fastText", "glove": "GloVe"}
CATEGORIES = ["DIA_", "PRO_", "MED_"]
CONDS_AUROC = ["AUROC-%s" % cond for cond in ["1L", "2L", "3L", "4L", "EM"]]
CONDS_AUPRC = ["AUPRC-%s" % cond for cond in ["1L", "2L", "3L", "4L", "EM"]]
MEASURE_SUFFIXES = ["", "-STD", "-STE"]
# RESULT_DIR = os.path.join(os.getcwd(), "logs_true_latest", "full_whole05_shuffle")
RESULT_DIR = os.path.join(os.getcwd(), "logs", "full_whole05_shuffle")
X_DATA = ["1L", "2L", "3L", "4L", "EM"]
DESCRS = {"DIA_": "ICD10-CM", "PRO_": "ICD10-PCS", "MED_": "ATC"}
Y_LIMS = {
    "auroc": {"DIA_": [0.0, 1.0], "PRO_": [0.0, 1.0], "MED_": [0.0, 1.0]},
    "auprc": {"DIA_": [0.0, 1.0], "PRO_": [0.0, 1.0], "MED_": [0.0, 1.0]},
    # "auprc": {"DIA_": [0.0, 0.25], "PRO_": [0.0, 0.65], "MED_": [0.0, 0.25]},
}
MARKER_SPECS = {"marker": "o", "markersize": 3}
PERF_COLORS = {"word2vec": "tab:green", "fasttext": "tab:blue", "glove": "tab:red"}
RAND_COLORS = {"auroc": "k", "auprc": "k"}
PERFS_RAND = {
    "auroc": {
      "DIA_": [[0.5] * 5],
      "PRO_": [[0.5] * 5],
      "MED_": [[0.5] * 5],  
    },
    "auprc": {  # simulated, but corresponds to n_positives / n_total
        "DIA_": [[0.048,0.006,0.001,0.000,0.000]],
        "PRO_": [[0.272,0.022,0.004,0.001,0.000]],
        "MED_": [[0.098,0.071,0.015,0.007,0.001]],
    },
}


def main():
    """ Load most important diagnosis prediction and next procedure / medication
        prediction performance of all models and plots them in figure 9
    """
    perf_dict = load_data()
    generate_figure_9(perf_dict)


def load_data():
    """ Load AUROC and AUPRC performance of all models as a dictionary
    """
    perf_dict = {"auroc": {}, "auprc": {}}
    for model in MODELS:
        
        # Load prediction performance dictionary from model directory
        model_dir = next((
            os.path.join(RESULT_DIR, item) for item in os.listdir(RESULT_DIR)
            if model in item
        ))
        traj_file = next((
            os.path.join(model_dir, item) for item in os.listdir(model_dir)
            if "trajectorization" in item and "rc_False" in item
        ))
        model_df = pd.read_csv(traj_file)
        
        # Fill dictionary with AUROC and AUPRC performance
        perfs_auroc = load_perf(model_df, CONDS_AUROC)
        perfs_auprc = load_perf(model_df, CONDS_AUPRC)
        perf_dict["auroc"].update({model: perfs_auroc})
        perf_dict["auprc"].update({model: perfs_auprc})
    
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


def generate_figure_9(perf_dict: dict) -> None:
    """ Plot model AUROC and AUPRC performance in a png file
    """
    # One figure per measurement
    for measure in ["auroc", "auprc"]:
        _, axs = plt.subplots(1, 3, figsize=(9, 2.5))
        
        # Go through each category and get random performance
        for i, (ax, cat) in enumerate(zip(axs, CATEGORIES)):
            p_rand = np.array(PERFS_RAND[measure][cat]).mean(axis=0)
            
            # Go through each model and loop over conditions
            for j, (model, perfs) in enumerate(perf_dict[measure].items()):
                for means, stds, stes in zip(
                        perfs[cat][""], perfs[cat]["-STD"], perfs[cat]["-STE"],
                ):
                    
                    # Plot model prediction performance
                    ax.plot(
                        X_DATA, means, **MARKER_SPECS,
                        c=PERF_COLORS[model], label=MODEL_TITLES[model]
                    )
                    
                    # Plot model performance with error bars
                    errs = [s / 2 for s in stds]  # one on each side
                    ax.errorbar(
                        X_DATA, means, yerr=errs, color=PERF_COLORS[model],
                        ecolor=PERF_COLORS[model], elinewidth=1, capsize=3,
                    )
                    
                    # Plot random prediction performance
                    if j == len(perf_dict[measure].items()) - 1:
                        ax.plot(
                            X_DATA, p_rand, "--", **MARKER_SPECS,
                            c=RAND_COLORS[measure], label="random"
                        )
                    
                    # Polish figure
                    ax.set_ylim(Y_LIMS[measure][cat])
                    ax.legend(
                        ncol=(2 if measure == "auroc" else 1),
                        loc=("lower center" if measure == "auroc" else "best"),
                        fontsize=("small" if measure == "auroc" else "medium"),
                    )
                    ax.grid()
                    if i == 0: ax.set_ylabel(measure.upper(), fontsize="large")
                    ax.set_title(DESCRS[cat], fontsize="large")
        
        # Save figure to a png file
        plt.tight_layout()
        plt.savefig("figures/figure_9_%s.png" % measure, dpi=300)


if __name__ == "__main__":
    main()
    