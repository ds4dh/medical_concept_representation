import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


MODELS = ["word2vec", "fasttext"]  #, "glove"]
MODEL_TITLES = {"word2vec": "word2vec", "fasttext": "fastText", "glove": "GloVe"}
RESULT_DIR = os.path.join(os.getcwd(), "logs", "full_whole05_shuffle")
CATEGORIES = ["DIA_", "PRO_", "MED_"]
CONDS_AUROC = ["AUROC-%s" % cond for cond in ["1L", "2L", "3L", "4L", "EM"]]
CONDS_AUPRC = ["AUPRC-%s" % cond for cond in ["1L", "2L", "3L", "4L", "EM"]]
MEASURE_SUFFIXES = ["", "-STD", "-STE"]
PARTIALS = ["P = 0.0", "P = 0.1", "P = 0.3", "P = 0.5", "P = 0.8"]
ALPHAS = [0.2, 0.4, 0.6, 0.8, 1.0]
X_DATA = ["1L", "2L", "3L", "4L", "EM"]
DESCRS = {"DIA_": "ICD10-CM", "PRO_": "ICD10-PCS", "MED_": "ATC"}
Y_LIMS = {
    "auroc": {"DIA_": [0.0, 1.0], "PRO_": [0.0, 1.0], "MED_": [0.0, 1.0]},
    "auprc": {"DIA_": [0.0, 1.0], "PRO_": [0.0, 1.0], "MED_": [0.0, 1.0]},
}
MARKER_SPECS = {"marker": ".", "markersize": 2}
PERF_COLORS = {
    "word2vec": "tab:green",
    "fasttext": "tab:blue",
    "glove": "tab:red",
}
RAND_COLORS = {"auroc": "k", "auprc": "k"}
PERF_RAND = {
    "auroc": {
      "DIA_": [[0.5] * 5] * 5,
      "PRO_": [[0.5] * 5] * 5,
      "MED_": [[0.5] * 5] * 5,
    },
    "auprc": {  # simulated, but corresponds to n_positives / n_total
        "DIA_": [[0.265, 0.053, 0.011, 0.003, 0.001]] * 5,
        "PRO_": [[0.331, 0.044, 0.010, 0.002, 0.001]] * 5,
        "MED_": [[0.539, 0.507, 0.209, 0.123, 0.018]] * 5,
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
    perf_dict = {"auroc": {}, "auprc": {}}
    for model in MODELS:
        
        # Load prediction performance dictionary from model directory
        model_dir = next((
            os.path.join(RESULT_DIR, item) for item in os.listdir(RESULT_DIR)
            if model in item
        ))
        model_file = next((
            os.path.join(model_dir, item) for item in os.listdir(model_dir)
            if "prediction" in item and "timed" in item
        ))
        model_df = pd.read_csv(model_file)
        
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


def generate_figure_8(perf_dict: dict) -> None:
    """ Plot model AUROC and AUPRC performance in a png file
    """
    # One figure per measurement
    for measure in ["auroc", "auprc"]:
        _, axs = plt.subplots(3, 3, figsize=(9, 6))
        
        # Go through each category and get random performance
        for i, (ax_row, cat) in enumerate(zip(axs, DESCRS.keys())):
            p_rand = np.array(PERF_RAND[measure][cat]).mean(axis=0)
            
            # Go through each model and loop over conditions
            for j, (model, perfs) in enumerate(perf_dict[measure].items()):
                for means, stds, stes, prt, alpha in zip(
                    perfs[cat][""],
                    perfs[cat]["-STD"],
                    perfs[cat]["-STE"],
                    PARTIALS,
                    ALPHAS,
                ):
                    
                    # Plot model prediction performance
                    ax_row[j].plot(
                        X_DATA, means, **MARKER_SPECS,
                        c=PERF_COLORS[model], alpha=alpha, label=prt,
                    )
                
                # Plot random performance
                ax_row[j].plot(
                    X_DATA, p_rand, "--", **MARKER_SPECS,
                    c=RAND_COLORS[measure], label="random",
                )
                
                # Polish figure
                ax_row[j].set_ylim(Y_LIMS[measure][cat])
                ax_row[j].legend(
                    ncol=(2 if measure == "auroc" else 1),
                    loc=("lower center" if measure == "auroc" else "best"),
                )
                ax_row[j].grid()
                if j == 0:
                    ax_row[j].set_ylabel(
                        "%s - %s" % (measure.upper(), DESCRS[cat]),
                        fontsize="large"
                )
                if i == 0:
                    ax_row[j].set_title(MODEL_TITLES[model], fontsize="large")
                
        # Save figure to a png file
        plt.tight_layout()
        plt.savefig("figures/figure_8_bis_timed_%s.png" % measure, dpi=300)

if __name__ == "__main__":
    main()
    