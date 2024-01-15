import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import data
import pytorch_lightning as pl
import itertools
from metrics.metric_utils import (
    compute_reduced_representation,
    log_figure_to_board,
    bootstrap_rate_reduction_ci,
)


CATEGORY_SUBLEVELS_WHO = {
    "DIA_": [("A", "B",), ("C", "D0", "D1", "D2", "D3", "D4",),
             ("D5", "D6", "D7", "D8", "D9",), ("E",), ("F",), ("G",),
             ("H0", "H1", "H2", "H3", "H4", "H5",), ("H6", "H7", "H8", "H9",),
             ("I",), ("J",), ("K",), ("L",), ("M",), ("N",), ("O",), ("P",),
             ("Q",), ("R",), ("S", "T",), ("V", "W", "X", "Y",), ("Z",), ("U",)],
    "PRO_": [("0",), ("1",), ("2",), ("3",), ("4",), ("5",), ("6",), ("7",),
             ("8",), ("9",), ("B",), ("C",), ("D",), ("F",), ("G",), ("H",), ("X",)],
    "MED_": [("A",), ("B",), ("C",), ("D",), ("G",), ("H",), ("J",), ("L",),
             ("M",), ("N",), ("P",), ("R",), ("S",), ("V",)],
}
CATEGORY_SUBLEVELS_FREQUENT = {
    "DIA_": [("C", "D0", "D1", "D2", "D3", "D4"), ("D5", "D6", "D7", "D8", "D9"),
             ("F",), ("G",), ("H0", "H1", "H2", "H3", "H4", "H5"),
             ("H6", "H7", "H8", "H9"), ("I",), ("J",), ("K",),
             ("L",), ("M",), ("N",), ("O",), ("S",), ("T",), ("Z",)],
    "PRO_": [("00", "01"), ("03", "04"), ("05", "06"), ("0B",),
             ("0D",), ("0F",), ("0H",), ("0J",), ("0P", "0Q", "0R", "0S",),
             ("0T", "0U", "0V"), ("0W",), ("0X", "0Y"), ("3",), ("B",)],
    "MED_": [("A",), ("B",), ("C",), ("D",), ("G",), ("J",), ("L",), ("N",),
             ("R",), ("S",)],
}
CATEGORY_SUBLEVELS = CATEGORY_SUBLEVELS_FREQUENT
FIG_SIZE = (14, 5)
BASE_TEXT_SIZE = 10
SMALL_TEXT_SIZE = 10
N_ANNOTATED_SAMPLES = 100  # per category
ALL_COLORS = list(plt.cm.tab20(np.arange(20)[0::2])) + ["#333333", "#ffffff"] +\
             list(plt.cm.tab20(np.arange(20)[1::2]))
LEGEND_PARAMS = {
    "loc": "upper center",
    "bbox_to_anchor": (0.5, -0.05),
    "fancybox": True,
    "shadow": True,
    "fontsize": SMALL_TEXT_SIZE,
}
SCATTER_PARAMS =  {
    "marker": "o",
    "s": 20,
    "linewidths": 0.5,
    "edgecolors": "k",
}
PLOT_CAT_MAP = {
    "DIA": "ICD10-CM code",
    "PRO": "ICD10-PCS code",
    "MED": "ATC code",
}


def visualization_task(
    model: torch.nn.Module,
    pipeline: data.DataPipeline,
    logger: pl.loggers.Logger,
    global_step: int,
) -> None:
    """ Reduce the dimensionality of concept embeddings for different categories
        and log a scatter plot of the low-dimensional data to tensorboard
    """
    print("\nProceeding with visualization testing metric")
    fig = plt.figure(figsize=FIG_SIZE)
    for subplot_idx, category in enumerate(CATEGORY_SUBLEVELS.keys()):
        
        # Load data from tokenizer vocabulary and model embeddings
        print(" - Reducing dimensionality and visualizing %s tokens" % category)
        token_info = get_token_info(model, pipeline.tokenizer, category)
        
        # Compute rate reduction (dr) for raw and reduced embeddings
        print("\n - Computing delta-R for raw embeddings (with bootstrap)")
        dr_raw, dr_raw_std, dr_raw_ste = \
            bootstrap_rate_reduction_ci(token_info["labels"], token_info["embedded"])
        print(" - Computing delta-R for reduced embeddings (with bootstrap)")
        dr_red, dr_red_std, dr_red_ste = \
            bootstrap_rate_reduction_ci(token_info["labels"], token_info["reduced"])
        
        dr_info = {
            "raw": {"mean": dr_raw, "std": dr_raw_std, "ste": dr_raw_ste},
            "red": {"mean": dr_red, "std": dr_red_std, "ste": dr_red_ste},
        }
        
        # Update figure with data of this category
        plot_reduced_data(
            fig,
            token_info,
            category,
            subplot_idx,
            dr_info,
        )

    # Log final figure to the board
    # plt.savefig("tmp.png", dpi=300)
    log_figure_to_board(fig, "visualization_metric", logger, global_step)


def plot_reduced_data(
    fig: matplotlib.figure.Figure,
    token_info: dict,
    category: str,
    subplot_idx: int,
    dr_info: dict[str, dict[str, float]],
) -> None:
    """ Plot data of reduced dimensionality to a 2d or 3d scatter plot
    """
    # Prepare subfigure and title
    data_dim = token_info["reduced"].shape[-1]
    plot_cat = PLOT_CAT_MAP[category.split("_")[0]]
    dr_str = "%.2f ± %.2f (%.2f ± %.2f)" % (
        dr_info["red"]["mean"],
        dr_info["red"]["std"] / 2,
        dr_info["raw"]["mean"],
        dr_info["raw"]["std"] / 2,
    )
    title_info = (plot_cat, data_dim, dr_str)
    plot_title = "%s embeddings\n%sd proj.\n deltaR = %s" % title_info
    kwargs = {} if data_dim <= 2 else {"projection": "3d"}
    ax = fig.add_subplot(1, 3, subplot_idx + 1, **kwargs)
    ax.set_title(plot_title, fontsize=BASE_TEXT_SIZE)
    
    # Plot data of reduced dimensionality
    unique_labels = sorted(list(set(token_info["labels"])))
    unique_colors = ALL_COLORS[:len(unique_labels)]
    label_array = np.empty(len(token_info["labels"]), dtype=object)
    label_array[:] = token_info["labels"]
    for label, color in zip(unique_labels, unique_colors):
        data = token_info["reduced"][[l == label for l in label_array]]
        data = [data[:, i] for i in range(data.shape[-1])]
        label = label[0] if len(label) == 1 else "-".join([label[0], label[-1]])
        ax.scatter(*data, **SCATTER_PARAMS, color=color, label=label)
    
    # # Add text annotation around some data points
    # if N_ANNOTATED_SAMPLES > 0:
    #     loop = list(zip(token_info["reduced"], token_info["tokens"]))
    #     np.random.shuffle(loop)
    #     texts = []
    #     for d, t in loop:  # [:N_ANNOTATED_SAMPLES]:
    #         if "DIA_T2" in t or "DIA_T30" in t or "DIA_T31" in t or "DIA_T32" in t or "DIA_T33" in t or "DIA_T34" in t:
    #             texts.append(ax.text(d[0], d[1], t, fontsize="xx-small"))
    #     arrowprops = dict(arrowstyle="->", color="k", lw=0.5)
    #     adjust_text(texts, ax=ax, min_arrow_len=5, arrowprops=arrowprops)
    
    # Polish figure
    ax.margins(x=0.0, y=0.0)
    handles, labels = ax.get_legend_handles_labels()
    # labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    def flip(items, ncol):
        return itertools.chain(*[items[i::ncol] for i in range(ncol)])
    ax.legend(flip(handles, 4), flip(labels, 4), **LEGEND_PARAMS, ncol=4)
    # ax.legend(**LEGEND_PARAMS, ncol=5)  # len(unique_labels) // 2)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if data_dim > 2: ax.set_zticklabels([])
    

def get_token_info(
    model: torch.nn.Module,
    tokenizer: data.tokenizers.Tokenizer,
    category: str
) -> dict[str, list[str]]:
    """ For each token, compute its embedding vector and its class labels
    """
    # Get tokens and labels from tokenizer vocabulary
    vocab = tokenizer.get_vocab()
    cat_tokens = [t for t in vocab if category in t and category != t]
    token_label_pairs = [select_plotted_token(t, category) for t in cat_tokens]
    tokens, labels = zip(*[p for p in token_label_pairs if p is not None])
    
    # Compute embeddings and reduced embeddings
    encoded = [tokenizer.encode(token) for token in tokens]
    embeddings = model.get_token_embeddings(encoded).numpy()
    reduced_embeddings = compute_reduced_representation(embeddings)
    
    return {
        "tokens": tokens,
        "embedded": embeddings,
        "reduced": reduced_embeddings,
        "labels": labels,
    }


def select_plotted_token(token: str, cat: str) -> tuple[str, str]:
    """ Return token / match pairs if the token starts with any of the strings
        included in any of the match tuple of a given category
    """
    to_check = token.split(cat)[-1]
    for s in CATEGORY_SUBLEVELS[cat]:
        if to_check.startswith(s):  # s is a tuple of strings
            return (token, s)

