import os
import data
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from matplotlib.lines import Line2D
from tqdm import tqdm
from metrics.metric_utils import (
    compute_reduced_representation,
    log_figure_to_board,
)


LOAD_REDUCED_DATA = False  # set this to True if you just want to change the plot
FIG_SIZE = (16, 5)
SMALL_TEXT_SIZE = 8
BIG_TEXT_SIZE = 14
OUTCOME_CLASSES = {
    "length-of-stay": ["LBL_SHORT", "LBL_LONG"],
    "readmission": ["LBL_AWAY", "LBL_READM"],
    "mortality": ["LBL_ALIVE", "LBL_DEAD"],
}
ALL_OUTCOME_LEVELS = [t for v in OUTCOME_CLASSES.values() for t in v]


def outcomization_task(
    model: torch.nn.Module,
    pipeline: data.DataPipeline,
    logger: pl.loggers.Logger,
    global_step: int,
) -> None:
    """ Reduce the dimensionality of concept embeddings for different categories
        and log a scatter plot of the low-dimensional data to tensorboard
    """
    # Load results directly if required (e.g., for re-plotting final figure)
    load_or_save_path = os.path.join(logger.save_dir, "outcomization_data.pkl")    
    if LOAD_REDUCED_DATA:
        patient_embeddings, outcome_embeddings, vocab_embeddings, gold_list = \
            load_or_save(load_or_save_path, "load")
    
    # Or compute them from the model and testing dataset
    else:
        print("\nProceeding with outcomization testing metric")
        # Get reduced embeddings for all tokens of the model"s vocabulary
        vocab_embedding_dict, vocab_count_dict = \
            get_vocab_embeddings_and_counts(model, pipeline)
            
        # Initialize testing data pipeline
        to_remove = pipeline.run_params["ngrams_to_remove"]
        dp = data.JsonReader(pipeline.data_fulldir, "test")
        dp = data.TokenFilter(dp, to_remove=to_remove, to_split=ALL_OUTCOME_LEVELS)
        
        # Compute patient embeddings and associated labels
        embedding_list, gold_list, seen_set = list(), list(), set()
        for sample, gold in tqdm(dp, desc="Getting patient embeddings"):
            embedding = get_patient_embedding(
                sample, vocab_count_dict, vocab_embedding_dict,
            )
            embedding_hash = hash(embedding.tobytes())
            if embedding_hash not in seen_set:  # avoiding duplicates
                seen_set.add(embedding_hash)
                embedding_list.append(embedding)
                gold_list.append(gold)
        
        # Get reduced representation of patients, single tokens, and labels
        patient_embeddings, outcome_embeddings, vocab_embeddings = \
            reduce_embeddings(embedding_list, vocab_embedding_dict)
            
        # Save embeddings for potential later use (e.g. re-plotting)
        load_or_save(
            load_or_save_path, "save", patient_embeddings, outcome_embeddings,
            vocab_embeddings, gold_list,
        )
    
    # Compute and send a scatter-plot of patients" evolutions to tensorboard
    fig = scatter_patients_and_outcomes(
        patient_embeddings, outcome_embeddings, vocab_embeddings, gold_list,
    )
    log_figure_to_board(fig, "outcomization_metric_test", logger, global_step)
    # fig_name = "figures/figure_7_v2/%s.png" % str(model).split("(")[0].lower()
    # fig.tight_layout()
    # plt.savefig(fig_name, dpi=300)
    
    
def reduce_embeddings(
    embeddings: list[np.ndarray],
    vocab_embedding_dict: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """ Combine patient and outcome embeddings into reduced representation
    """
    # Build patient and outcome embedding arrays
    patient_embeddings = np.stack([np.mean(e, axis=0) for e in embeddings])
    outcome_embeddings = np.stack(
        [vocab_embedding_dict[t] for t in ALL_OUTCOME_LEVELS
    ])
    vocab_embeddings = np.stack([
        vocab_embedding_dict[t] for t in vocab_embedding_dict
        if t not in ALL_OUTCOME_LEVELS
    ])
    
    # Collect all embeddings in a single array
    all_embeddings = (patient_embeddings, outcome_embeddings, vocab_embeddings)
    all_embeddings = np.concatenate(all_embeddings)
    # all_embeddings_means = all_embeddings.mean(axis=1, keepdims=True)
    # all_embeddings_stds = all_embeddings.std(axis=1, keepdims=True)
    # all_embeddings = (all_embeddings - all_embeddings_means) / all_embeddings_stds
    
    # Compute reduced representation of combined array (patients, tokens, labels)
    red_embeddings = compute_reduced_representation(
        all_embeddings, tsne_metric="cosine", rdm_metric="correlation",
    )
    
    # Separate reduced patient, vocab, and outcome embeddings
    red_patient_embeddings = red_embeddings[:len(patient_embeddings)]
    red_outcome_embeddings = red_embeddings[
        len(patient_embeddings):len(patient_embeddings) + len(outcome_embeddings)
    ]
    red_vocab_embeddings = red_embeddings[
        len(patient_embeddings) + len(outcome_embeddings):
    ]
    
    # Return all results
    return red_patient_embeddings, red_outcome_embeddings, red_vocab_embeddings


def get_vocab_embeddings_and_counts(
    model: torch.nn.Module,
    pipeline: data.DataPipeline,
) -> tuple:
    """ Compute embedding vector for all tokens of the dataset (even rare ones)
    """
    print("Computing reduced embeddings for all tokens of the testing dataset")
    # Get vocabulary and corresponding counts
    dp = data.JsonReader(pipeline.data_fulldir, "test")
    dp = data.TokenFilter(dp, pipeline.run_params["ngrams_to_remove"])
    all_tokens = [token for sentence in dp for token in sentence]
    vocab, counts = np.unique(all_tokens, return_counts=True)
    
    # Compute reduced embeddigs for all tokens from the vocabulary
    encodings = [pipeline.tokenizer.encode(token) for token in vocab]
    embeddings = model.get_token_embeddings(encodings).numpy()
    
    # Build and return dictionaries
    embedding_dict = {token: emb for token, emb in zip(vocab, embeddings)}
    count_dict = {token: count for token, count in zip(vocab, counts)}
    return embedding_dict, count_dict


def get_patient_embedding(
    sample: list[str],
    vocab_counts: dict[str, int],
    vocab_embeddings: dict[str, np.ndarray],
    use_weights: bool=True,
) -> np.ndarray:
    """ Generate a sequence embedding for a patient sample in which tokens that
        do not belong to a given category were removed
        - The result is a weighted average of all tokens embeddings
        - Weigths are proportional to token inverse frequency (in train dataset)
    """
    # Build a numpy array from patient token embeddings
    embeddings = np.stack([vocab_embeddings[t] for t in sample])
    if not use_weights:
        return embeddings
    
    # Weight each token using inverse token frequency, if required
    weights = np.array([1 / vocab_counts[t] for t in sample], dtype=embeddings.dtype)
    # weights = weights / weights.sum()  # to standardize differently lengthed patients?
    weighted_embeddings = embeddings * weights[:, np.newaxis]
    return weighted_embeddings  # shape (n_tokens, n_features)


def scatter_patients_and_outcomes(
    patient_embeddings: np.ndarray,
    outcome_embeddings: np.ndarray,
    vocab_embeddings: np.ndarray,
    gold_list: list[str],
) -> plt.Figure:
    """ Visualize n_samples patient embeddings for each outcome class
    """
    print("Plotting reduced patient and outcome embeddings")
    fig, axs = plt.subplots(1, len(OUTCOME_CLASSES), figsize=FIG_SIZE)
    for i, (cat, outcome_tokens) in enumerate(OUTCOME_CLASSES.items()):
        
        # Retrieve data
        cat_outcome_embeddings = outcome_embeddings[2 * i:2 * (i + 1)]
        pos_outcome_name = outcome_tokens[0].split("_")[1].lower()  # "LBL_..."
        neg_outcome_name = outcome_tokens[1].split("_")[1].lower()  # "LBL_..."
        pos_patient_embeddings = [
            r for r, g in zip(patient_embeddings, gold_list)
            if outcome_tokens[0] in g
        ]
        neg_patient_embeddings = [
            r for r, g in zip(patient_embeddings, gold_list)
            if outcome_tokens[1] in g
        ]
        
        # Plot retrieved data
        patient_params = {"s": 0.1, "alpha": 0.8, "zorder": 1}
        outcome_params = {"s": 250, "edgecolor": "black", "marker": "*", "zorder": 2}
        vocab_params = {"s": 0.1, "alpha": 0.7, "zorder": 0}
        ax = axs[i]
        ax.set_title("Category %s" % cat, fontsize=BIG_TEXT_SIZE)
        ax.scatter(*np.array(pos_patient_embeddings).T, c="blue", **patient_params)
        ax.scatter(*np.array(neg_patient_embeddings).T, c="red", **patient_params)
        ax.scatter(*np.array(cat_outcome_embeddings[0]).T, c="cyan", **outcome_params)
        ax.scatter(*np.array(cat_outcome_embeddings[1]).T, c="gold", **outcome_params)
        ax.scatter(*np.array(vocab_embeddings).T, c="gray", **vocab_params)
        
        # Polish figure (no ticks nor tick labels, legend with fixed symbol-size)
        label_fn = lambda name, type: "%s %s" % (name.capitalize(), type)
        patient_params = {"marker": "s", "color": "w", "markersize": 10}
        outcome_params = {"marker": "*", "color": "w", "markersize": 10, "markeredgecolor": "black"}
        vocab_params = {"marker": "s", "color": "w", "markersize": 10}
        ax.set_xticklabels([]); ax.set_xticks([])
        ax.set_yticklabels([]); ax.set_yticks([])
        # legend_markers = [
        #     Line2D([0], [0], label=label_fn(pos_outcome_name, "patients"), markerfacecolor="blue", **patient_params),
        #     Line2D([0], [0], label=label_fn(neg_outcome_name, "patients"), markerfacecolor="red", **patient_params),
        #     Line2D([0], [0], label=label_fn(pos_outcome_name, "token"), markerfacecolor="cyan", **outcome_params),
        #     Line2D([0], [0], label=label_fn(neg_outcome_name, "token"), markerfacecolor="gold", **outcome_params),
        #     Line2D([0], [0], label="Single tokens", markerfacecolor="gray", **vocab_params),
        # ]
        # ax.legend(
        #     handles=legend_markers, loc="upper center",
        #     ncol=3, columnspacing=0.75, fontsize=SMALL_TEXT_SIZE,
        # )
        
    # Return final figure
    return fig


def load_or_save(
    path: str,
    mode: str,
    patient_embeddings: list=None,
    outcome_embeddings: list=None,
    vocab_embeddings: list=None,
    gold_list: list=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """ Load or save data for replotting
    """
    # Save embeddings and labels for later use (e.g., replotting)
    if mode == "save":
        with open(path, "wb") as handle:
            dict_to_save = {
                "patient_embeddings": patient_embeddings,
                "outcome_embeddings": outcome_embeddings,
                "vocab_embeddings": vocab_embeddings,
                "gold_list": gold_list}
            pickle.dump(dict_to_save, handle)
    
    # Load embeddings and labels from previously saved run
    else:
        with open(path, "rb") as handle:
            to_load = pickle.load(handle)
            return (
                to_load["patient_embeddings"],
                to_load["outcome_embeddings"],
                to_load["vocab_embeddings"],
                to_load["gold_list"],
            )
            
            
# def get_patient_embedding(sample: list[str],
#                           vocab_counts: dict[str, int],
#                           vocab_embeddings: dict[str, np.ndarray],
#                           ) -> torch.Tensor:
#     """ Generate a sequence embedding for a patient sample in which tokens that
#         do not belong to a given category were removed
#         - The result is a weighted average of all tokens embeddings
#         - Weigths are proportional to token inverse frequency (in train dataset)
#         - N embeddings per patient are generated, N = len(PARTIAL_INFO_LEVELS)
#     """
#     # Get patient tokens embeddings and weight them by term frequencies
#     embeddings = np.stack([vocab_embeddings[t] for t in sample])
#     weights = np.array([1 / vocab_counts[t] for t in sample])
#     weighted_embeddings = embeddings * weights[:, np.newaxis]
        
#     # Separate fixed embeddings from new information
#     is_dem = np.array(["DEM_" in t for t in sample])
#     fixed_embeddings = weighted_embeddings[is_dem]
#     new_embeddings = weighted_embeddings[~is_dem]
    
#     # Build vectors from growing partial information
#     averaged_partial_embedding_list = []
#     for partial in PARTIAL_INFO_LEVELS:
#         n_new_tokens = int(len(new_embeddings) * partial)
#         partial_info = (fixed_embeddings, new_embeddings[:n_new_tokens])
#         partial_embeddings = np.concatenate(partial_info, axis=0)
#         averaged_partial_embedding = np.mean(partial_embeddings, axis=0)
#         averaged_partial_embedding_list.append(averaged_partial_embedding)
        
#     # Return partial sentences stacked over a new dimension
#     import pdb; pdb.set_trace()
#     return np.stack(averaged_partial_embedding_list)
