import os
import pandas as pd
import pytorch_lightning as pl
import numpy as np
import torch
import matplotlib.pyplot as plt
import data
from torchdata.datapipes.iter import Shuffler
from collections import defaultdict
from typing import Union
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from metrics.metric_utils import (
    compute_reduced_representation,
    log_figure_to_board,
    bootstrap_auroc_ci,
    bootstrap_auprc_ci,
)


MAX_SAMPLES = 100_000  # not np.inf, since "MED_" gives too many combinations
LENIENT_LETTER_MATCHES = [1, 2, 3, 4, "Exact"]
MULTI_CATEGORIES = ["DIA_", "PRO_", "MED_"]
BINARY_CLASSES = {
    "mortality": ["LBL_ALIVE", "LBL_DEAD"],
    "readmission": ["LBL_AWAY", "LBL_READM"],
    "length-of-stay": ["LBL_SHORT", "LBL_LONG"],
}
ALL_BINARY_LEVELS = [t for v in BINARY_CLASSES.values() for t in v]
REDUCED_DIM = None  # None for not using dimensionality reduction
USE_TIME_WEIGHTS = False
TIME_WEIGHTS = [1 / (t + 1) for t in range(100_000)][::-1]


def trajectorization_task(
    model: torch.nn.Module,
    pipeline: data.DataPipeline,
    logger: pl.loggers.Logger,
    global_step: int,
) -> None:
    """ Compute prediction task metric for a trained model, where target tokens
        are separated from the patient code sequence, and patient embeddings
        predict target tokens in an unsupervised way, using cosine similarity
    """
    # One run with model predictions, one baseline with random predictions
    for random_mode in [False]:  # [True, False]  <-- random mode only for sanity check
        
        # Compute performance using model embeddings
        print("\nProceeding with the trajectorization testing metric")
        multi_perf = {}
        for cat in MULTI_CATEGORIES:
            multi_perf[cat] = \
                compute_trajectory_prediction(model, pipeline, cat, random_mode)
                
        # Log performance (no plot because not used in this form in the publication)
        generate_csv(multi_perf, logger.save_dir, random_mode)
        # fig = generate_figure(multi_perf)
        # fig_title = "trajectorization_metric_rd_%s_rc_%s" % \
        #     (REDUCED_DIMENSIONALITY, RANDOM_MODE)
        # log_fig_to_tensorboard(fig, fig_title, logger, global_step)
    

def compute_trajectory_prediction(
    model: torch.nn.Module,
    pipeline: data.DataPipeline,
    cat: str,
    random_mode: bool,
) -> dict[str, dict[str, float]]:
    """ Compute top-k accuracy for a multi-label classification task
        - A model embeds sequences of tokens in which all tokens belonging
        to a category are removed
        - Sequence embeddings are compared to label token embeddings
        - Top-k accuracy is computed based on cosine similarity
    """
    # Retrieve data and initialize prediction sample-label pairs
    labels, label_embeddings = get_labels_multi(model, pipeline, cat)
    to_remove = pipeline.run_params["ngrams_to_remove"] + ALL_BINARY_LEVELS
    dp = data.JsonReader(pipeline.data_fulldir, "valid")
    dp = data.TokenFilter(dp, to_remove=to_remove)  # to_split=[cat])
    if cat == "DIA_":
        dp = generate_first_diagnosis_pred_pairs(dp)
    else:
        dp = generate_traj_pred_pairs(dp, cat)
        dp = Shuffler(dp)  # to avoid taking everything from the first samples
        
    # Compute patient embeddings and store gold labels
    encode_fn = pipeline.tokenizer.encode
    unk_encoding = encode_fn("[UNK]")
    patient_embeddings, golds = [], []
    for n, (sample, gold) in enumerate(
        tqdm(dp, desc=" - Patients embedded - no %s token" % cat)
    ):
        if n >= MAX_SAMPLES: break
        if pipeline.tokenizer.encode(gold) == unk_encoding: continue
        if len(gold) == 0: continue
        embedding = get_patient_embedding(model, sample, pipeline, random_mode)
        patient_embeddings.append(embedding)
        golds.append([gold])
    
    # Compute prediction scores based on patient-label embedding similarities
    print(" - Comparing patient embeddings to %s tokens" % cat)
    patient_embeddings = torch.stack(patient_embeddings, dim=0)
    patient_embeddings, label_embeddings = \
        reduce_embeddings(patient_embeddings, label_embeddings)
    scores = cosine_similarity(patient_embeddings, label_embeddings)
    
    # Return performance for different lenient settings, using the scores
    return {
        lenient: compute_micro_performance(scores, golds, labels, lenient)
        for lenient in LENIENT_LETTER_MATCHES
    }


def generate_first_diagnosis_pred_pairs(dp):
    """ Generate input label pairs for the first (i.e., most important) diagnosis
        prediction task
    """
    for sample in dp:
        if any(["DIA_" in token for token in sample]):
            input_data = [token for token in sample if "DIA_" not in token]
            label = next(token for token in sample if "DIA_" in token)
            yield (input_data, label)
            

def generate_traj_pred_pairs(dp, cat):
    """ Generate all possible sample-label pairs of trajectory prediction for
        each sample, using any category of predicted tokens as labels
        E.g.: [A1, A2, B1, A3, B2] -> [([A1, A2], B1), ([A1, A2, B1, A3], B2)]
    """
    condition = lambda token: cat in token
    for sample in dp:
        for j, elem in enumerate(sample):
            if condition(elem):
                yield (sample[:j], elem)  # (input_data, label)
                

def compute_micro_performance(
    scores: np.ndarray,
    golds: list[set[str]],
    labels: list[str],
    n_matched_letters: int,
) -> dict[str, Union[list[float], float]]:
    """ Evaluate model embedding performance using micro averaged metrics
    """
    # Load collapsed scores and labels given lenient level
    scores, golds, labels = \
        lenient_match_collapse(scores, golds, labels, n_matched_letters)
    one_hotter = MultiLabelBinarizer(classes=labels)
    golds = one_hotter.fit_transform(golds)
    
    # Post-process labels for evaluation
    # scores, golds = filter_out_rare_classes(scores, golds)  # not used in the end
    scores, golds = scores.ravel(), golds.ravel()
    
    # Compute AUROC (+ CI) for current number of matched letters
    print("\n --- Computing AUROC for %s letters" % n_matched_letters, end="")
    auroc, auroc_std, auroc_ste = bootstrap_auroc_ci(golds, scores)
    
    # Compute AUPRC (+ CI) for current number of matched letters
    print("\n --- Computing AUPRC for %s letters" % n_matched_letters, end="")
    auprc, auprc_std, auprc_ste = bootstrap_auprc_ci(golds, scores)
    
    return {
        "auroc": auroc, "auroc_std": auroc_std, "auroc_ste": auroc_ste,
        "auprc": auprc, "auprc_std": auprc_std, "auprc_ste": auprc_ste,
    }


def filter_out_rare_classes(
    scores: np.ndarray,
    golds: np.ndarray,
    n_classes_to_keep: int=500,
) -> None:
    """ Remove columns corresponding to rare classes in score and gold arrays
    """
    # Case where no filtering is needed (i.e., not too many classes)
    if golds.shape[1] <= n_classes_to_keep:
        return scores, golds
    
    # Remove rare classes (mostly classes with only one sample)
    gold_column_sums = np.sum(golds, axis=0)
    largest_sum_columns = np.argsort(gold_column_sums)[-n_classes_to_keep:]
    scores = scores[:, largest_sum_columns]
    golds = golds[:, largest_sum_columns]
    
    # Remove samples whose classes were removed (to avoid NaN in AUROC / AUPRC)
    rows_with_a_class = np.sum(golds, axis=1) > 0
    scores = scores[rows_with_a_class, :]
    golds = golds[rows_with_a_class, :]
    
    return scores, golds
    

def lenient_match_collapse(
    scores: np.ndarray,
    golds: list[set[str]],
    labels: list[str],
    n_matched_letters: int,
) -> tuple[np.ndarray, list[str]]:
    """ Collapse scores, golds and labels given lenient match level
    """
    if n_matched_letters == "Exact": return scores, golds, labels
    score_map, gold_map = defaultdict(list), {}
    for i, label in enumerate(labels):
        
        key = label.split("_")[-1][:n_matched_letters]
        score_map[key].append(i)
        gold_map[label] = key
        
    new_scores = [scores[:, cols].sum(axis=1) for cols in score_map.values()]
    new_scores = np.stack(new_scores, axis=-1)
    new_golds = [set([gold_map.get(key) for key in g]) for g in golds]
    new_labels = list(score_map.keys())
    
    return new_scores, new_golds, new_labels


# def get_labels_multi(
#     model: torch.nn.Module,
#     pipeline: data.DataPipeline,
#     cat: str,
# ) -> tuple[list[int], torch.Tensor]:
#     """ Generate embeddings for all possible tokens of a given category
#         - Tokens are retrieved from the tokenizer vocabulary
#         - The result is a dict of token indices and corresponding embeddings
#     """
#     vocab = pipeline.tokenizer.get_vocab()
#     labels = [t for t in vocab if cat in t and cat != t]
#     label_encodings = [pipeline.tokenizer.encode(t) for t in labels]
#     label_embeddings = model.get_token_embeddings(label_encodings)
#     return labels, label_embeddings


def get_labels_multi(
    model: torch.nn.Module,
    pipeline: data.DataPipeline,
    cat: str,
) -> tuple[list[int], torch.Tensor]:
    """ Compute embedding vector for all tokens of the dataset (even rare ones)
    """
    print("Computing reduced embeddings for all tokens of the testing dataset")
    # Get vocabulary and corresponding counts
    dp = data.JsonReader(pipeline.data_fulldir, "test")
    dp = data.TokenFilter(dp, pipeline.run_params["ngrams_to_remove"])
    all_tokens = [token for sentence in dp for token in sentence]
    vocab_with_rare_tokens, counts = np.unique(all_tokens, return_counts=True)
    
    # Embed labels    
    labels = [t for t in vocab_with_rare_tokens if cat in t and cat != t]
    label_encodings = [pipeline.tokenizer.encode(t) for t in labels]
    label_embeddings = model.get_token_embeddings(label_encodings)
    
    return labels, label_embeddings


def get_patient_embedding(
    model: torch.nn.Module,
    sample: list[str],
    pipeline: data.DataPipeline,
    random_mode: bool,
) -> torch.Tensor:
    """ Generate a sequence embedding for a patient sample in which tokens that
        do not belong to a given category were removed
        - The result is a weighted average of all tokens embeddings
        - Weigths are proportional to token inverse frequency (in train dataset)
    """
    # This allows to compute performance for a random classifier
    if random_mode:
        one_encoding = [pipeline.tokenizer.encode(sample[0])]
        embed_dim = model.get_sequence_embeddings(one_encoding).shape[-1]
        return 2 * (torch.rand(embed_dim) - 0.5)
    
    # Encode patient tokens and compute weights based on term frequencies
    encoded = [pipeline.tokenizer.encode(t) for t in sample]
    if isinstance(encoded[0], list):
        weights = [1 / pipeline.tokenizer.word_counts[t[0]] for t in encoded]
    else:
        weights = [1 / pipeline.tokenizer.word_counts[t] for t in encoded]
    
    # Apply eligibility trace if required
    if USE_TIME_WEIGHTS:
        time_weights = TIME_WEIGHTS[-len(weights):]
        weights = [w * t for w, t in zip(weights, time_weights)]
    
    # Return fixed-length embedding vector for one patient admission
    return model.get_sequence_embeddings(encoded, weights)


def reduce_embeddings(
    patient_embeddings: torch.Tensor,
    label_embeddings: torch.Tensor,
) -> np.ndarray:
    """ Reduce the dimensionality of high-dimensional concept embeddings
        - Patients and labels are concatenated for dimensionality reduction
        - Then, the reduced representations are returned separately
    """
    if REDUCED_DIM is None:
        return patient_embeddings.numpy(), label_embeddings.numpy()
    embeddings = torch.cat((patient_embeddings, label_embeddings), dim=0).numpy()
    reduced = compute_reduced_representation(data=embeddings, dim=REDUCED_DIM)
    reduced_patient_embeddings = reduced[:patient_embeddings.shape[0]]
    reduced_label_embeddings = reduced[patient_embeddings.shape[0]:]
    return reduced_patient_embeddings, reduced_label_embeddings
    

def generate_csv(
    multi_perfs: dict[str, list[float]],
    save_dir: str,
    random_mode: bool,
) ->  plt.Figure:
    """ Generate a csv file that summarizes the results of the prediction task
    """
    # Initialize headers and rows to write
    base_specs = ["1L", "2L", "3L", "4L", "EM"]
    suffixes = ["", "-STD", "-STE"]
    specs = [
        "%s%s" % (base_spec, suffix) if suffix else base_spec
        for base_spec in base_specs for suffix in suffixes
    ]
    heads = ["Category"] + \
            ["AUROC-" + spec for spec in specs] + \
            ["AUPRC-" + spec for spec in specs]
    rows = []
    
    # Fill the rows with the results
    for cat, multi_results in multi_perfs.items():
        new_row = [cat]
        for metric in ["auroc", "auprc"]:
            for multi_result in multi_results.values():  # loop over 1L, 2L, ...
                new_row.append("%.03f" % multi_result[metric])
                new_row.append("%.03f" % multi_result[metric + "_std"])
                new_row.append("%.03f" % multi_result[metric + "_ste"])
        
        rows.append(new_row)

    # Save the results as a csv file in the correct logs directory
    csv_path = \
        "trajectorization_results_rd_%s_rc_%s.csv" % (REDUCED_DIM, random_mode)
    save_path = os.path.join(save_dir, csv_path)
    df = pd.DataFrame(rows, columns=heads)
    df.to_csv(save_path, index=False, header=heads)


# def generate_figure(multi_results: dict[str, list[float]]) ->  plt.Figure:
#     """ Generate a big figure that contains all plots of the prediction task
#     """
#     # Multi category prediction subplot
#     print(" - Plotting figure with all prediction results")    
#     fig, axs = plt.subplots(3, 5, figsize=(8, 10))
#     for row, (cat, results) in enumerate(multi_results.items()):
#         for col, (letter_lenient_match, result) in enumerate(results.items()):
#             ax = axs[row, col]

#             params = {
#                 "label": "AVG-PR: %.03f" % result["metric_prc"],
#                 "color": "C0",  # "C%1i" % i
#             }
#             ax.plot(result["rec"], result["prec"], **params)
            
#             info = "%s letter(s) match" % letter_lenient_match
#             ax.set_xlabel("Recall (%s) - %s" % (cat.split("_")[0], info),
#                           fontsize="small",
#                           labelpad=0.5)
#             ax.set_ylabel("Precision (%s)" % cat.split("_")[0],
#                           fontsize="small",
#                           labelpad=0.5)
#             ax.set_ylim(0.0, 1.0)
#             ax.tick_params(axis="both", labelsize="x-small", pad=0.5)
#             ax.legend(fontsize="x-small", labelspacing=0.5)
#             ax.grid()
    
#     # Send figure to tensorboard
#     return fig
