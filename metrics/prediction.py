import os
import pandas as pd
import pytorch_lightning as pl
import numpy as np
import torch
import matplotlib.pyplot as plt
import data
from collections import defaultdict
from typing import Union
from tqdm import tqdm
from torchdata.datapipes.iter import Shuffler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from metrics.metric_utils import (
    compute_reduced_representation,
    log_figure_to_board,
    bootstrap_auroc_ci,
    bootstrap_auprc_ci,
)


MAX_SAMPLES = np.inf  # set to < np.inf for debug!
PARTIAL_LEVELS = [0.0, 0.1, 0.3, 0.6, 1.0]
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


def prediction_task(
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
        
        # Compute performance using model embeddings (or random embeddings)
        to_print = "\nProceeding with prediction testing metric"
        print(to_print if not random_mode else to_print + " (random baseline)")
        
        # Multi-class prediction performance (ICD10-CM, ICD10-PCS, ATC)
        multi_perf = {}
        for cat in MULTI_CATEGORIES:
            multi_perf[cat] = \
                compute_prediction_multi(model, pipeline, cat, random_mode)       
        
        # Binary prediction performance (mortality, readmission, length-of-stay)
        binary_perf = {}
        for k, v in BINARY_CLASSES.items():
            binary_perf[k] = \
                compute_prediction_binary(model, pipeline, v, random_mode)
        
        # Log performance (no plot because not used in this form in the publication)
        generate_csv(multi_perf, binary_perf, logger.save_dir, random_mode)
        # fig = generate_figure(multi_perf, binary_perf)
        # fig_title = "prediction_metric_rd_%s_rc_%s" % (REDUCED_DIM, random_mode)
        # log_figure_to_board(fig, fig_title, logger, global_step)
        

def compute_prediction_multi(
    model: torch.nn.Module,
    pipeline: data.DataPipeline,
    cat: str,
    random_mode: bool,
) -> dict[str, dict[str, float]]:
    """ Compute top-k accuracy for a multi-label classification task
        - A model embeds sequences of tokens in which all tokens belonging to a
        category are removed
        - Sequence embeddings are compared to label token embeddings
        - Top-k accuracy is computed based on cosine similarity
    """
    # Retrieve labels and initialize
    encode_fn = pipeline.tokenizer.encode
    unk_encoding = encode_fn("[UNK]")
    labels, label_embeddings = get_labels_multi(model, pipeline, cat)
    # to_remove = pipeline.run_params["ngrams_to_remove"]
    to_remove = pipeline.run_params["ngrams_to_remove"] + ALL_BINARY_LEVELS
    dp = data.JsonReader(pipeline.data_fulldir, "valid")
    dp = data.TokenFilter(dp, to_remove=to_remove, to_split=[cat])
    
    # Compute patient embeddings and store gold labels
    patient_embeddings, golds = [], []
    loop = tqdm(dp, desc=" - Embedding patients (no %s tokens)" % cat)
    for n, (sample, gold) in enumerate(loop):
        if n > MAX_SAMPLES: break
        if len(gold) == 0: continue
        embedding = get_patient_embedding(model, sample, pipeline, random_mode)
        gold_set = set([g for g in gold if encode_fn(g) != unk_encoding])
        patient_embeddings.append(embedding)
        golds.append(gold_set)
    
    # Compute prediction scores based on patient-label embedding similarities
    print(" - Comparing patient embeddings to %s tokens" % cat)
    patient_embeddings = torch.cat(patient_embeddings, dim=0)
    patient_embeddings, label_embeddings = \
        reduce_embeddings(patient_embeddings, label_embeddings)
    scores = cosine_similarity(patient_embeddings, label_embeddings)
    
    # Return performance for different lenient settings, using the scores
    return {
        lenient: compute_micro_performance(scores, golds, labels, lenient)
        for lenient in LENIENT_LETTER_MATCHES
    }


def compute_micro_performance(
    scores: np.ndarray,
    golds: list[set[str]],
    labels: list[str],
    n_matched_letters: int,
) -> dict[str, Union[list[float], float]]:
    """ Evaluate model embedding performance using micro averaged metrics
    """
    # Load collapsed scores and labels given lenient level
    lenient_scores, lenient_golds, lenient_labels = \
          lenient_match_collapse(scores, golds, labels, n_matched_letters)
    one_hotter = MultiLabelBinarizer(classes=lenient_labels)
    one_hot_lenient_golds = one_hotter.fit_transform(lenient_golds).ravel()
    
    # Compute AUROC and AUPRC for given lenient level
    perf_dict = {}
    desc = " --- Computing AUROC & AUPRC for %s letters" % n_matched_letters
    for i, partial in tqdm(list(enumerate(PARTIAL_LEVELS)), desc=desc):
        
        # Compute AUROC and AUPRC (+ CI) for this level of partial information
        partial_scores = lenient_scores[i::len(PARTIAL_LEVELS)].ravel()
        import ipdb; ipdb.set_trace()
        auroc, auroc_std, auroc_ste = \
            bootstrap_auroc_ci(one_hot_lenient_golds, partial_scores)
        auprc, auprc_std, auprc_ste = \
            bootstrap_auprc_ci(one_hot_lenient_golds, partial_scores)
        perf_dict[partial] = {
            "auroc": auroc, "auroc_std": auroc_std, "auroc_ste": auroc_ste,
            "auprc": auprc, "auprc_std": auprc_std, "auprc_ste": auprc_ste,
        }
    
    # Return performance dictionary
    return perf_dict


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


def compute_prediction_binary(
    model: torch.nn.Module,
    pipeline: data.DataPipeline,
    classes: list[str],
    random_mode: bool,
) -> float:
    """ Compute accuracy for a binary classification task
        - A model embeds sequences of tokens in which the binary classification
        tokens is removed
        - Sequence embeddings are compared to both class token embeddings
        - Accuracy is computed based on cosine similarity (closest = prediction)
        
        Note: same patient embeddings are computed for each pair of classes,
              which is pretty inefficient, but I had no time to correct this! 
    """
    # Pipeline removing all class tokens and splitting the predicted class tokens
    class_embeddings = get_labels_binary(model, pipeline, classes)
    to_remove = pipeline.run_params["ngrams_to_remove"] +\
                [t for t in ALL_BINARY_LEVELS if t not in classes]
    dp = data.JsonReader(pipeline.data_fulldir, "test")
    dp = data.TokenFilter(dp, to_remove=to_remove, to_split=classes)
    
    # Compute patient embeddings and store gold labels
    patient_embeddings, golds = [], []
    loop = tqdm(enumerate(dp), desc=" - Embedding patients (no class tokens)")
    for n, (sample, gold) in loop:
        if n > MAX_SAMPLES: break
        embedding = get_patient_embedding(model, sample, pipeline, random_mode)
        patient_embeddings.append(embedding)
        golds.append(gold)
    
    # Compute similarities between patient embeddings and class tokens
    print(" - Comparing patient embeddings to %s tokens" % classes)
    patient_embeddings = torch.cat(patient_embeddings, dim=0)
    patient_embeddings, class_embeddings = \
        reduce_embeddings(patient_embeddings, class_embeddings)
    similarities = cosine_similarity(patient_embeddings, class_embeddings)
    
    # Compute prediction of probability based on cosine similarity with labels
    perf = {}
    desc = " --- Computing AUROC & AUPRC for binary prediction task"
    for i, partial in tqdm(list(enumerate(PARTIAL_LEVELS)), desc=desc):
        probs, trues = [], []
        for similarity, gold in zip(similarities[i::len(PARTIAL_LEVELS)], golds):
            distance = 1.0 - similarity
            probs.append(distance[0] / distance.sum())  # between 0.0 and 1.0
            trues.append(classes.index(gold[0]))  # index of true
        
        # Compute AUROC and AUPRC (+ CI)
        trues, probs = np.array(trues), np.array(probs)
        auroc, auroc_std, auroc_ste = bootstrap_auroc_ci(trues, probs)
        auprc, auprc_std, auprc_ste = bootstrap_auprc_ci(trues, probs)
        perf[partial] = {
            "auroc": auroc, "auroc_std": auroc_std, "auroc_ste": auroc_ste,
            "auprc": auprc, "auprc_std": auprc_std, "auprc_ste": auprc_ste,
        }
        
    return perf


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


def get_labels_binary(
    model: torch.nn.Module,
    pipeline: data.DataPipeline,
    classes: list[str]
) -> dict[list[str], torch.Tensor]:
    """ Generate embeddings for a given pair of classes
        - The result is a dict of token classes and corresponding embeddings
    """
    assert len(classes) == 2, "Number of classes should be 2."
    class_encodings = [pipeline.tokenizer.encode(c) for c in classes]
    class_embeddings = model.get_token_embeddings(class_encodings)
    return class_embeddings
    

def get_patient_embedding(
    model: torch.nn.Module,
    sample: list[str],
    pipeline: data.DataPipeline,
    random_mode: bool,
    stack_partial_levels: bool=True,
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
        return 2 * (torch.rand((len(PARTIAL_LEVELS), embed_dim)) - 0.5)
    
    # Encode patient tokens and compute weights based on term frequencies
    encodings = [pipeline.tokenizer.encode(t) for t in sample]
    if isinstance(encodings[0], list):
        weights = [1 / pipeline.tokenizer.word_counts[t[0]] for t in encodings]
    else:
        weights = [1 / pipeline.tokenizer.word_counts[t] for t in encodings]
    
    # Return embeddings as required
    if stack_partial_levels:
        return stack_partial_level_fn(model, sample, encodings, weights)
    else:
        return model.get_sequence_embeddings(encodings, weights)
    

def stack_partial_level_fn(model, sample, encodings, weights):
    """ Encode a sequence at different levels of partial information and return
        a list of embedding vectors stacked over a new dimension
    """
    # Compute patient embeddings for different levels of partial information
    fixed_enc = [
        encodings[n] for n, t in enumerate(sample)
        if "DEM_" in t and not any([s in t for s in BINARY_CLASSES])
    ]
    fixed_wgt = [weights[n] for n, t in enumerate(sample) if "DEM_" in t]
    timed_idx = [n for n, t in enumerate(sample) if "DEM_" not in t]
    partial_patient_embeddings = []
    for partial in PARTIAL_LEVELS:
        
        # Get partial encodings and associated weights
        partial_timed_idx = timed_idx[:int(len(timed_idx) * partial)]
        enc = fixed_enc + [encodings[i] for i in partial_timed_idx]
        wgt = fixed_wgt + [weights[i] for i in partial_timed_idx]
        
        # Apply eligibility trace if required and compute sentence embeddings
        if USE_TIME_WEIGHTS:
            time_wgt = TIME_WEIGHTS[-len(wgt):]
            wgt = [w * t for w, t in zip(wgt, time_wgt)]
        
        # Append partial patient embedding to the list
        partial_patient_embeddings.append(model.get_sequence_embeddings(enc, wgt))
    
    # Return partial patient sequences stacked over a new dimension
    return torch.stack(partial_patient_embeddings, dim=0)


def generate_csv(
    multi_perfs: dict[str, list[float]],
    binary_perfs: dict[str, float],
    save_dir: str,
    random_mode: bool,
) ->  plt.Figure:
    """ Generate a csv summary of the multi and binary prediction tasks
    """
    # Initialize headers and rows to write
    base_specs = ["1L", "2L", "3L", "4L", "EM"]
    suffixes = ["", "-STD", "-STE"]
    specs = [
        "%s%s" % (base_spec, suffix) if suffix else base_spec
        for base_spec in base_specs for suffix in suffixes
    ]
    heads = ["Category", "Partial"] + \
            ["AUROC-" + spec for spec in specs] + \
            ["AUPRC-" + spec for spec in specs] + \
            ["AUROC-BI" + suffix for suffix in suffixes] + \
            ["AUPRC-BI" + suffix for suffix in suffixes]
    
    # Fill the rows with the results
    rows = []
    for (cat, multi_results), (_, binary_result) in \
        zip(multi_perfs.items(), binary_perfs.items()):
        new_rows = [[cat, "%s" % p] for p in PARTIAL_LEVELS]
        
        # Update rows with multi-class prediction metric (predict medical tokens)
        for metric in ["auroc", "auprc"]:
            for multi_result in multi_results.values():
                for new_row, perf in zip(new_rows, multi_result.values()):
                    new_row.append("%.03f" % perf[metric])
                    new_row.append("%.03f" % perf[metric + "_std"])
                    new_row.append("%.03f" % perf[metric + "_ste"])
        
        # Update rows with binary prediction metric (predict outcome tokens)
        for metric in ["auroc", "auprc"]:
            for new_row, perf in zip(new_rows, binary_result.values()):
                new_row.append("%.03f" % perf[metric])
                new_row.append("%.03f" % perf[metric + "_std"])
                new_row.append("%.03f" % perf[metric + "_ste"])
                
        # Add all rows
        rows.extend(new_rows)
    
    # Save the results as a csv file in the correct logs directory
    csv_filepath = "prediction_results_rd_%s_rc_%s.csv" % \
        (REDUCED_DIM, random_mode)
    save_path = os.path.join(save_dir, csv_filepath)
    df = pd.DataFrame(rows, columns=heads)
    df.to_csv(save_path, index=False, header=heads)
    
    
# def generate_figure(
#     multi_results: dict[str, list[float]],
#     binary_results: dict[str, float]
# ) ->  plt.Figure:
#     """ Generate a big figure that contains all plots of the prediction task
#     """
#     # Create main figure plot
#     print(" - Plotting figure with all prediction results")
#     fig = plt.figure(figsize=(15, 10), constrained_layout=True)
#     subfigs = fig.subfigures(3, 1, height_ratios=[2.5, 1, 1])

#     # Multi category prediction subplot
#     axs = subfigs[0].subplots(3, 5)
#     for row, (cat, results) in enumerate(multi_results.items()):
#         for col, (letter_lenient_match, result) in enumerate(results.items()):
#             ax = axs[row, col]
#             for i, (partial, perf) in enumerate(result.items()):

#                 params = {
#                     "label": "Partial: %s - AVG-PR: %.03f" %\
#                         (partial, perf["metric_prc"]),
#                     "color": "C%1i" % i,
#                 }
#                 ax.plot(perf["rec"], perf["prec"], **params)
            
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
    
#     # Binary prediction subplot (receiver-operating-characteristics curve)
#     axs = subfigs[1].subplots(1, 3)
#     for ax, (i, (cat, result)) in zip(axs, enumerate(binary_results.items())):
#         for i, (partial, perf) in enumerate(result.items()):
#             label = "Partial: %s - AUROC: %.03f" % (partial, perf["metric_roc"])
#             ax.plot(perf["fpr"], perf["tpr"], color="C%1i" % i, label=label)
#         ax.plot((0, 1), (0, 1), "--", color="gray")
#         ax.set_xlabel("FP rate (%s)" % cat, labelpad=0.5)
#         ax.set_ylabel("TP rate (%s)" % cat, labelpad=0.5)
#         ax.set_ylim(0.0, 1.0)
#         ax.tick_params(axis="both", labelsize="smaller", pad=0.5)
#         ax.legend(fontsize="smaller")
#         ax.grid()
    
#     # Binary prediction subplot (precision-recall curve)
#     axs = subfigs[2].subplots(1, 3)
#     for ax, (i, (cat, result)) in zip(axs, enumerate(binary_results.items())):
#         for i, (partial, perf) in enumerate(result.items()):
#             label = "Partial: %s - AVG-PR: %.03f" % (partial, perf["metric_prc"])
#             ax.plot(perf["rec"], perf["prec"], color="C%1i" % i, label=label)
#         ax.set_xlabel("Recall (%s)" % cat, labelpad=0.5)
#         ax.set_ylabel("Precision (%s)" % cat, labelpad=0.5)
#         ax.set_ylim(0.0, 1.0)
#         ax.tick_params(axis="both", labelsize="smaller", pad=0.5)
#         ax.legend(fontsize="smaller")
#         ax.grid()

#     # Send figure to tensorboard
#     return fig
