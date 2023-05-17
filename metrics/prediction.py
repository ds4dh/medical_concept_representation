import cProfile
import pstats
import os
import pandas as pd
import pytorch_lightning as pl
import numpy as np
import torch
import matplotlib.pyplot as plt
import data
import tempfile
from collections import defaultdict
from typing import Union, Callable
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve as pr_curve,
    average_precision_score as avg_prec
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


RANDOM_MODE = False
MAX_SAMPLES = np.inf  # used for debug if < np.inf
PARTIALS = [0.0, 0.1, 0.3, 0.6, 1.0]
LENIENT_LETTER_MATCHES = [1, 2, 3, 4, 'Exact']
MULTI_CATEGORIES = ['DIA_', 'PRO_', 'MED_']
BINARY_CLASSES = {
    'mortality': ['LBL_ALIVE', 'LBL_DEAD'],
    'readmission': ['LBL_AWAY', 'LBL_READM'],
    'length-of-stay': ['LBL_SHORT', 'LBL_LONG'],
}
ALL_BINARY_LEVELS = [t for v in BINARY_CLASSES.values() for t in v]
DIMENSIONALITY_REDUCTION_ALGORITHM = 'pca'  # 'tsne', 'pca'
REDUCED_DIMENSIONALITY = None  # None for not using dimensionality reduction
USE_TIME_WEIGHTS = True
TIME_WEIGHTS = [1 / (t + 1) for t in range(100_000)][::-1]


def profile_it(fn: Callable):
    """ Decorator used to profile execution time of any function
    """
    def profiled_fn(*args, **kwargs):
        with cProfile.Profile() as pr:
            result = fn(*args, **kwargs)
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME) 
        stats.dump_stats(filename='profiling.prof')
        return result
    return profiled_fn


def prediction_task(model: torch.nn.Module,
                    pipeline: data.DataPipeline,
                    logger: pl.loggers.tensorboard.TensorBoardLogger,
                    global_step: int,
                    ) -> None:
    """ Compute prediction task metric for a trained model, where target tokens
        are separated from the patient code sequence, and patient embeddings
        predict target tokens in an unsupervised way, using cosine similarity
    """
    print('\nProceeding with prediction testing metric')
    multi_perf, binary_perf = {}, {}
    for cat in MULTI_CATEGORIES:
        multi_perf[cat] = compute_prediction_multi(model, pipeline, cat)
    for k, v in BINARY_CLASSES.items():
        binary_perf[k] = compute_prediction_binary(model, pipeline, v)
    
    generate_csv(multi_perf, binary_perf, logger)
    fig = generate_figure(multi_perf, binary_perf)
    fig_title = 'prediction_metric_rd_%s_rc_%s' %\
        (REDUCED_DIMENSIONALITY, RANDOM_MODE)
    log_figure_to_tensorboard(fig, fig_title, logger, global_step)
    

def compute_prediction_multi(model: torch.nn.Module,
                             pipeline: data.DataPipeline,
                             cat: str,
                             ) -> dict[str, dict[str, float]]:
    """ Compute top-k accuracy for a multi-label classification task
        - A model embeds sequences of tokens in which all tokens belonging to a
        category are removed
        - Sequence embeddings are compared to label token embeddings
        - Top-k accuracy is computed based on cosine similarity
    """
    # Retrieve labels and initialize
    unk_encoding = pipeline.tokenizer.encode('[UNK]')
    labels, label_embeddings = get_labels_multi(model, pipeline, cat)
    # to_remove = pipeline.run_params['ngrams_to_remove']
    to_remove = pipeline.run_params['ngrams_to_remove'] + ALL_BINARY_LEVELS
    dp = data.JsonReader(pipeline.data_fulldir, 'valid')
    dp = data.TokenFilter(dp, to_remove=to_remove, to_split=[cat])
    
    # Compute patient embeddings and store gold labels
    patient_embeddings, golds = [], []
    loop = tqdm(dp,
                desc=' - Embedding patients (no %s tokens)' % cat,
                leave=False)
    for n, (sample, gold) in enumerate(loop):
        if n > MAX_SAMPLES: break
        if len(gold) == 0: continue
        patient_embeddings.append(get_patient_embedding(model, sample, pipeline))
        golds.append(set([g for g in gold
                     if pipeline.tokenizer.encode(g) != unk_encoding]))
    
    # Compute prediction scores based on patient-label embedding similarities
    print(' - Comparing patient embeddings to %s tokens' % cat)
    patient_embeddings = torch.cat(patient_embeddings, dim=0)
    patient_embeddings, label_embeddings =\
        reduce_dimensionality(patient_embeddings, label_embeddings)
    scores = cosine_similarity(patient_embeddings, label_embeddings)
    
    # Return performance for different lenient settings, using the scores
    return {
        llm: compute_micro_performance(scores, golds, labels, llm)
        for llm in LENIENT_LETTER_MATCHES
    }


def compute_micro_performance(scores: np.ndarray,
                              golds: list[set[str]],
                              labels: list[str],
                              n_matched_letters: int,
                              ) -> dict[str, Union[list[float], float]]:
    """ ...
    """
    lenient_scores, lenient_golds, lenient_labels =\
          lenient_match_collapse(scores, golds, labels, n_matched_letters)
    one_hotter = MultiLabelBinarizer(classes=lenient_labels)
    lenient_golds = one_hotter.fit_transform(lenient_golds).ravel()
    
    perf = {}
    desc = ' - Precision & recall computed for %s letters' % n_matched_letters
    for i, partial in tqdm(enumerate(PARTIALS), desc=desc, leave=False):

        partial_scores = lenient_scores[i::len(PARTIALS)].ravel()
        prec, rec, _ = pr_curve(lenient_golds, partial_scores)
        fpr, tpr, _ = roc_curve(lenient_golds, partial_scores)
        metric_prc = avg_prec(lenient_golds, partial_scores, average='micro')
        metric_roc = auc(fpr, tpr)
        perf[partial] = {'prec': prec, 'rec': rec, 'metric_prc': metric_prc,
                         'fpr': fpr, 'tpr': tpr, 'metric_roc': metric_roc}
    
        print(partial, n_matched_letters, metric_roc, metric_prc)
    
    return perf


def lenient_match_collapse(scores: np.ndarray,
                           golds: list[set[str]],
                           labels: list[str],
                           n_matched_letters: int,
                           ) -> tuple[np.ndarray, list[str]]:
    """ ...
    """
    if n_matched_letters == 'Exact': return scores, golds, labels
    score_map, gold_map = defaultdict(list), {}
    for i, label in enumerate(labels):

        key = label.split('_')[-1][:n_matched_letters]
        score_map[key].append(i)
        gold_map[label] = key

    new_scores = [scores[:, cols].sum(axis=1) for cols in score_map.values()]
    new_scores = np.stack(new_scores, axis=-1)
    new_golds = [set([gold_map.get(key) for key in g]) for g in golds]
    new_labels = list(score_map.keys())

    return new_scores, new_golds, new_labels


def compute_prediction_binary(model: torch.nn.Module,
                              pipeline: data.DataPipeline,
                              classes: list[str]
                              ) -> float:
    """ Compute accuracy for a binary classification task
        - A model embeds sequences of tokens in which the binary classification
        tokens is removed
        - Sequence embeddings are compared to both class token embeddings
        - Accuracy is computed based on cosine similarity (closest = prediction)
    """
    # Retrieve labels and initialize
    class_embeddings = get_labels_binary(model, pipeline, classes)
    to_remove = pipeline.run_params['ngrams_to_remove'] +\
                [t for t in ALL_BINARY_LEVELS if t not in classes]
    dp = data.JsonReader(pipeline.data_fulldir, 'test')
    dp = data.TokenFilter(dp, to_remove=to_remove, to_split=classes)
    
    # Compute patient embeddings and store gold labels
    patient_embeddings, golds = [], []
    loop = tqdm(enumerate(dp),
                desc=' - Embedding patients (no %s tokens)' % classes,
                leave=False)
    for n, (sample, gold) in loop:
        if n > MAX_SAMPLES: break
        patient_embeddings.append(get_patient_embedding(model, sample, pipeline))
        golds.append(gold)
    
    # Compute similarities between patient embeddings and class tokens
    print(' - Comparing patient embeddings to %s tokens' % classes)
    patient_embeddings = torch.cat(patient_embeddings, dim=0)
    similarities = cosine_similarity(patient_embeddings, class_embeddings)
    
    # Compute prediction of probability based on cosine similarity with labels
    perf = {}
    for i, partial in enumerate(PARTIALS):
        probs, trues = [], []
        for similarity, gold in zip(similarities[i::len(PARTIALS)], golds):
            distance = 1.0 - similarity
            probs.append(distance[0] / distance.sum())  # between 0.0 and 1.0
            trues.append(classes.index(gold[0]))  # index of true
        
        # Compute area under the receiver operating characteristic curve
        fpr, tpr, _ = roc_curve(trues, probs, pos_label=1)
        prec, rec, _ = pr_curve(trues, probs, pos_label=1)
        metric_roc = auc(fpr, tpr)
        metric_prc = avg_prec(trues, probs, pos_label=1)
        perf[partial] = {'fpr': fpr, 'tpr': tpr, 'metric_roc': metric_roc,
                         'prec': prec, 'rec': rec, 'metric_prc': metric_prc}
        
    return perf


def reduce_dimensionality(patient_embeddings: torch.Tensor,
                          label_embeddings: torch.Tensor,
                          ) -> np.ndarray:
    """ Reduce the dimensionality of high-dimensional concept embeddings
        - Patients and labels are concatenated for dimensionality reduction
        - Then, the reduced representations are returned separately
    """
    if REDUCED_DIMENSIONALITY is None:
        return patient_embeddings, label_embeddings
    embeddings = torch.cat((patient_embeddings, label_embeddings), dim=0)
    if DIMENSIONALITY_REDUCTION_ALGORITHM == 'pca':
        reduced = PCA().fit_transform(embeddings)[:, :REDUCED_DIMENSIONALITY]
    elif DIMENSIONALITY_REDUCTION_ALGORITHM == 'tsne':
        params = {'learning_rate': 'auto', 'init': 'pca'}
        reduced = TSNE(REDUCED_DIMENSIONALITY, **params).fit_transform(embeddings)
    reduced_patient_embeddings = reduced[:patient_embeddings.shape[0]]
    reduced_label_embeddings = reduced[patient_embeddings.shape[0]:]
    return reduced_patient_embeddings, reduced_label_embeddings
    

def get_labels_multi(model: torch.nn.Module,
                     pipeline: data.DataPipeline,
                     cat: str,
                     ) -> dict[list[int], torch.Tensor]:
    """ Generate embeddings for all possible tokens of a given category
        - Tokens are retrieved from the tokenizer vocabulary
        - The result is a dict of token indices and corresponding embeddings
    """
    vocab = pipeline.tokenizer.get_vocab()
    labels = [t for t in vocab if cat in t and cat != t]
    label_encodings = [pipeline.tokenizer.encode(t) for t in labels]
    label_embeddings = model.get_token_embeddings(label_encodings)
    return labels, label_embeddings


def get_labels_binary(model: torch.nn.Module,
                      pipeline: data.DataPipeline,
                      classes: list[str]
                      ) -> dict[list[str], torch.Tensor]:
    """ Generate embeddings for a given pair of classes
        - The result is a dict of token classes and corresponding embeddings
    """
    assert len(classes) == 2, 'Number of classes should be 2.'
    class_encodings = [pipeline.tokenizer.encode(c) for c in classes]
    class_embeddings = model.get_token_embeddings(class_encodings)
    return class_embeddings
    

def get_patient_embedding(model: torch.nn.Module,
                          sample: list[str],
                          pipeline: data.DataPipeline,
                          ) -> torch.Tensor:
    """ Generate a sequence embedding for a patient sample in which tokens that
        do not belong to a given category were removed
        - The result is a weighted average of all tokens embeddings
        - Weigths are proportional to token inverse frequency (in train dataset)
    """
    # This allows to compute performance for a random classifier
    if RANDOM_MODE:
        return 2 * (torch.rand((len(PARTIALS), 512)) - 0.5)  # last minute :D
    
    # Encode patient tokens and compute weights based on term frequencies
    encoded = [pipeline.tokenizer.encode(t) for t in sample]
    if isinstance(encoded[0], list):
        weights = [1 / pipeline.tokenizer.word_counts[t[0]] for t in encoded]
    else:
        weights = [1 / pipeline.tokenizer.word_counts[t] for t in encoded]
    
    # Compute patient embeddings for different levels of partial information
    fixed_enc = [encoded[n] for n, t in enumerate(sample) if 'DEM_' in t
                 and not any([s in t for s in BINARY_CLASSES])]
    fixed_wgt = [weights[n] for n, t in enumerate(sample) if 'DEM_' in t]
    timed_idx = [n for n, t in enumerate(sample) if 'DEM_' not in t]
    sentence_embeddings = []
    for partial in PARTIALS:
        
        # Get partial encodings and associated weights 
        partial_timed_idx = timed_idx[:int(len(timed_idx) * partial)]
        enc = fixed_enc + [encoded[i] for i in partial_timed_idx]
        wgt = fixed_wgt + [weights[i] for i in partial_timed_idx]
        
        # Apply eligibility trace if required and compute sentence embeddings
        if USE_TIME_WEIGHTS:
            time_wgt = TIME_WEIGHTS[-len(wgt):]
            wgt = [w * t for w, t in zip(wgt, time_wgt)]
        sentence_embeddings.append(model.get_sequence_embeddings(enc, wgt))
    
    # Return partial sentences stacked over a new dimension
    return torch.stack(sentence_embeddings, dim=0)
    

def generate_figure(multi_results: dict[str, list[float]],
                    binary_results: dict[str, float]
                    ) ->  plt.Figure:
    """ Generate a big figure that contains all plots of the prediction task
    """
    # Create main figure plot
    print(' - Plotting figure with all prediction results')
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    subfigs = fig.subfigures(3, 1, height_ratios=[2.5, 1, 1])

    # Multi category prediction subplot
    axs = subfigs[0].subplots(3, 5)
    for row, (cat, results) in enumerate(multi_results.items()):
        for col, (letter_lenient_match, result) in enumerate(results.items()):
            ax = axs[row, col]
            for i, (partial, perf) in enumerate(result.items()):

                params = {
                    'label': 'Partial: %s - AVG-PR: %.03f' %\
                        (partial, perf['metric_prc']),
                    'color': 'C%1i' % i,
                }
                ax.plot(perf['rec'], perf['prec'], **params)
            
            info = '%s letter(s) match' % letter_lenient_match
            ax.set_xlabel('Recall (%s) - %s' % (cat.split('_')[0], info),
                          fontsize='small',
                          labelpad=0.5)
            ax.set_ylabel('Precision (%s)' % cat.split('_')[0],
                          fontsize='small',
                          labelpad=0.5)
            ax.set_ylim(0.0, 1.0)
            ax.tick_params(axis='both', labelsize='x-small', pad=0.5)
            ax.legend(fontsize='x-small', labelspacing=0.5)
            ax.grid()
    
    # Binary prediction subplot (receiver-operating-characteristics curve)
    axs = subfigs[1].subplots(1, 3)
    for ax, (i, (cat, result)) in zip(axs, enumerate(binary_results.items())):
        for i, (partial, perf) in enumerate(result.items()):
            label = 'Partial: %s - AUROC: %.03f' % (partial, perf['metric_roc'])
            ax.plot(perf['fpr'], perf['tpr'], color='C%1i' % i, label=label)
        ax.plot((0, 1), (0, 1), '--', color='gray')
        ax.set_xlabel('FP rate (%s)' % cat, labelpad=0.5)
        ax.set_ylabel('TP rate (%s)' % cat, labelpad=0.5)
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(axis='both', labelsize='smaller', pad=0.5)
        ax.legend(fontsize='smaller')
        ax.grid()
    
    # Binary prediction subplot (precision-recall curve)
    axs = subfigs[2].subplots(1, 3)
    for ax, (i, (cat, result)) in zip(axs, enumerate(binary_results.items())):
        for i, (partial, perf) in enumerate(result.items()):
            label = 'Partial: %s - AVG-PR: %.03f' % (partial, perf['metric_prc'])
            ax.plot(perf['rec'], perf['prec'], color='C%1i' % i, label=label)
        ax.set_xlabel('Recall (%s)' % cat, labelpad=0.5)
        ax.set_ylabel('Precision (%s)' % cat, labelpad=0.5)
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(axis='both', labelsize='smaller', pad=0.5)
        ax.legend(fontsize='smaller')
        ax.grid()

    # Send figure to tensorboard
    return fig 


def log_figure_to_tensorboard(fig: plt.Figure,
                              fig_title: str,
                              logger: pl.loggers.tensorboard.TensorBoardLogger,
                              global_step: int,
                              ) -> None:
    """ Log any figure to tensorboard as an image stream
    """
    print(' - Saving figure and displaying it on tensorboard')
    temp_file_name = tempfile.NamedTemporaryFile(suffix='.png').name
    fig.savefig(temp_file_name, dpi=300, bbox_inches='tight')  
    image = np.asarray(Image.open(temp_file_name)).transpose(2, 0, 1)
    logger.experiment.add_image(fig_title, image, global_step=global_step)


def generate_csv(multi_perfs: dict[str, list[float]],
                 binary_perfs: dict[str, float],
                 logger: pl.loggers.tensorboard.TensorBoardLogger,
                 ) ->  plt.Figure:
    """ Generate a csv file that summarizes the results of the prediction task
    """
    # Initialize headers and rows to write
    heads = ['Category', 'Partial'] +\
            ['AUROC-' + s for s in ['1L', '2L', '3L', '4L', 'EM']] +\
            ['AUPRC-' + s for s in ['1L', '2L', '3L', '4L', 'EM']] +\
            ['AUROC-BI', 'AUPRC-BI']
    rows = []
    
    # Fill the rows with the results
    for (cat, multi_results), (_, binary_result) in\
        zip(multi_perfs.items(), binary_perfs.items()):
        new_rows = [[cat, str(p)] for p in PARTIALS]
        for metric in ['metric_roc', 'metric_prc']:
            for multi_result in multi_results.values():
                for new_row, perf in zip(new_rows, multi_result.values()):
                    new_row.append('%.03f' % perf[metric])
        for metric in ['metric_roc', 'metric_prc']:
            for new_row, perf in zip(new_rows, binary_result.values()):
                new_row.append('%.03f' % perf[metric])
        rows.extend(new_rows)
    
    # Save the results as a csv file in the correct logs directory
    csv_filepath = 'prediction_results_rd_%s_rc_%s.csv' %\
        (REDUCED_DIMENSIONALITY, RANDOM_MODE)
    save_path = os.path.join(logger.root_dir, csv_filepath)
    df = pd.DataFrame(rows, columns=heads)
    df.to_csv(save_path, index=False, header=heads)
    