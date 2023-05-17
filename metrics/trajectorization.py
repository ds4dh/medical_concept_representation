import cProfile
import pstats
import pytorch_lightning as pl
import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import data
import tempfile
from torchdata.datapipes.iter import Shuffler
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


RANDOM_MODE = True
MAX_SAMPLES = 100_000  # np.inf  # used for debug if < np.inf
LENIENT_LETTER_MATCHES = [1, 2, 3, 4, 'Exact']
MULTI_CATEGORIES = ['DIA_', 'PRO_', 'MED_']
BINARY_CLASSES = {
    'mortality': ['LBL_ALIVE', 'LBL_DEAD'],
    'readmission': ['LBL_AWAY', 'LBL_READM'],
    'length-of-stay': ['LBL_SHORT', 'LBL_LONG'],
}
ALL_BINARY_LEVELS = [t for v in BINARY_CLASSES.values() for t in v]
DIMENSIONALITY_REDUCTION_ALGORITHM = 'pca'  # 'tsne', 'pca'
REDUCED_DIMENSIONALITY = None  # if None, no dimensionality reduction performed
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


def trajectorization_task(model: torch.nn.Module,
                          pipeline: data.DataPipeline,
                          logger: pl.loggers.tensorboard.TensorBoardLogger,
                          global_step: int,
                          ) -> None:
    """ Compute prediction task metric for a trained model, where target tokens
        are separated from the patient code sequence, and patient embeddings
        predict target tokens in an unsupervised way, using cosine similarity
    """
    print('\nProceeding with prediction testing metric')
    multi_perf = {}
    for cat in MULTI_CATEGORIES:
        multi_perf[cat] = compute_trajectory_prediction(model, pipeline, cat)
    
    generate_csv(multi_perf, logger)
    fig = generate_figure(multi_perf)
    fig_title = 'trajectorization_metric_rd_%s_rc_%s' %\
        (REDUCED_DIMENSIONALITY, RANDOM_MODE)
    log_fig_to_tensorboard(fig, fig_title, logger, global_step)
    

def compute_trajectory_prediction(model: torch.nn.Module,
                                  pipeline: data.DataPipeline,
                                  cat: str,
                                  ) -> dict[str, dict[str, float]]:
    """ Compute top-k accuracy for a multi-label classification task
        - A model embeds sequences of tokens in which all tokens belonging to a
        category are removed
        - Sequence embeddings are compared to label token embeddings
        - Top-k accuracy is computed based on cosine similarity
    """
    # Retrieve data and initialize prediction sample-label pairs
    unk_encoding = pipeline.tokenizer.encode('[UNK]')
    labels, label_embeddings = get_labels_multi(model, pipeline, cat)
    to_remove = pipeline.run_params['ngrams_to_remove'] + ALL_BINARY_LEVELS
    dp = data.JsonReader(pipeline.data_fulldir, 'valid')
    dp = data.TokenFilter(dp, to_remove=to_remove)  # to_split=[cat])
    if cat == 'DIA_':
        dp = generate_first_diag_pred_pairs(dp)
    else:
        dp = generate_traj_pred_pairs(dp, lambda token: cat in token)
        dp = Shuffler(dp)  # to avoid taking everything from the first samples
    
    # Compute patient embeddings and store gold labels
    patient_embeddings, golds = [], []
    loop = tqdm(dp, desc=' - Patients embedded - no %s token' % cat, leave=False)
    for n, (sample, gold) in enumerate(loop):
        if n >= MAX_SAMPLES: break
        if pipeline.tokenizer.encode(gold) == unk_encoding: continue
        patient_embeddings.append(get_patient_embedding(model, sample, pipeline))
        golds.append([gold])
        
    # Compute prediction scores based on patient-label embedding similarities
    print(' - Comparing patient embeddings to %s tokens' % cat)
    patient_embeddings = torch.stack(patient_embeddings, dim=0)
    # patient_embeddings, label_embeddings =\
    #     reduce_dimensionality(patient_embeddings, label_embeddings)
    scores = cosine_similarity(patient_embeddings, label_embeddings)
    
    # Return performance for different lenient settings, using the scores
    return {
        llm: compute_micro_performances(scores, golds, labels, llm)
        for llm in LENIENT_LETTER_MATCHES
    }


def generate_first_diag_pred_pairs(dp):
    """ ...
    """
    for sample in dp:
        if any(['DIA_' in token for token in sample]):
            input_data = [token for token in sample if 'DIA_' not in token]
            label = next(token for token in sample if 'DIA_' in token)
            yield (input_data, label)
            

def generate_traj_pred_pairs(dp, condition):
    """ Generate all possible sample-label pairs of trajectory prediction for
        each sample, using any category of predicted tokens as labels
        E.g.: [A1, A2, B1, A3, B2] -> [([A1, A2], B1), ([A1, A2, B1, A3], B2)]
    """
    for sample in dp:
        for j, elem in enumerate(sample):
            if condition(elem):
                yield (sample[:j], elem)  # (input_data, label)
    

def compute_micro_performances(scores: np.ndarray,
                               golds: list[set[str]],
                               labels: list[str],
                               n_matched_letters: int,
                               ) -> dict[str, Union[list[float], float]]:
    """ ...
    """
    lenient_scores, lenient_golds, lenient_labels =\
          lenient_match_collapse(scores, golds, labels, n_matched_letters)
    one_hotter = MultiLabelBinarizer(classes=lenient_labels)
    onehot_lenient_golds = one_hotter.fit_transform(lenient_golds)
    prec, rec, _ = pr_curve(onehot_lenient_golds.ravel(), lenient_scores.ravel())
    fpr, tpr, _ = roc_curve(onehot_lenient_golds.ravel(), lenient_scores.ravel())
    metric_prc = avg_prec(onehot_lenient_golds, lenient_scores, average='micro')
    metric_roc = auc(fpr, tpr)

    return {'prec': prec, 'rec': rec, 'metric_prc': metric_prc,
            'fpr': fpr, 'tpr': tpr, 'metric_roc': metric_roc}


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
        
        # TODO: HAVE THE {None} case handled (for unkown tokens of the test set)
        key = label.split('_')[-1][:n_matched_letters]
        score_map[key].append(i)
        gold_map[label] = key
        
    new_scores = [scores[:, cols].sum(axis=1) for cols in score_map.values()]
    new_scores = np.stack(new_scores, axis=-1)
    new_golds = [set([gold_map.get(key) for key in g]) for g in golds]
    new_labels = list(score_map.keys())
    
    return new_scores, new_golds, new_labels


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
        return 2 * (torch.rand(512) - 0.5)  # last minute :D
    
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
    

def generate_figure(multi_results: dict[str, list[float]]) ->  plt.Figure:
    """ Generate a big figure that contains all plots of the prediction task
    """
    # Multi category prediction subplot
    print(' - Plotting figure with all prediction results')    
    fig, axs = plt.subplots(3, 5, figsize=(8, 10))
    for row, (cat, results) in enumerate(multi_results.items()):
        for col, (letter_lenient_match, result) in enumerate(results.items()):
            ax = axs[row, col]

            params = {
                'label': 'AVG-PR: %.03f' % result['metric_prc'],
                'color': 'C0',  # 'C%1i' % i
            }
            ax.plot(result['rec'], result['prec'], **params)
            
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
    
    # Send figure to tensorboard
    return fig


def log_fig_to_tensorboard(fig: plt.Figure,
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


def reduce_dimensionality(patient_embeddings: torch.Tensor,
                          label_embeddings: torch.Tensor,
                          ) -> np.ndarray:
    """ Reduce the dimensionality of high-dimensional concept embeddings
        - Leave embeddings untouched if reduced_dimensionality is None
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


def generate_csv(multi_perfs: dict[str, list[float]],
                 logger: pl.loggers.tensorboard.TensorBoardLogger,
                 ) ->  plt.Figure:
    """ Generate a csv file that summarizes the results of the prediction task
    """
    # Initialize headers and rows to write
    heads = ['Category'] +\
            ['AUROC-' + s for s in ['1L', '2L', '3L', '4L', 'EM']] +\
            ['AUPRC-' + s for s in ['1L', '2L', '3L', '4L', 'EM']]
    rows = []

    # Fill the rows with the results
    for cat, multi_results in multi_perfs.items():
        new_row = [cat]
        for metric in ['metric_roc', 'metric_prc']:
            for multi_result in multi_results.values():  # loop over 1L, 2L, ...
                new_row.append('%.03f' % multi_result[metric])
        rows.append(new_row)

    # Save the results as a csv file in the correct logs directory
    csv_filepath = 'trajectorization_results_rd_%s_rc_%s.csv' %\
        (REDUCED_DIMENSIONALITY, RANDOM_MODE)
    save_path = os.path.join(logger.root_dir, csv_filepath)
    df = pd.DataFrame(rows, columns=heads)
    df.to_csv(save_path, index=False, header=heads)
