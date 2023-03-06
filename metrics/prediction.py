import cProfile
import pstats
import pytorch_lightning as pl
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import data
import tempfile
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import (
    roc_curve,
    auc as auroc,
    precision_recall_curve as pr_curve,
    average_precision_score as auprc
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale


MAX_SAMPLES = 200  # np.inf  # used for debug if < np.inf
PARTIALS = [0.0, 0.1, 0.3, 0.6, 1.0]  # None
TOPKS = [1, 3, 10, 30]
N_MATCHES = [1, 2, 3, 4, 0]
MULTI_PARAMS = [{'topk': k, 'n_matches': m} for k in TOPKS for m in N_MATCHES]
MULTI_CATEGORIES = ['DIA_', 'PRO_', 'MED_']
BINARY_CLASSES = {
    'mortality': ['LBL_ALIVE', 'LBL_DEAD'],
    'readmission': ['LBL_AWAY', 'LBL_READM'],
    'length-of-stay': ['LBL_SHORT', 'LBL_LONG'],
}
BINARY_LEVELS = [t for v in BINARY_CLASSES.values() for t in v]
MULTI_PLOT_PARAMS = {
    k: {'lw': 2, 'color': 'C%s' % i, 'label': 'top-%s' % k}
    for i, k in enumerate(TOPKS)
}
BINARY_PLOT_PARAMS = {
    'width': 0.6,
    'color': ['%s' % (i / len(BINARY_CLASSES))
              for i, _ in enumerate(BINARY_CLASSES)],
}
THRESHOLDS = np.linspace(0.0, 1.0, 100)


def profile_it(fn):
    def profiled_fn(*args, **kwargs):
        with cProfile.Profile() as pr:
            fn(*args, **kwargs)
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME) 
        stats.dump_stats(filename='profiling.prof')
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
    multi_accuracy, binary_perf = {}, {}
    for cat in MULTI_CATEGORIES:
        multi_accuracy[cat] = compute_prediction_multi(model, pipeline, cat)
        exit()
    for k, v in BINARY_CLASSES.items():
        binary_perf[k] = compute_prediction_binary(model, pipeline, v)
    figure = generate_figure(multi_accuracy, binary_perf)
    log_figure_to_tensorboard(figure, 'prediction_metric', logger, global_step)
    

@profile_it
def compute_prediction_multi(model: torch.nn.Module,
                             pipeline: data.DataPipeline,
                             cat: str,
                             ) -> float:
    """ Compute top-k accuracy for a multi-label classification task
        - A model embeds sequences of tokens in which all tokens belonging to a
        category are removed
        - Sequence embeddings are compared to label token embeddings
        - Top-k accuracy is computed based on cosine similarity
    """
    # Retrieve labels and initialize
    unk_encoding = pipeline.tokenizer.encode('[UNK]')
    label_encodings, label_embeddings = get_labels_multi(model, pipeline, cat)
    to_remove = pipeline.run_params['ngrams_to_remove']
    dp = data.JsonReader(pipeline.data_fulldir, 'valid')
    dp = data.TokenFilter(dp, to_remove=to_remove, to_split=[cat])
    
    # Compute patient embeddings and store gold labels
    input_embeddings, golds = [], []
    loop = tqdm(dp, desc=' - Embedding patients (no %s tokens)' % cat)
    for n, (sample, gold) in enumerate(loop):
        if n > MAX_SAMPLES: break
        if len(gold) == 0: continue
        input_embeddings.append(get_input_embedding(model, sample, pipeline))
        encoded_gold = [pipeline.tokenizer.encode(g) for g in gold]
        golds.append([e for e in encoded_gold if e != unk_encoding])
    
    # # TODO: Train a linear regression with golds and embeddings?
    
    # Compute similarities between patient embeddings and tokens of the category
    print(' - Comparing patient embeddings to %s tokens' % cat)
    input_embeddings = torch.cat(input_embeddings, dim=0)
    similarities = cosine_similarity(input_embeddings, label_embeddings)
    class_eye = np.eye(len(label_encodings))  # used to speed-up computations
    
    # Compute top-k accuracy for exact and different n-char lenient matches
    # hits, trials = [0 for _ in MULTI_PARAMS], 0
    # topk_maxs = np.argsort(similarities, axis=-1)[:, :-max(TOPKS):-1]
    # for topk_max, gold in tqdm(zip(topk_maxs, golds), desc='test'):
    #     for i, p in enumerate(MULTI_PARAMS):  # different task settings
    #         topk = [label_encodings[k] for k in topk_max[:p['topk']]]
    #         hits[i] += compute_hit_multi(topk, gold, pipeline, p['n_matches'])
    #     trials += 1
    # return [h / trials for h in hits]
    perf = {}
    for i, partial in enumerate(PARTIALS):
        TP, FP, FN = np.zeros((3, len(THRESHOLDS)))
        for similarity, gold in tqdm(zip(similarities[i::len(PARTIALS)], golds),
                                     desc='test'):
            prob = minmax_scale(similarity)
            gold_ids = [label_encodings.index(g) for g in gold]
            sample_tp, sample_fp, sample_fn = compute_tp_fp_fn(prob,
                                                               gold_ids,
                                                               class_eye)
            TP += sample_tp; FP += sample_fp; FN += sample_fn
        precision = [tp / (tp + fp) for tp, fp in zip(TP, FP)]
        recall = [tp / (tp + fn) for tp, fn in zip(TP, FN)]
        perf[partial] = {'prec': precision, 'rec': recall}
                    

def single_hot(label_list, class_eye):
    """ Like a one-hot matrix, summed over label dimension
    """
    return class_eye[label_list].sum(axis=0)


def compute_tp_fp_fn(prob, gold_ids, class_eye):
    single_hot_gold = single_hot(gold_ids, class_eye)[np.newaxis, :]
    preds = np.tile(prob, (len(THRESHOLDS), 1)) > THRESHOLDS[:, np.newaxis]
    tp = (preds * single_hot_gold).sum(axis=1)
    fp = (preds * (1 - single_hot_gold)).sum(axis=1)
    fn = len(gold_ids) - tp
    return tp, fp, fn


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
                [t for t in BINARY_LEVELS if t not in classes]
    dp = data.JsonReader(pipeline.data_fulldir, 'test')
    dp = data.TokenFilter(dp, to_remove=to_remove, to_split=classes)
    
    # Compute patient embeddings and store gold labels
    input_embeddings, golds = [], []
    loop = tqdm(enumerate(dp),
                desc=' - Embedding patients (no %s tokens)' % classes)
    for n, (sample, gold) in loop:
        if n > MAX_SAMPLES: break
        input_embeddings.append(get_input_embedding(model, sample, pipeline))
        golds.append(gold)
    
    # Compute similarities between patient embeddings and class tokens
    print(' - Comparing patient embeddings to %s tokens' % classes)
    input_embeddings = torch.cat(input_embeddings, dim=0)
    similarities = cosine_similarity(input_embeddings, class_embeddings)
    
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
        metric_auroc = auroc(fpr, tpr)
        metric_auprc = auprc(trues, probs, pos_label=1)
        perf[partial] = {'fpr': fpr, 'tpr': tpr, 'auroc': metric_auroc,
                         'prec': prec, 'rec': rec, 'auprc': metric_auprc}
    
    return perf


def get_labels_multi(model: torch.nn.Module,
                     pipeline: data.DataPipeline,
                     cat: str,
                     ) -> dict[list[int], torch.Tensor]:
    """ Generate embeddings for all possible tokens of a given category
        - Tokens are retrieved from the tokenizer vocabulary
        - The result is a dict of token indices and corresponding embeddings
    """
    vocab = pipeline.tokenizer.get_vocab()
    label_encodings = [pipeline.tokenizer.encode(t)
                       for t in vocab if cat in t and cat != t]
    label_embeddings = model.get_token_embeddings(label_encodings)
    return label_encodings, label_embeddings


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
    return classes, class_embeddings
    

def get_input_embedding(model: torch.nn.Module,
                        sample: list[str],
                        pipeline: data.DataPipeline,
                        mode: str='partial',  # 'full', 'partial'
                        ) -> torch.Tensor:
    """ Generate a sequence embedding for a patient sample in which tokens that
        do not belong to a given category were removed
        - The result is a weighted average of all tokens embeddings
        - Weigths are proportional to token inverse frequency (in train dataset)
    """
    encoded = [pipeline.tokenizer.encode(t) for t in sample]
    if isinstance(encoded[0], list):
        weights = [1 / pipeline.tokenizer.word_counts[t[0]] for t in encoded]
    else:
        weights = [1 / pipeline.tokenizer.word_counts[t] for t in encoded]
    if mode == 'full':
        return model.get_sequence_embeddings(encoded, weights)
    else:
        fixed_enc = [encoded[n] for n, t in enumerate(sample)
                     if 'DEM_' in t and not any([s in t for s in BINARY_CLASSES])]
        fixed_wgt = [weights[n] for n, t in enumerate(sample) if 'DEM_' in t]
        timed_idx = [n for n, t in enumerate(sample) if 'DEM_' not in t]
        sentence_embeddings = []
        for partial in PARTIALS:
            partial_timed_idx = timed_idx[:int(len(timed_idx) * partial)]
            enc = fixed_enc + [encoded[i] for i in partial_timed_idx]
            wgt = fixed_wgt + [weights[i] for i in partial_timed_idx]
            sentence_embeddings.append(model.get_sequence_embeddings(enc, wgt))
        return torch.stack(sentence_embeddings, dim=0)
    

def compute_hit_multi(topk_indices: np.ndarray,
                      gold_indices: list,
                      pipeline: data.DataPipeline,
                      n_matches: int,
                      ) -> int:
    """ Compute whether any label of a patient's sample belongs to the topk
        neighbours of its embedded representation, using cosine similarity
        - The result is cast to int, so that hits can be summed afterwards
    """
    if n_matches > 0:  # lenient match (only requiring n first chars matching)
        if isinstance(gold_indices[0], list):
            gold_indices = [g[0] for g in gold_indices]
            topk_indices = [t[0] for t in topk_indices]
        gold_indices = [pipeline.tokenizer.decode(g) for g in gold_indices]
        topk_indices = [pipeline.tokenizer.decode(t) for t in topk_indices]
        gold_indices = [g.split('_')[-1][:n_matches] for g in gold_indices]
        topk_indices = [t.split('_')[-1][:n_matches] for t in topk_indices]
    return int(any([g in topk_indices for g in gold_indices]))


def generate_figure(multi_task_results: dict[str, list[float]],
                    binary_task_results: dict[str, float]
                    ) ->  matplotlib.figure.Figure:
    """ Generate a big figure that contains all plots of the prediction task
    """
    # Multi category prediction plot
    fig = plt.figure(figsize=(15, 10))
    # x_axis = range(len(N_MATCHES))
    # x_tick_labels = [str(n) if n > 0 else 'Exact' for n in N_MATCHES]
    # for i, (cat, accuracies) in enumerate(multi_task_results.items()):
    #     ax = plt.subplot(3, 3, i + 1)
    #     for k in TOPKS:
    #         matching_indices = [[p['topk'] == k and p['n_matches'] == n
    #                             for p in MULTI_PARAMS].index(True)
    #                             for n in N_MATCHES]
    #         accuracies_to_plot = [accuracies[i] for i in matching_indices]
    #         ax.plot(x_axis, accuracies_to_plot, '-o', **MULTI_PLOT_PARAMS[k])
    #     ax.set_xticks(x_axis)
    #     ax.set_xticklabels(x_tick_labels)
    #     ax.tick_params(labelsize=10)
    #     ax.set_xlabel('Character match (%s)' % cat[:-1], fontsize=12)
    #     ax.set_ylabel('Top-k accuracy', fontsize=12)
    #     ax.set_ylim([-0.05, 1.05])
    #     ax.legend(loc='lower left', ncols=len(TOPKS) // 2, fontsize=10)
    #     ax.grid()
    
    # Binary prediction plot (auroc)
    for i, (cat, result) in enumerate(binary_task_results.items()):
        ax = plt.subplot(3, 3, 3 + i + 1)
        for i, (partial, perf) in enumerate(result.items()):
            label = 'Partial: %s - AUROC: %.02f' % (partial, perf['auroc'])
            ax.plot(perf['fpr'], perf['tpr'], color='C%1i' % i, label=label)
        ax.plot((0, 1), (0, 1), '--', color='gray')
        ax.set_xlabel('False positive rate (%s)' % cat, fontsize=12)
        ax.set_ylabel('True postive rate (%s)' % cat, fontsize=12)
        ax.legend()
        ax.grid()
    
    # Binary prediction plot (auprc)
    for i, (cat, result) in enumerate(binary_task_results.items()):
        ax = plt.subplot(3, 3, 6 + i + 1)
        for i, (partial, perf) in enumerate(result.items()):
            label = 'Partial: %s - AUPRC: %.02f' % (partial, perf['auprc'])
            ax.plot(perf['rec'], perf['prec'], color='C%1i' % i, label=label)
        ax.set_xlabel('Recall (%s)' % cat, fontsize=12)
        ax.set_ylabel('Precision (%s)' % cat, fontsize=12)
        ax.legend()
        ax.grid()

    # Send figure to tensorboard
    return fig 


def log_figure_to_tensorboard(fig: matplotlib.figure.Figure,
                              fig_title: str,
                              logger: pl.loggers.tensorboard.TensorBoardLogger,
                              global_step: int,
                              ) -> None:
    """ Log any figure to tensorboard as an image stream
    """
    temp_file_name = tempfile.NamedTemporaryFile(suffix='.png').name
    fig.savefig(temp_file_name, dpi=150, bbox_inches='tight')  
    image = np.asarray(Image.open(temp_file_name)).transpose(2, 0, 1)
    logger.experiment.add_image(fig_title, image, global_step=global_step)
