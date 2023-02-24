import pytorch_lightning as pl
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import data
import tempfile
from tqdm import tqdm
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

MAX_SAMPLES = 100  # np.inf  # used for debug if < np.inf
TOPKS = [1, 3, 10, 30]
N_MATCHES = [1, 2, 3, 4, 0]
MULTI_PARAMS = [{'topk': k, 'n_matches': m} for k in TOPKS for m in N_MATCHES]
MULTI_CATEGORIES = ['DIA_', 'PRO_', 'MED_']
BINARY_CLASSES = {
    'mortality': ['DEM_ALIVE', 'DEM_DEAD'],
    'readmission': ['DEM_AWAY', 'DEM_READM'],
    'length_of_stay': ['DEM_SHORT', 'DEM_LONG'],
}
MULTI_PLOT_PARAMS = {
    k: {'lw': 2, 'color': 'C%s' % i, 'label': 'top-%s' % k}
    for i, k in enumerate(TOPKS)
}
BINARY_PLOT_PARAMS = {
    'width': 0.6,
    'color': ['%s' % (i / len(BINARY_CLASSES))
              for i, _ in enumerate(BINARY_CLASSES)],
}


def prediction_task(model: torch.nn.Module,
                    pipeline: data.DataPipeline,
                    logger: pl.loggers.tensorboard.TensorBoardLogger,
                    global_step: int,
                    ) -> None:
    """ Compute prediction task metric for a trained model, where target tokens
        are separated from the patient code sequence, and patient embeddings
        predict target tokens in an unsupervised way, using cosine similarity 
    """
    multi_accuracy, binary_accuracy = {}, {}
    for cat in MULTI_CATEGORIES:
        multi_accuracy[cat] = compute_prediction_multi(model, pipeline, cat)
    for k, v in BINARY_CLASSES.items():
        binary_accuracy[k] = compute_prediction_binary(model, pipeline, v)
    figure = generate_figure(multi_accuracy, binary_accuracy)
    log_figure_to_tensorboard(figure, 'prediction_metric', logger, global_step)
        

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
    label_encodings, label_embeddings = get_labels_multi(model, pipeline, cat)
    to_remove = pipeline.run_params['ngrams_to_remove']
    dp = data.JsonReader(pipeline.data_fulldir, 'test')
    dp = data.TokenFilter(dp, to_remove=to_remove, to_split=[cat])
    
    # Compute patient embeddings and store gold labels
    input_embeddings, golds = [], []
    loop = tqdm(dp, desc=' - Embedding patients (without %s tokens)' % cat)
    for n, (sample, gold) in enumerate(loop):
        if n > MAX_SAMPLES: break
        if len(gold) == 0: continue
        golds.append([pipeline.tokenizer.encode(g) for g in gold])
        input_embeddings.append(get_input_embedding(model, sample, pipeline))
    
    # # TODO: Train a linear regression with golds and embeddings
    # import pdb; pdb.set_trace()
    # exit()  # in case
    
    # Compute similarities between patient embeddings and tokens of the category
    print(' - Computing similarities between patient embeddings and tokens')
    input_embeddings = torch.stack(input_embeddings, dim=0)
    similarities = cosine_similarity(input_embeddings, label_embeddings)
    topk_maxs = np.argsort(similarities, axis=-1)[:, :-max(TOPKS):-1]
    
    # Compute top-k accuracy for exact and different n-char lenient matches
    hits, trials = [0 for _ in MULTI_PARAMS], 0
    loop = tqdm(zip(topk_maxs, golds), desc=' - Predicting %s' % cat)
    for topk_max, gold in loop:
        for i, p in enumerate(MULTI_PARAMS):  # different task settings
            topk = [label_encodings[k] for k in topk_max[:p['topk']]]
            hits[i] += compute_hit_multi(topk, gold, pipeline, p['n_matches'])
        trials += 1
    return [h / trials for h in hits]


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
    class_encodings, class_embeddings = get_labels_binary(model, pipeline, classes)
    to_remove = pipeline.run_params['ngrams_to_remove']
    dp = data.JsonReader(pipeline.data_fulldir, 'test')
    dp = data.TokenFilter(dp, to_remove=to_remove, to_split=classes)
    
    # Compute patient embeddings and store gold labels
    input_embeddings, golds = [], []
    loop = tqdm(dp, desc=' - Embedding patients, removing %s' % classes)
    for n, (sample, gold) in enumerate(loop):
        if n > MAX_SAMPLES: break
        input_embeddings.append(get_input_embedding(model, sample, pipeline))
        golds.append(gold)
    
    # Compute similarities between patient embeddings and class tokens
    print(' - Computing similarities between patient embeddings and classes')
    input_embeddings = torch.stack(input_embeddings, dim=0)
    similarities = cosine_similarity(input_embeddings, class_embeddings)
    
    # Return accuracy for the prediction task given by the class labels
    hits = []
    loop = tqdm(zip(similarities, golds), desc=' - Predicting %s' % classes)
    for similarity, gold in loop:
        hits.append(class_encodings[np.argmax(similarity)] in gold)
    return sum(hits) / len(golds)


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
    return model.get_sequence_embeddings(encoded, weights)


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


def generate_figure(multi_accuracy_results: dict[str, list[float]],
                    binary_accuracy_results: dict[str, float]
                    ) ->  matplotlib.figure.Figure:
    """ Generate a big figure that contains all plots of the prediction task
    """
    # Multi category prediction plot
    n_graphs = len(multi_accuracy_results) + 1
    fig = plt.figure(figsize=(n_graphs * 5, 4))
    x_axis = range(len(N_MATCHES))
    x_tick_labels = [str(n) if n > 0 else 'Exact' for n in N_MATCHES]
    for i, (cat, accuracies) in enumerate(multi_accuracy_results.items()):
        ax = plt.subplot(1, n_graphs, i + 1)
        for k in TOPKS:
            matching_indices = [[p['topk'] == k and p['n_matches'] == n
                                for p in MULTI_PARAMS].index(True)
                                for n in N_MATCHES]
            accuracies_to_plot = [accuracies[i] for i in matching_indices]
            ax.plot(x_axis, accuracies_to_plot, '-o', **MULTI_PLOT_PARAMS[k])
        ax.set_xticks(x_axis)
        ax.set_xticklabels(x_tick_labels)
        ax.tick_params(labelsize=10)
        ax.set_xlabel('Character match (%s)' % cat[:-1], fontsize=12)
        ax.set_ylabel('Top-k accuracy', fontsize=12)
        ax.set_ylim([-0.05, 1.05])
        ax.legend(loc='lower left', ncols=len(TOPKS) // 2, fontsize=10)
        ax.grid()
    
    # Binary prediction plot
    x_axis = [n + 0.5 for n, _ in enumerate(binary_accuracy_results.keys())]
    labels = [' '.join(l.capitalize().split('_'))
              for l in binary_accuracy_results.keys()]
    to_plot = binary_accuracy_results.values()
    ax = plt.subplot(1, n_graphs, n_graphs)
    ax.bar(labels, to_plot, **BINARY_PLOT_PARAMS)
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Prediction task', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.grid()
    ax.set_axisbelow(True)
    
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
