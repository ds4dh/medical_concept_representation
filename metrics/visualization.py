import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tempfile
import data
import pytorch_lightning as pl
import itertools
from PIL import Image
from adjustText import adjust_text
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


CATEGORY_SUBLEVELS_FREQUENT = {
    'DIA_': [('C', 'D0', 'D1', 'D2', 'D3', 'D4'), ('D5', 'D6', 'D7', 'D8', 'D9'),
             ('F',), ('G',), ('H0', 'H1', 'H2', 'H3', 'H4', 'H5'),
             ('H6', 'H7', 'H8', 'H9'), ('I',), ('J',), ('K',),
             ('L',), ('M',), ('N',), ('O',), ('S',), ('T',), ('Z',)],
    'PRO_': [('00',), ('04',), ('0B',), ('0D',), ('0F',), ('0H',), ('0J',),
             ('0Q',), ('0S',), ('0T',), ('0U',), ('0W',), ('3',), ('B',)],
    'MED_': [('A',), ('B',), ('C',), ('D',), ('G',), ('J',), ('L',), ('N',),
             ('R',), ('S',)],
}
CATEGORY_SUBLEVELS_OFFICIAL = {
    'DIA_': [('A', 'B',), ('C', 'D0', 'D1', 'D2', 'D3', 'D4',),
             ('D5', 'D6', 'D7', 'D8', 'D9',), ('E',), ('F',), ('G',),
             ('H0', 'H1', 'H2', 'H3', 'H4', 'H5',), ('H6', 'H7', 'H8', 'H9',),
             ('I',), ('J',), ('K',), ('L',), ('M',), ('N',), ('O',), ('P',),
             ('Q',), ('R',), ('S', 'T',), ('V', 'W', 'X', 'Y',), ('Z',), ('U',)],
    'PRO_': [('0',), ('1',), ('2',), ('3',), ('4',), ('5',), ('6',), ('7',),
             ('8',), ('9',), ('B',), ('C',), ('D',), ('F',), ('G',), ('H',), ('X',)],
    'MED_': [('A',), ('B',), ('C',), ('D',), ('G',), ('H',), ('J',), ('L',),
             ('M',), ('N',), ('P',), ('R',), ('S',), ('V',)],
}
CATEGORY_SUBLEVELS_FREQUENT_NEW = {
    # 'DIA_': [('C', 'D0', 'D1', 'D2', 'D3', 'D4'), ('D5', 'D6', 'D7', 'D8', 'D9'),
    #          ('F',), ('G',), ('H0', 'H1', 'H2', 'H3', 'H4', 'H5'),
    #          ('H6', 'H7', 'H8', 'H9'), ('I',), ('J',), ('K',),
    #          ('L',), ('M',), ('N',), ('O',), ('S',), ('T',), ('Z',)],
    'PRO_': [('00', '01'), ('03', '04'), ('05', '06'), ('0B',),
             ('0D',), ('0F',), ('0H',), ('0J',), ('0P', '0Q', '0R', '0S',),
             ('0T', '0U', '0V'), ('0W',), ('0X', '0Y'), ('3',), ('B',)],
    # 'MED_': [('A',), ('B',), ('C',), ('D',), ('G',), ('J',), ('L',), ('N',),
    #          ('R',), ('S',)],
}
CATEGORY_SUBLEVELS = CATEGORY_SUBLEVELS_FREQUENT_NEW
DIMENSIONALITY_REDUCTION_ALGORITHM = 'tsne'  # 'pca', 'tsne'
assert DIMENSIONALITY_REDUCTION_ALGORITHM in ('pca', 'tsne'),\
    'Invalid algorithm for dimensionality reduction [pca, tsne]'
REDUCED_DIM = 2  # 2, 3
assert REDUCED_DIM in [2, 3], 'Invalid reduced dimensionality [2, 3]'
FIG_SIZE = (14, 5)
SMALL_TEXT_SIZE = 14
BIG_TEXT_SIZE = 18
N_ANNOTATED_SAMPLES = 100  # per category
ALL_COLORS = list(plt.cm.tab20(np.arange(20)[0::2])) + ['#333333', '#ffffff'] +\
             list(plt.cm.tab20(np.arange(20)[1::2]))
LEGEND_PARAMS = {
    'loc': 'upper center',
    'bbox_to_anchor': (0.5, -0.05),
    'fancybox': True,
    'shadow': True,
    'fontsize': 10  # BASE_TEXT_SIZE
}
SCATTER_PARAMS =  {
    'marker': 'o',
    's': 20,
    'linewidths': 0.5,
    'edgecolors': 'k'
}
PLOT_CAT_MAP = {
    'DIA': 'ICD10-CM code',
    'PRO': 'ICD10-PCS code',
    'MED': 'ATC code',
}


def visualization_task(model: torch.nn.Module,
                       pipeline: data.DataPipeline,
                       logger: pl.loggers.tensorboard.TensorBoardLogger,
                       global_step: int,
                       ) -> None:
    """ Reduce the dimensionality of concept embeddings for different categories
        and log a scatter plot of the low-dimensional data to tensorboard
    """    
    print('\nProceeding with prediction testing metric')
    fig = plt.figure(figsize=FIG_SIZE)
    for subplot_idx, category in enumerate(CATEGORY_SUBLEVELS.keys()):
        print(' - Reducing dimensionality and visualizing %s tokens' % category)
        token_info = get_token_info(model, pipeline.tokenizer, category)
        reduced = compute_reduced_representation(token_info['embedded'])
        partitions = compute_partitions(token_info['labels'])
        delta_r_raw_dim = rate_reduction(token_info['embedded'], partitions, normalize=False)
        delta_r_low_dim = rate_reduction(reduced, partitions, normalize=False)
        delta_rs = (delta_r_raw_dim, delta_r_low_dim)

        # # debug to check my computations of rate reduction are correct
        # delta_r_raw_dim_ma = rate_reduction_ma(token_info['embedded'], partitions, normalize=False)
        # delta_r_low_dim_ma = rate_reduction_ma(reduced, partitions, normalize=False)
        # delta_rs_ma = (delta_r_raw_dim_ma, delta_r_low_dim_ma)
        # print('***RATE REDUCTION CHECK')
        # print(category)
        # print(delta_rs)
        # print(delta_rs_ma)
        # print('***RATE REDUCTION CHECK')
        # # debug to check my computations of rate reduction are correct
        
        plot_reduced_data(reduced, fig, token_info, category, subplot_idx, delta_rs)
    log_figure_to_tensorboard(fig, 'visualization_metric', logger, global_step)


def plot_reduced_data(reduced_data: np.ndarray,
                      fig: matplotlib.figure.Figure,
                      token_info: dict,
                      category: str,
                      subplot_idx: int,
                      delta_rs: tuple[float, float],
                      ) -> None:
    """ Plot data of reduced dimensionality to a 2d or 3d scatter plot
    """
    # Prepare subfigure and title
    data_dim = reduced_data.shape[-1]
    plot_cat = PLOT_CAT_MAP[category.split('_')[0]]
    plot_title = '%s embeddings\n%sd proj. (%s) - deltaR = %.2f (%.2f)' %\
        (plot_cat, data_dim, DIMENSIONALITY_REDUCTION_ALGORITHM, *delta_rs)
    kwargs = {} if data_dim <= 2 else {'projection': '3d'}
    ax = fig.add_subplot(1, 3, subplot_idx + 1, **kwargs)
    ax.set_title(plot_title, fontsize=SMALL_TEXT_SIZE)
    
    # Plot data of reduced dimensionality
    unique_labels = sorted(list(set(token_info['labels'])))
    unique_colors = ALL_COLORS[:len(unique_labels)]
    label_array = np.empty(len(token_info['labels']), dtype=object)
    label_array[:] = token_info['labels']
    for label, color in zip(unique_labels, unique_colors):
        data = reduced_data[[l == label for l in label_array]]
        data = [data[:, i] for i in range(data.shape[-1])]
        label = label[0] if len(label) == 1 else '-'.join([label[0], label[-1]])
        ax.scatter(*data, **SCATTER_PARAMS, color=color, label=label)
    
    # Add text annotation around some data points
    if N_ANNOTATED_SAMPLES > 0:
        loop = list(zip(reduced_data, token_info['tokens']))
        np.random.shuffle(loop)
        texts = []
        for d, t in loop:  # [:N_ANNOTATED_SAMPLES]:
            if 'DIA_T2' in t or 'DIA_T30' in t or 'DIA_T31' in t or 'DIA_T32' in t or 'DIA_T33' in t or 'DIA_T34' in t:
                texts.append(ax.text(d[0], d[1], t, fontsize='xx-small'))
        arrowprops = dict(arrowstyle='->', color='k', lw=0.5)
        adjust_text(texts, ax=ax, min_arrow_len=5, arrowprops=arrowprops)
    
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
    
    
def log_figure_to_tensorboard(fig: matplotlib.figure.Figure,
                              fig_title: str,
                              logger: pl.loggers.tensorboard.TensorBoardLogger,
                              global_step: int
                              ) -> None:
    """ Log the visualization plot to tensorboard as an image stream
    """
    temp_file_name = tempfile.NamedTemporaryFile(suffix='.png').name
    fig.savefig(temp_file_name, dpi=300, bbox_inches='tight')  
    image = np.asarray(Image.open(temp_file_name)).transpose(2, 0, 1)
    logger.experiment.add_image(fig_title, image, global_step=global_step)
    

def compute_reduced_representation(embeddings: np.ndarray) -> np.ndarray:
    """ Reduce the dimensionality of high-dimensional concept embeddings
    """
    if DIMENSIONALITY_REDUCTION_ALGORITHM == 'pca':
        return PCA().fit_transform(embeddings)[:, :REDUCED_DIM]
    elif DIMENSIONALITY_REDUCTION_ALGORITHM == 'tsne':
        # params = {'learning_rate': 'auto', 'init': 'pca'}
        params = {
            'perplexity': 30.0,
            'learning_rate': 'auto',  # or any value in [10 -> 1000] may be good
            'n_iter': 10000,
            'n_iter_without_progress': 1000,
            'metric': 'cosine',
            'init': 'pca',
            'n_jobs': 20,
        }
        return TSNE(REDUCED_DIM, **params).fit_transform(embeddings)

def get_token_info(model: torch.nn.Module,
                   tokenizer: data.tokenizers.Tokenizer,
                   category: str
                   ) -> dict[str, list[str]]:
    """ For each token, compute its embedding vector and its class labels
    """
    vocab = tokenizer.get_vocab()
    cat_tokens = [t for t in vocab if category in t and category != t]
    token_label_pairs = [select_plotted_token(t, category) for t in cat_tokens]
    tokens, labels = zip(*[p for p in token_label_pairs if p is not None])
    encoded = [tokenizer.encode(token) for token in tokens]
    embeddings = model.get_token_embeddings(encoded).numpy()
    return {'tokens': tokens, 'embedded': embeddings, 'labels': labels}


def select_plotted_token(token: str, cat: str) -> tuple[str, str]:
    """ Return token / match pairs if the token starts with any of the strings
        included in any of the match tuple of a given category
    """
    to_check = token.split(cat)[-1]
    for s in CATEGORY_SUBLEVELS[cat]:
        if to_check.startswith(s):  # s is a tuple of strings
            return (token, s)


def compute_partitions(labels: list,  # n_samples elements
                       )-> np.ndarray:  # (n_classes, n_samples, n_samples)
    """ Generate class membership matrix from a vector of labels
    """
    classes = sorted(list(set(labels)))
    label_indices = np.array([classes.index(l) for l in labels])
    partitions = np.zeros((len(classes), len(labels), len(labels)))   
    for j, k in enumerate(label_indices):
        partitions[k, j, j] = 1.0
    return partitions


def covariance(samples: np.ndarray) -> np.ndarray:
    """ ...
    """
    return samples.T @ samples
    

def logdet(matrix: np.ndarray) -> float:
    """ ...
    """
    sign, log_det = np.linalg.slogdet(matrix)
    return sign * log_det


def rate_distortion(data: np.ndarray,  # (n_samples, n_features)
                    epsilon: float  # precision parameter
                    ) -> float:  # rate distortion
    """ Compute non-asymptotic rate distortion for finite samples
    """
    n_samples, n_features = data.shape
    identity_matrix = np.eye(n_features)
    scaling_factor = n_features / (n_samples * epsilon)
    rate_matrix = identity_matrix + scaling_factor * covariance(data)
    return 1 / 2 * logdet(rate_matrix)


def rate_reduction(samples: np.ndarray,  # (n_samples, n_features)
                   partitions: np.ndarray,  # (n_classes, n_samples, n_samples)
                   epsilon: float=0.01,  # precision parameter
                   normalize: bool=False,  # normalize data or not
                   ) -> float:  # rate reduction
    """ Compute rate reduction as the reduction between rate distortion as
        computed on the whole data and computed for each class separately
    """
    # Normalize data if required
    if normalize:
        scaler = StandardScaler()
        samples = scaler.fit_transform(samples)
        
    # Compute total rate distortion
    n_samples, _ = samples.shape
    total_r = rate_distortion(samples, epsilon)
    
    # Compute total and mean-class rate distortions
    mean_class_r = 0
    for class_partition in partitions:
        class_samples = samples[np.diag(class_partition) == 1]
        n_class_samples = class_samples.shape[0]
        if n_class_samples == 0: continue
        class_rate = rate_distortion(class_samples, epsilon)
        mean_class_r += n_class_samples / n_samples * class_rate
    
    # Return rate reduction
    return total_r - mean_class_r


def rate_reduction_ma(samples: np.ndarray,  # (n_samples, n_features)
                      partitions: np.ndarray,  # (n_classes, n_samples, n_samples)
                      epsilon: float=0.01,  # precision parameter
                      normalize: bool=True,  # normalize data or not
                      ) -> float:  # rate reduction
    """ Almost copy / pasted from https://github.com/Ma-Lab-Berkeley/ReduNet
        Used to check that my own way of computing rate reduction is correct
    """
    # Normalize data if required
    if normalize:
        scaler = StandardScaler()
        samples = scaler.fit_transform(samples)
        
    # Compute expansion rate
    n_samples, n_features = samples.shape
    identity = np.eye(n_features)
    scaling_factor = n_features / (n_samples * epsilon)
    loss_expd = logdet(scaling_factor * covariance(samples) + identity) / 2.
    
    # Compute compression rate
    loss_comp = 0.
    for class_partition in partitions:
        class_samples = samples[np.diag(class_partition) == 1]
        n_class_samples = class_samples.shape[0]
        if n_class_samples == 0: continue
        class_scaling_factor = n_features / (n_class_samples * epsilon)
        class_logdet = logdet(identity + class_scaling_factor * covariance(class_samples))
        loss_comp += class_logdet * n_class_samples / (2 * n_samples)
    
    # Return rate reduction
    return loss_expd - loss_comp
