import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tempfile
import data
import pytorch_lightning as pl
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Data parameters computed using vocab_count of data.pipeline.tokenizer
# pprint({i: sum([counts[c] for c in counts if 'PRO_%s' % i in c]) for i in
#         ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
#          'O', 'P', 'Q', 'R', 'S', 'T' 'U', 'V', 'W', 'X', 'Y' 'Z']})
COUNTS = {
    'DIA_': {
        '0': 1864, '1': 127, '2': 2322, '3': 92, '4': 518, '5': 1056, '6': 13,
        '7': 1393, '9': 1322, 'A': 25563, 'B': 58675, 'C': 75211, 'D': 152666,
        'E': 437974, 'F': 258978, 'G': 145767, 'H': 30337, 'I': 586006,
        'J': 143878, 'K': 253130, 'L': 52071, 'M': 151653, 'N': 181119,
        'O': 59641, 'P': 14, 'Q': 8793, 'R': 296604, 'S': 57747, 'T': 80214,
        'V': 13056, 'W': 24100, 'X': 5664, 'Y': 95488, 'Z': 590571,
    },
    'PRO_': {
        '0': 328080, '1': 13363, '2': 1375, '3': 43818, '4': 28270, '5': 37023,
        '6': 698, '8': 1794, 'A': 8046, 'B': 54164, 'C': 473, 'D': 2109,
        'F': 897, 'G': 3207, 'H': 884, 'L': 38, 'S': 190, 'X': 71,
    },
    'MED_': {
        '0': 5264, 'A': 4320236, 'B': 1233614, 'C': 1452716, 'D': 263891,
        'G': 46747, 'H': 189027, 'J': 698460, 'L': 65901, 'M': 101169,
        'N': 2046850, 'P': 7411, 'R': 260230, 'S': 164420, 'V': 49583,
    }
}
CATEGORY_SUBLEVELS_MIMIC_NOT_CLEAN = {
    cat: [[t] for t in COUNTS[cat].keys()] for cat in COUNTS.keys()
}
FREQS = {
    k: {kk: vv / sum(v.values()) for kk, vv in v.items()}
    for k, v in COUNTS.items()
}
FREQUENCY_TRESHOLD = 0.01
CATEGORY_SUBLEVELS_FREQUENT = {
    k: [k for k, v in v.items() if v > FREQUENCY_TRESHOLD]
    for k, v in FREQS.items()
}
CATEGORY_SUBLEVELS_FREQUENT = {  # here computed with FREQUENT_TRESHOLD = 0.01
    'DIA_': [('B',), ('C',), ('D',), ('E',), ('F',), ('G',), ('I',), ('J',),
             ('K',), ('L',), ('M',), ('N',), ('O',), ('R',), ('S',), ('T',),
             ('Y',), ('Z')],
    'PRO_': [('0',), ('1',), ('3',), ('4',), ('5',), ('A',), ('B',)],
    'MED_': [('A',), ('B',), ('C',), ('D',), ('H',), ('J',), ('N',), ('R',),
             ('S',)]
}
CATEGORY_SUBLEVELS_OFFICIAL = {
    'DIA_': [('A', 'B',), ('C', 'D0', 'D1', 'D2', 'D3', 'D4',),
             ('D5', 'D6', 'D7', 'D8', 'D9',), ('E',), ('F',), ('G',),
             ('H0', 'H1', 'H2', 'H3', 'H4', 'H5',), ('H6', 'H7', 'H8', 'H9',),
             ('I',), ('J',), ('K',), ('L',), ('M',), ('N',), ('O',), ('P',),
             ('Q',), ('R',), ('S', 'T',), ('V', 'W', 'X', 'Y',), ('Z',),
             ('U',)],
    'PRO_': [('0',), ('1',), ('2',), ('3',), ('4',), ('5',), ('6',), ('7',),
             ('8',), ('9',), ('B',), ('C',), ('D',), ('F',), ('G',), ('H',),
             ('X',)],
    'MED_': [('A',), ('B',), ('C',), ('D',), ('G',), ('H',), ('J',), ('L',),
             ('M',), ('N',), ('P',), ('R',), ('S',), ('V',)],
}
CATEGORY_SUBLEVELS_FERNANDO = {  # why these?
    'DIA_': [('S'), ('T',), ('O',), ('P',), ('F',), ('J',), ('K',), ('I',),
             ('N',)],
    'PRO_': [('00',), ('02',), ('04',), ('0B',), ('0D',), ('0F',), ('0H',),
             ('0S',), ('0T',), ('0U',)],
    'MED_': [('C',), ('J',), ('L',), ('N',), ('R',), ('S',)]
}
CATEGORY_SUBLEVELS = CATEGORY_SUBLEVELS_FERNANDO

# Metric parameters
DIMENSIONALITY_REDUCTION_ALGORITHM = 'tsne'  # 'pca', 'tsne'
assert DIMENSIONALITY_REDUCTION_ALGORITHM in ('pca', 'tsne'),\
    'Invalid algorithm for dimensionality reduction [pca, tsne]'
REDUCED_DIMENSIONALITY = 2  # 2, 3
assert REDUCED_DIMENSIONALITY in [2, 3], 'Invalid reduced dimensionality [2, 3]'
FIG_SIZE = (14, 8)
BASE_TEXT_SIZE = 8
MARKER_SIZE = BASE_TEXT_SIZE * 2
ALL_COLORS = (list(plt.cm.tab10(np.arange(10))) + ["crimson", "indigo"]) * 10
LEGEND_PARAMS = {
    'loc': 'upper center',
    'bbox_to_anchor': (0.5, -0.05),
    'fancybox': True,
    'shadow': True,
    'fontsize': BASE_TEXT_SIZE
}
SCATTER_PARAMS =  {
    'marker': 'o',
    's': MARKER_SIZE,
    'linewidths': 0.5,
    'edgecolors': 'k'
}


def visualization_task(model: torch.nn.Module,
                       pipeline: data.DataPipeline,
                       logger: pl.loggers.tensorboard.TensorBoardLogger,
                       global_step: int,
                       ) -> None:
    """ Reduce the dimensionality of concept embeddings for different categories
        and log a scatter plot of the low-dimensional data to tensorboard
    """
    fig = plt.figure(figsize=FIG_SIZE)
    for subplot_idx, category in enumerate(CATEGORY_SUBLEVELS.keys()):
        token_info = get_token_info(model, pipeline.tokenizer, category)
        reduced = compute_reduced_representation(token_info['embedded'])
        plot_reduced_data(reduced, fig, token_info, category, subplot_idx)
    log_figure_to_tensorboard(fig, 'visualization_metric', logger, global_step)


def plot_reduced_data(reduced_data: np.ndarray,
                      fig: matplotlib.figure.Figure,
                      token_info: dict,
                      category: str,
                      subplot_idx: int
                      ) -> None:
    """ Plot data of reduced dimensionality to a 2d or 3d scatter plot
    """
    # Prepare subfigure and title
    data_dim = reduced_data.shape[-1]
    plot_title = '%s embeddings\n%sd projection (%s)' %\
        (category.split('_')[0], data_dim, DIMENSIONALITY_REDUCTION_ALGORITHM)
    kwargs = {} if data_dim <= 2 else {'projection': '3d'}
    ax = fig.add_subplot(1, 3, subplot_idx + 1, **kwargs)
    ax.set_title(plot_title, fontsize=BASE_TEXT_SIZE * 2)
    
    # Plot data of reduced dimensionality
    unique_labels = list(set(token_info['labels']))
    unique_colors = ALL_COLORS[:len(unique_labels)]
    label_array = np.empty(len(token_info['labels']), dtype=object)
    label_array[:] = token_info['labels']
    for label, color in zip(unique_labels, unique_colors):
        data = reduced_data[[l == label for l in label_array]]
        data = [data[:, i] for i in range(data.shape[-1])]
        ax.scatter(*data, **SCATTER_PARAMS, color=color, label=label)
        
    # Add token indices as text annotation a few data point
    for word, coord in zip(token_info['tokens'], reduced_data):
        if np.random.rand() < 0.01:
            coord = [c + 0.05 for c in coord]
            ax.text(*coord, word, fontsize=BASE_TEXT_SIZE // 2)
            
    # Polish figure
    ax.margins(x=0.0, y=0.0)
    ax.legend(**LEGEND_PARAMS, ncol=len(unique_labels) // 2)
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
    fig.savefig(temp_file_name, dpi=150, bbox_inches='tight')  
    image = np.asarray(Image.open(temp_file_name)).transpose(2, 0, 1)
    logger.experiment.add_image(fig_title, image, global_step=global_step)
    

def compute_reduced_representation(embeddings: torch.Tensor) -> np.ndarray:
    """ Reduce the dimensionality of high-dimensional concept embeddings
    """
    if DIMENSIONALITY_REDUCTION_ALGORITHM == 'pca':
        return PCA().fit_transform(embeddings)[:, :REDUCED_DIMENSIONALITY]
    elif DIMENSIONALITY_REDUCTION_ALGORITHM == 'tsne':
        params = {'learning_rate': 'auto', 'init': 'pca'}
        return TSNE(REDUCED_DIMENSIONALITY, **params).fit_transform(embeddings)
    

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
    embeddings = model.get_token_embeddings(encoded)
    return {'tokens': tokens, 'embedded': embeddings, 'labels': labels}


def select_plotted_token(token: str, cat: str) -> tuple[str, str]:
    """ Return token / match pairs if the token starts with any of the strings
        included in any of the match tuple of a given category
    """
    to_check = token.split(cat)[-1]
    for s in CATEGORY_SUBLEVELS[cat]:
        if to_check.startswith(s):  # s is a tuple of strings
            return (token, s)
        