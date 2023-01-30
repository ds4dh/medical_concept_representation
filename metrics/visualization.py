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


CATEGORY_SUBLEVELS = {
    'DIA_': ['S', 'T', 'O', 'P', 'F', 'J', 'K', 'I', 'N'],
    'PRO_': ['00', '02', '04', '0B', '0D', '0F', '0H', '0S', '0T', '0U'],
    'MED_': ['C', 'J', 'L', 'N', 'R', 'S']
}
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


def visualization_task_ehr(model: torch.nn.Module,
                           tokenizer: data.tokenizers.Tokenizer,
                           logger: pl.loggers.tensorboard.TensorBoardLogger
                           ) -> None:
    """ Reduce the dimensionality of concept embeddings for different categories
        and log a scatter plot of the low-dimensional data to tensorboard
    """
    fig = plt.figure(figsize=FIG_SIZE)
    for subplot_idx, category in enumerate(CATEGORY_SUBLEVELS.keys()):
        token_info = get_token_info(model, tokenizer, category)
        reduced = compute_reduced_representation(token_info['embedded'])
        plot_reduced_data(reduced, fig, token_info, category, subplot_idx)
    log_figure_to_tensorboard(fig, logger)


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
    label_array = np.array(token_info['labels'])
    for label, color in zip(unique_labels, unique_colors):
        sub_category_indices = np.where(label_array == label)
        data = reduced_data[sub_category_indices]
        data = [data[:, i] for i in range(data.shape[-1])]
        ax.scatter(*data, **SCATTER_PARAMS, color=color, label=label)
        
    # Add token indices as text annotation for every data point
    for word, coord in zip(token_info['tokens'], reduced_data):
        if np.random.rand() < 0.1:
            coord = [c + 0.05 for c in coord]
            ax.text(*coord, word, fontsize=BASE_TEXT_SIZE // 2)
            
    # Polish figure
    ax.margins(x=0.0, y=0.0)
    ax.legend(**LEGEND_PARAMS, ncol=len(unique_labels) // 2)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if data_dim > 2: ax.set_zticklabels([])
    
def log_figure_to_tensorboard(fig: matplotlib.figure.Figure,
                              logger: pl.loggers.tensorboard.TensorBoardLogger
                              ) -> None:
    """ Log the visualization plot to tensorboard as an image stream
    """
    temp_file_name = tempfile.NamedTemporaryFile(suffix='.png').name
    fig.savefig(temp_file_name, dpi=300, bbox_inches='tight')  
    image = np.asarray(Image.open(temp_file_name)).transpose(2, 0, 1)
    logger.experiment.add_image('visualization', image)
    

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
    to_check = token.split(cat)[-1]
    for s in CATEGORY_SUBLEVELS[cat]:
        if to_check.startswith(s):
            return (token, s)
        