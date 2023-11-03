import torch
import numpy as np
import hdbscan
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tempfile
import data
import pytorch_lightning as pl
from PIL import Image
from adjustText import adjust_text
from hdbscan import flat
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure as hcv_measure,
)


# Metric parameters
CATEGORY_SUBLEVELS = {
    'DIA_': [('C', 'D0', 'D1', 'D2', 'D3', 'D4')],
    'PRO_': [('B',)],
    'MED_': [('L',)],
}
PLOT_CAT_MAP = {
    'DIA_': 'ICD10-CM codes (C0 -> D4)',
    'PRO_': 'ICD10-PCS codes (B0 -> BY)',
    'MED_': 'ATC codes (L01 -> L04)',
}
DIMENSIONALITY_REDUCTION_ALGORITHM = 'tsne'  # 'pca', 'tsne'
assert DIMENSIONALITY_REDUCTION_ALGORITHM in ('pca', 'tsne'),\
    'Invalid algorithm for dimensionality reduction [pca, tsne]'
REDUCED_DIMENSIONALITY = 2  # None for no dimensionality reduction
FIG_SIZE = (15, 15)
TEXT_SIZE = 16
N_ANNOTATED_SAMPLES = 60
COLORS = (list(plt.cm.tab20(np.arange(20)[0::2])) +\
          list(plt.cm.tab20(np.arange(20)[1::2]))) * 10
SCATTER_PARAMS = {'s': 50, 'linewidth': 0, 'alpha': 0.5}
LEAF_SEPARATION = 0.3
MIN_CLUSTER_SIZE = 6
NA_COLOR = np.array([0.0, 0.0, 0.0, 1.0])  # black (never a cluster color)


def hierachization_task(model: torch.nn.Module,
                        pipeline: data.DataPipeline,
                        logger: pl.loggers.tensorboard.TensorBoardLogger,
                        global_step: int,
                        ) -> None:
    """ Reduce the dimensionality of concept embeddings for different categories
        and log a scatter plot of the low-dimensional data to tensorboard
    """
    print('\nProceeding with hierarchization testing metric')
    fig, axs = plt.subplots(3, 3, figsize=FIG_SIZE)
    for cat_idx, cat in enumerate(CATEGORY_SUBLEVELS.keys()):
        print('Clustering %s tokens (d=%s)' % (cat, REDUCED_DIMENSIONALITY))
        token_info = get_token_info(model, pipeline.tokenizer, cat)
        if REDUCED_DIMENSIONALITY != 2:
            token_info['data'] = compute_reduced_representation(
                token_info['embedded'],
                dim=REDUCED_DIMENSIONALITY,
                algorithm='pca',
            )
            cluster_info = get_cluster_info(token_info)
            if cluster_info == None: continue
            token_info['data'] = compute_reduced_representation(
                token_info['data'],
                dim=2,
                algorithm=DIMENSIONALITY_REDUCTION_ALGORITHM,
            )
            plot_hierarchy(token_info, cluster_info, axs, cat, cat_idx)
        else:
            data = compute_reduced_representation(
                token_info['embedded'],
                dim=REDUCED_DIMENSIONALITY,
                algorithm=DIMENSIONALITY_REDUCTION_ALGORITHM,
            )
            token_info['data'] = data
            cluster_info = get_cluster_info(token_info)
            if cluster_info == None: continue
            plot_hierarchy(token_info, cluster_info, axs, cat, cat_idx)
        
    plt.tight_layout()
    log_figure_to_tensorboard(fig, 'hierarchization_metric', logger, global_step)
    

def get_cluster_info(token_info, fixed_n_clusters=True):
    """ ...
    """
    # Find cluster affordances based on cluster hierarchy
    clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE)
    clusterer.fit(token_info['data'])
    unique_classes = sorted(list(set(token_info['class_lbls'])))
    if fixed_n_clusters:
        try:
            n_clusters = len([c for c in unique_classes
                              if token_info['class_lbls'].count(c) > MIN_CLUSTER_SIZE])
            clusterer = flat.HDBSCAN_flat(token_info['data'],
                                        n_clusters=n_clusters,
                                        clusterer=clusterer)
        except IndexError:
            print(' - Clustering algorithm did not converge!')
            return None
            
    # Return everything needed
    return {'clusterer': clusterer,
            'cluster_lbls': clusterer.labels_,
            'n_clusters': n_clusters}


def plot_hierarchy(token_info, cluster_info, axs, cat, cat_idx):
    """ ...
    """
    # Find and measure best match between empirical and theoretical clusters
    cluster_colors, class_colors = align_cluster_colors(token_info, cluster_info)
    cluster_labels, class_labels = compute_labels(cluster_colors, class_colors)
    ari_match = adjusted_rand_score(class_labels, cluster_labels)
    nmi_match = normalized_mutual_info_score(class_labels, cluster_labels)
    homog, compl, v_match = hcv_measure(class_labels, cluster_labels)
    print(' - ari: %.5f\n - nmi: %.5f\n - hom: %.5f\n - comp: %.5f\n - v: %.5f'%
          (ari_match, nmi_match, homog, compl, v_match))

    # Visualize theoretical clusters (using, e.g., ICD10-CM code hierarchy)
    data = token_info['data']
    axs[0, cat_idx].scatter(*data.T, c=class_colors, **SCATTER_PARAMS)
    axs[0, cat_idx].set_xticklabels([]); axs[0, cat_idx].set_yticklabels([])
    axs[0, cat_idx].set_xticks([]); axs[0, cat_idx].set_yticks([])
    
    # Add text annotation around some data points
    loop = list(zip(data, token_info['tokens']))
    np.random.shuffle(loop)
    texts = []
    for d, t in loop[:N_ANNOTATED_SAMPLES]:
        texts.append(axs[0, cat_idx].text(d[0], d[1], t, fontsize='xx-small'))
    arrowprops = dict(arrowstyle='->', color='k', lw=0.5)
    adjust_text(texts, x=data[:, 0], y=data[:, 1], ax=axs[0, cat_idx],
                force_text=(0.2, 0.3), arrowprops=arrowprops)
    
    # Visualize empirical clusters
    axs[1, cat_idx].scatter(*data.T, c=cluster_colors, **SCATTER_PARAMS)
    axs[1, cat_idx].set_xticklabels([]); axs[1, cat_idx].set_yticklabels([])
    axs[1, cat_idx].set_xticks([]); axs[1, cat_idx].set_yticks([])
    
    # Visualize empirical cluster tree
    cluster_info['clusterer'].condensed_tree_.plot(
        axis=axs[2, cat_idx], leaf_separation=LEAF_SEPARATION, colorbar=True)
    add_cluster_info(axs[2, cat_idx],
                     cluster_info['clusterer'],
                     cluster_info['n_clusters'],
                     cluster_colors)
    axs[2, cat_idx].invert_yaxis()
    
    # Add title around the figure
    axs[0, cat_idx].set_title(PLOT_CAT_MAP[cat], fontsize=TEXT_SIZE)
    axs[0, 0].set_ylabel('Clusters derived from terminology', fontsize=TEXT_SIZE)
    axs[1, 0].set_ylabel('Empirical clusters', fontsize=TEXT_SIZE)
    axs[2, 0].set_ylabel('Empirical cluster tree', fontsize=TEXT_SIZE)
    axs[2, cat_idx].get_yaxis().set_tick_params(left=False)
    axs[2, cat_idx].get_yaxis().set_tick_params(right=False)
    axs[2, cat_idx].set_yticks([])
    axs[2, cat_idx].spines['top'].set_visible(True)
    axs[2, cat_idx].spines['right'].set_visible(True)
    axs[2, cat_idx].spines['bottom'].set_visible(True)
    if cat_idx > 0: axs[2, cat_idx].set_ylabel('', fontsize=TEXT_SIZE)


def add_cluster_info(ax: plt.Axes,
                     clusterer: hdbscan.HDBSCAN,
                     n_clusters: int,
                     cluster_colors: list[np.ndarray]):
    """ ...
    """
    # Select the branches that will be highlighted (empirical clusters)
    branch_df = clusterer.condensed_tree_.to_pandas().sort_values('lambda_val')
    branch_df = branch_df[branch_df['child_size'] > 1]
    selected_branches, selected_branch_sizes = [], []
    for _, branch in branch_df.iterrows():
        if len(selected_branches) == n_clusters: break
        selected_branches.append(branch['child'])
        selected_branch_sizes.append(branch['child_size'])
        if branch['parent'] in selected_branches:
            selected_branches.remove(branch['parent'])
            selected_branch_sizes.remove(branch['child_size'])
    
    # Get all cluster bounds and draw a white rectangle to add info on it
    cluster_bounds = clusterer.condensed_tree_.get_plot_data()['cluster_bounds']
    cluster_bottom = min([cluster_bounds[b][-1] for b in selected_branches])
    left, right = ax.get_xlim()
    _, bottom = ax.get_ylim()  # top would be the end of the tree
    rectangle_width = (right - left) * 0.998
    bottom_point = (left + 0.001 * rectangle_width, cluster_bottom - 0.001)
    rectangle_height = cluster_bottom / 15
    rectangle_specs = [bottom_point, rectangle_width, rectangle_height]
    mask = patches.Rectangle(*rectangle_specs, facecolor='white', zorder=10)
    ax.add_patch(mask)
    top_lim = bottom_point[1] + rectangle_height * 1.025
    bottom_lim = bottom - 0.11 * (top_lim - bottom)
    ax.set_ylim([top_lim, bottom_lim])
    
    # Retrieve colours and match then to the correct clusters
    assigned_colors = [c for c in cluster_colors if not np.array_equal(c, NA_COLOR)]
    unique_colors = np.unique(assigned_colors, axis=0)
    sizes_from_colors = [len([ac for ac in assigned_colors if (ac == uc).all()])
                         for uc in unique_colors]
    sizes_from_tree = [int(cluster_bounds[b][1] - cluster_bounds[b][0])
                       for b in selected_branches]
    match_indices = [sizes_from_colors.index(v) for v in sizes_from_tree]
    unique_colors = unique_colors[match_indices]
    
    # Plot small circles with the right color, below each branch of the tree
    for b, c in zip(selected_branches, unique_colors):
        anchor_point = (
            (cluster_bounds[b][0] + cluster_bounds[b][1]) / 2 * LEAF_SEPARATION,
            bottom_point[1] + rectangle_height / 2
        )
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        zorder = int(1e6 + anchor_point[0])
        ax.plot(*anchor_point, 'o', color=c, markersize=10, zorder=zorder)
        ax.set_xlim(xlim), ax.set_ylim(ylim)


def align_cluster_colors(token_info, cluster_info):
    """ ...
    """
    # Identify possible colour values
    cluster_lbls = cluster_info['cluster_lbls']
    class_lbls = token_info['class_lbls']
    unique_classes = sorted(list(set(class_lbls)))
    unique_clusters = sorted(list(set(cluster_lbls)))

    # In this case, true labels define the maximum number of colours
    if len(unique_classes) >= len(unique_clusters):
        color_map = best_color_match(cluster_lbls, class_lbls, unique_clusters, unique_classes)
        color_map = {k: unique_classes.index(v) for k, v in color_map.items()}
        cluster_colors = [COLORS[color_map[i]] if i >= 0 else NA_COLOR for i in cluster_lbls]
        class_colors = [COLORS[unique_classes.index(l)] for l in class_lbls]
    
    # In this case, empirical clusters define the maximum number of colours
    else:
        color_map = best_color_match(class_lbls, cluster_lbls, unique_classes, unique_clusters)
        color_map = {unique_classes.index(k): v for k, v in color_map.items()}
        cluster_colors = [COLORS[i] if i >= 0 else NA_COLOR for i in cluster_lbls]
        class_colors = [COLORS[color_map[unique_classes.index(l)]] for l in class_lbls]
    
    # Return aligned empirical and theorical clusters
    return cluster_colors, class_colors


def compute_labels(cluster_colors, class_colors):
    """ ...
    """
    # Check for existence of unassigned samples in the empirical clusters
    cluster_colors, class_colors = cluster_colors.copy(), class_colors.copy()
    assigned_sample_ids = np.where(np.any(cluster_colors != NA_COLOR, axis=1))
    unassigned_samples_exist = len(assigned_sample_ids) == len(cluster_colors)

    # Compute cluster and class labels
    if unassigned_samples_exist: class_colors.append(NA_COLOR)
    _, class_labels = np.unique(class_colors, axis=0, return_inverse=True)
    _, cluster_labels = np.unique(cluster_colors, axis=0, return_inverse=True)
    if unassigned_samples_exist: class_labels = class_labels[:-1]

    # Return cluster and class labels
    return cluster_labels, class_labels


def compute_reduced_representation(embeddings: np.ndarray,
                                   dim='2',
                                   algorithm='pca'
                                   ) -> np.ndarray:
    """ Reduce the dimensionality of high-dimensional concept embeddings
    """
    if dim is None:
        return embeddings
    if algorithm == 'pca':
        return PCA().fit_transform(embeddings)[:, :dim]
    elif algorithm == 'tsne':
        params = {
            'perplexity': 30.0,
            'learning_rate': 'auto',  # or any value in [10 -> 1000] may be good
            'n_iter': 10000,
            'n_iter_without_progress': 1000,
            'metric': 'cosine',
            'init': 'pca',
            'n_jobs': 20,
        }
        return TSNE(dim, **params).fit_transform(embeddings)


def get_token_info(model: torch.nn.Module,
                   tokenizer: data.tokenizers.Tokenizer,
                   category: str
                   ) -> dict[str, list[str]]:
    """ For each token, compute its embedding vector and its class lbls
    """
    vocab = tokenizer.get_vocab()
    cat_tokens = [t for t in vocab if category in t and category != t]
    token_label_class = [select_plotted_token(t, category) for t in cat_tokens]
    token_label_class = [p for p in token_label_class if p is not None]
    tokens, lbls, class_lbls = zip(*token_label_class)
    encoded = [tokenizer.encode(token) for token in tokens]
    embeddings = model.get_token_embeddings(encoded).numpy()
    return {
        'tokens': [t.split('_')[-1] for t in tokens],  # remove category text
        'embedded': embeddings,
        'lbls': lbls,
        'class_lbls': class_lbls
    }


def select_plotted_token(token: str, cat: str) -> tuple[str, str]:
    """ Return token / match pairs if the token starts with any of the strings
        included in any of the match tuple of a given category
    """
    to_check = token.split(cat)[-1]
    n_letters = 3 if cat == 'MED_' else 2
    for matches in CATEGORY_SUBLEVELS[cat]:
        for match in matches:
            if to_check.startswith(match):  # s is not a tuple here, just str
                return (token, matches, to_check[:n_letters])


def best_color_match(src_lbls, tgt_lbls, unique_src_lbls, unique_tgt_lbls):
    """ Find the best match between subcategories, based on cluster memberships
    """
    cost_matrix = np.zeros((len(unique_src_lbls), len(unique_tgt_lbls)))
    for i, src_lbl in enumerate(unique_src_lbls):
        for j, tgt_lbl in enumerate(unique_tgt_lbls):
            count = sum(s == src_lbl and t == tgt_lbl
                        for s, t in zip(src_lbls, tgt_lbls))
            cost_matrix[i, j] = -count
    
    rows, cols = linear_sum_assignment(cost_matrix)
    return {unique_src_lbls[i]: unique_tgt_lbls[j] for i, j in zip(rows, cols)}


def log_figure_to_tensorboard(fig: plt.Figure,
                              fig_title: str,
                              logger: pl.loggers.tensorboard.TensorBoardLogger,
                              global_step: int
                              ) -> None:
    """ Log the outcomization plot to tensorboard as an image stream
    """
    temp_file_name = tempfile.NamedTemporaryFile(suffix='.png').name
    fig.savefig(temp_file_name, dpi=300, bbox_inches='tight')  
    image = np.asarray(Image.open(temp_file_name)).transpose(2, 0, 1)
    logger.experiment.add_image(fig_title, image, global_step=global_step)
