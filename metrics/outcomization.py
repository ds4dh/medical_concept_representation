import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import tempfile
import data
import pytorch_lightning as pl
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm


DIMENSIONALITY_REDUCTION_ALGORITHM = 'tsne'  # 'pca', 'tsne'
assert DIMENSIONALITY_REDUCTION_ALGORITHM in ('pca', 'tsne'),\
    'Invalid algorithm for dimensionality reduction [pca, tsne]'
REDUCED_DIM = 2  # 2, 3
assert REDUCED_DIM in [2, 3], 'Invalid reduced dimensionality [2, 3]'
FIG_SIZE = (14, 5)
SMALL_TEXT_SIZE = 8
BIG_TEXT_SIZE = 14
OUTCOME_CLASSES = {
    'length-of-stay': ['LBL_SHORT', 'LBL_LONG'],
    'readmission': ['LBL_AWAY', 'LBL_READM'],
    'mortality': ['LBL_ALIVE', 'LBL_DEAD'],
}
ALL_OUTCOME_LEVELS = [t for v in OUTCOME_CLASSES.values() for t in v]
N_FRAMES = 100  # actually corresponds to n_frames - 1
PARTIAL_INFO_LEVELS = np.arange(0.0, 1.0 + 1.0 / N_FRAMES, 1.0 / N_FRAMES)
N_SAMPLES = 1807  # 1807 dead patients = limiting factor
USE_TIME_WEIGHTS = True
TIME_WEIGHTS = [1 / (t + 1) for t in range(100_000)][::-1]


def outcomization_task(model: torch.nn.Module,
                       pipeline: data.DataPipeline,
                       logger: pl.loggers.tensorboard.TensorBoardLogger,
                       global_step: int,
                       ) -> None:
    """ Reduce the dimensionality of concept embeddings for different categories
        and log a scatter plot of patients with different outcome labels
    """
    print('\nProceeding with outcomization testing metric')
    proj_params = {'projection': '3d'} if REDUCED_DIM == 3 else {}
    patient_reduced, class_reduced, golds = get_reduced_data(pipeline, model)
    fig_axs_frames = []
    desc = 'Generating gif frame for different levels of partial information'
    for i, partial_info in tqdm(enumerate(PARTIAL_INFO_LEVELS), desc=desc):
        fig, axs = plt.subplots(1, 3, figsize=(16, 5), subplot_kw=proj_params)
        partial_patient_reduced = patient_reduced[i::len(PARTIAL_INFO_LEVELS)]
        scatter_reduced_data(partial_patient_reduced, partial_info,
                             class_reduced, golds, axs)
        plt.tight_layout()
        fig_axs_frames.append((fig, axs))
    gif_title = 'outcomization_metric_%s_samples' % N_SAMPLES
    log_gif_to_tensorboard(fig_axs_frames, gif_title, logger, global_step)


def log_gif_to_tensorboard(fig_axs_frames: list[tuple[plt.Figure, plt.Axes]],
                           gif_title: str,
                           logger: pl.loggers.tensorboard.TensorBoardLogger,
                           global_step: int
                           ) -> None:
    """ Log a list of figures as a video using tensorboard gif visualizer
    """
    fig_frames, axs_frames = zip(*fig_axs_frames)
    normalize_boundaries(axs_frames)
    tensor_frames = []
    for fig_frame in fig_frames:
        temp_file_name = tempfile.NamedTemporaryFile(suffix='.png').name
        fig_frame.savefig(temp_file_name, dpi=100, bbox_inches='tight')  
        np_frame = np.asarray(Image.open(temp_file_name)).transpose(2, 0, 1)
        tensor_frames.append(torch.tensor(np_frame))  # (C, H, W)
    min_c, min_h, min_w = [min([t.shape[i] for t in tensor_frames]) for i in [0, 1, 2]]
    tensor_frames = [t[:min_c,:min_h,:min_w] for t in tensor_frames]  # unified shape
    vid_tensor = torch.stack(tensor_frames, axis=0).unsqueeze(dim=0)
    logger.experiment.add_video(tag=gif_title,
                                vid_tensor=vid_tensor,  # (N, T, C, H, W)
                                fps=12,
                                global_step=global_step)
    

def compute_reduced_representation(embeddings: np.ndarray) -> np.ndarray:
    """ Reduce the dimensionality of high-dimensional concept embeddings
    """
    if DIMENSIONALITY_REDUCTION_ALGORITHM == 'pca':
        return PCA().fit_transform(embeddings)[:, :REDUCED_DIM]
    elif DIMENSIONALITY_REDUCTION_ALGORITHM == 'tsne':
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


def get_reduced_data(pipeline: data.DataPipeline,
                     model: torch.nn.Module):
    """ ...
    """
    # Initialize testing data pipeline
    to_remove = pipeline.run_params['ngrams_to_remove']
    dp = data.JsonReader(pipeline.data_fulldir, 'test')
    dp = data.TokenFilter(dp, to_remove=to_remove, to_split=ALL_OUTCOME_LEVELS)
    
    # Compute patient embeddings and associated labels
    embeddings, golds = [], []
    for s, g in tqdm(dp, desc='Getting patient embeddings'):
        embeddings.append(get_patient_embedding(model, s, pipeline))
        golds.append(g)
    
    # Add outcome embeddings to the set of embeddings that will be reduced
    outcomes_encoded = [pipeline.tokenizer.encode(l) for l in ALL_OUTCOME_LEVELS]
    outcomes_embedded = model.get_token_embeddings(outcomes_encoded)
    embeddings.append(outcomes_embedded)  # all at once
    # if isinstance(outcomes_encoded[0], list):
    #     weights = [1 / pipeline.tokenizer.word_counts[t[0]] for t in outcomes_encoded]
    # else:
    #     weights = [1 / pipeline.tokenizer.word_counts[t] for t in outcomes_encoded]
    # embeddings.extend([w * e for w, e in zip(weights, outcomes_embedded)])

    # Compute reduced representation of both patients and class labels
    print('Reducing embedded data')
    reduced = compute_reduced_representation(torch.cat(embeddings, dim=0))
    patient_reduced = reduced[:-len(ALL_OUTCOME_LEVELS)]
    class_reduced = reduced[-len(ALL_OUTCOME_LEVELS):]

    # Return everything needed for the plot
    return patient_reduced, class_reduced, golds


def scatter_reduced_data(patient_reduced: np.ndarray,
                         partial_info: float,
                         class_reduced: np.ndarray,
                         golds: list[str],
                         axs: list[plt.Axes]):
    """ ...
    """
    # Plot all reduced embeddings for different outcome classes
    for i, (cat, classes) in enumerate(OUTCOME_CLASSES.items()):
        this_class_reduced = class_reduced[2 * i:2 * (i + 1)]
        scatter_reduced_outcomes(classes, this_class_reduced, patient_reduced,
                                 golds, axs[i])
        title = '%s visualization, P = %.2f' % (cat.capitalize(), partial_info)
        axs[i].set_title(title, fontsize=BIG_TEXT_SIZE)


def scatter_reduced_outcomes(classes: list[str],
                             class_reduced: np.ndarray,
                             patient_reduced: np.ndarray,
                             golds: list[str],
                             ax: plt.Axes,
                             n_samples: int=N_SAMPLES):    
    """ Visualize n_samples patient embeddings for each value of a class
    """
    pos_reduced = [r for r, g in zip(patient_reduced, golds) if classes[0] in g]
    neg_reduced = [r for r, g in zip(patient_reduced, golds) if classes[1] in g]  
    pos_samples = random.sample(pos_reduced, min(n_samples, len(pos_reduced)))
    neg_samples = random.sample(neg_reduced, min(n_samples, len(neg_reduced)))
    pos_class_name = classes[0].split('_')[1].lower()
    neg_class_name = classes[1].split('_')[1].lower()
    leg_pos_pat = '%s patient' % pos_class_name.capitalize()
    leg_neg_pat = '%s patient' % neg_class_name.capitalize()
    leg_pos_tok = '%s token' % pos_class_name.capitalize()
    leg_neg_tok = '%s token' % neg_class_name.capitalize()    
    ax.scatter(*np.array(pos_samples).T, c='blue', s=3, alpha=0.5, zorder=1, label=leg_pos_pat)
    ax.scatter(*np.array(neg_samples).T, c='red', s=3, alpha=0.5, zorder=1, label=leg_neg_pat)
    ax.scatter(*np.array(class_reduced[0]).T, c='cyan', s=50, zorder=2, label=leg_pos_tok)
    ax.scatter(*np.array(class_reduced[1]).T, c='yellow', s=50, zorder=2, label=leg_neg_tok)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=SMALL_TEXT_SIZE, ncol=2, loc='upper left')
    

def get_patient_embedding(model: torch.nn.Module,
                          sample: list[str],
                          pipeline: data.DataPipeline,
                          ) -> torch.Tensor:
    """ Generate a sequence embedding for a patient sample in which tokens that
        do not belong to a given category were removed
        - The result is a weighted average of all tokens embeddings
        - Weigths are proportional to token inverse frequency (in train dataset)
        - Time-weights can be added too (elilgibility trace)
        - N embeddings per patient are generated, N = len(PARTIAL_INFO_LEVELS)
    """
    # Encode patient tokens and compute weights based on term frequencies
    encoded = [pipeline.tokenizer.encode(t) for t in sample]
    if isinstance(encoded[0], list):
        weights = [1 / pipeline.tokenizer.word_counts[t[0]] for t in encoded]
    else:
        weights = [1 / pipeline.tokenizer.word_counts[t] for t in encoded]
    
    # Compute patient embeddings for different levels of partial information
    fixed_enc = [encoded[n] for n, t in enumerate(sample) if 'DEM_' in t
                 and not any([s in t for s in OUTCOME_CLASSES])]
    fixed_wgt = [weights[n] for n, t in enumerate(sample) if 'DEM_' in t]
    timed_idx = [n for n, t in enumerate(sample) if 'DEM_' not in t]
    sentence_embeddings = []
    for partial in PARTIAL_INFO_LEVELS:
        
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


def normalize_boundaries(axs_frames: list[plt.Axes]):
    """ Set the same x and y boundaries for all figure frames
    """
    x_lim_mins = [min([ax.get_xlim()[0] for ax in axs_list])
                  for axs_list in list(zip(*axs_frames))]
    x_lim_maxs = [max([ax.get_xlim()[1] for ax in axs_list])
                  for axs_list in list(zip(*axs_frames))]
    y_lim_mins = [min([ax.get_ylim()[0] for ax in axs_list])
                  for axs_list in list(zip(*axs_frames))]
    y_lim_maxs = [max([ax.get_ylim()[1] for ax in axs_list])
                  for axs_list in list(zip(*axs_frames))]
    for axs in axs_frames:
        zipped = zip(axs, x_lim_mins, x_lim_maxs, y_lim_mins, y_lim_maxs)
        for ax, x_min, x_max, y_min, y_max in zipped:
            ax.set_xlim((x_min, x_max))
            ax.set_ylim((y_min, y_max))
        