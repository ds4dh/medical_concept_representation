import pandas as pd
import torch 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def compute_reduced_representation(embeddings, algorithm='tsne', n_dims=3):
    if algorithm == 'pca':
        return PCA().fit_transform(embeddings)[:, :n_dims]
    elif algorithm == 'tsne':
        params = {'learning_rate': 'auto', 'init': 'pca'}
        return TSNE(n_dims, **params).fit_transform(embeddings)
    else:
        raise ValueError('Invalid algorithm name (pca, tsne).')


def log_visualization(reduced, dfc, labels_list, cat, logger):
    fig = plt.figure()
    for i in labels_list:
        to_visualize = reduced.reshape(2, -1)[:, dfc['label'] == i]
        plt.scatter(*to_visualize, s=4)
    plt.legend(labels_list,
               bbox_to_anchor=(1.05, 1.0),
               loc='upper left',
               borderaxespad=0.0)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(cat)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = torch.from_numpy(data).permute(2, 0, 1)
    logger.experiment.add_image('test', data)


def get_most_populous_labels(words: list, number_of_chars: int=2): 
    arr = pd.Series([w.split('_')[1][:number_of_chars] for w in words])
    return arr.value_counts().index.to_list()[:10]


def get_categories_per_token(vocabulary):
    all_rows = []
    for w in vocabulary:
        for cat in ['MED','DIA', 'LAB', 'LOC', 'DEM' , 'PRO']:
            if (w != '%s_' % cat  # filter out category tokens 'MED_' 'DIA_' etc
            and w.split('_')[0] == cat): # exclude words that don't start with 'UNK' 'PAD' 'P' O X  MED_XX DIA_XX etc
                all_rows.append({'category': w.split('_')[0],
                                 'word': w})
    return pd.DataFrame.from_records(all_rows).reset_index(drop=True)


def visualization_task_ehr(tokenizer, model, logger):
    vocabulary = tokenizer.get_vocab()
    df = get_categories_per_token(vocabulary)
    
    for cat in ['MED','DIA', 'LAB']:
        print(' - Visualizing %s data embeddings' % cat)
        dfc = df[df['category'] == cat].copy()
        words = dfc['word']
        labels_list = get_most_populous_labels(words)
        dfc['label'] = dfc['word'].apply(lambda x: x.split('_')[1][:2]
                                         if (x.split('_')[1][:2] in labels_list) 
                                         else 'OTHER')
        
        labels_list = labels_list + ['OTHER']
        tokens = words.apply(lambda x: tokenizer.encode(x))
        embeddings = model.get_token_embeddings(tokens)
        
        # Reduce dimensionality
        reduced = compute_reduced_representation(embeddings, n_dims=2)
        log_visualization(reduced, dfc, labels_list, cat, logger)
