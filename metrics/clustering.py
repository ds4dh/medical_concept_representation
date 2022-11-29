import numpy as np
import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# all_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'w'] * 10
all_colors = (list(plt.cm.tab10(np.arange(10))) + ["crimson", "indigo"]) * 10


def clustering_task_ehr(model, tokenizer):
    model.eval()  # may not be needed when used from test_fn
    fig = plt.figure()
    for i, category in enumerate(['DIA_', 'PRO_', 'MED_']):
        
        token_info = get_token_info(model, tokenizer, category)
        reduced = compute_reduced_representation(token_info['embedded'])        
        unique_initials = list(set(token_info['initials']))
        unique_colors = all_colors[:len(unique_initials)]

        kwargs = {} if reduced.shape[-1] <= 2 else {'projection': '3d'}
        ax = fig.add_subplot(1, 3, i + 1, **kwargs)
        ax.set_title('Low-d projection of %s embeddings' % category[:-1])
        for initial, color in zip(unique_initials, unique_colors):
            sub_category_indices = np.where(token_info['initials'] == initial)
            data = reduced[sub_category_indices]
            data = [data[:, i] for i in range(data.shape[-1])]
            ax.scatter(*data, c=color, edgecolors='k', label=initial)
            ax.legend()

        for word, coord in zip(token_info['tokens'], reduced):
            if np.random.rand() < 0.1:
                coord = [c + 0.05 for c in coord]
                ax.text(*coord, word, fontsize=4)
    
    plt.tight_layout()
    plt.show()

def compute_reduced_representation(embeddings, algorithm='pca', n_dims=2):
    if algorithm == 'pca':
        return PCA().fit_transform(embeddings)[:, :n_dims]
    elif algorithm == 'tsne':
        params = {'learning_rate': 'auto', 'init': 'pca'}
        return TSNE(n_dims, **params).fit_transform(embeddings)
    else:
        raise ValueError('Invalid algorithm name (pca, tsne).')

def get_token_info(model, tokenizer, cat):
    vocab = tokenizer.get_vocab()
    if cat == 'PRO_':
        pos = 1
        tokens = [token for token in vocab if cat in token
                                          and cat != token
                                          and token.split('_')[-1][0] == '0']
    else:
        pos = 0
        tokens = [token for token in vocab if cat in token and cat != token]
    encoded = [tokenizer.encode(token) for token in tokens]
    embeddings = model.get_token_embeddings(encoded)
    initials = np.array([token.split('_')[-1][pos] for token in tokens])
    return {'tokens': tokens, 'embedded': embeddings, 'initials': initials}

clustering_task_ehr(model, tokenizer)