import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def clustering_task_ehr(model, tokenizer, n_dims=3, algorithm='pca'):
    model.eval()  # may not be needed when used from test_fn
    assert n_dims <= 3, 'Parameter n_dims must be at most 3.'
    categories = ['DEM_', 'LOC_', 'LAB_', 'MED_', 'PRO_', 'DIA_']
    fig = plt.figure()
    for i, category in enumerate(categories):
        
        vocab = tokenizer.get_vocab()
        tokens = [token for token in vocab
                  if category in token and category != token]
        token_indices = [tokenizer.encode(token) for token in tokens]
        word_embeddings = model.get_token_embeddings(token_indices)

        if algorithm == 'pca':
            reduced = PCA().fit_transform(word_embeddings)[:, :n_dims]
        elif algorithm == 'tsne':
            params = {'learning_rate': 'auto', 'init': 'pca'}
            reduced = TSNE(n_dims, **params).fit_transform(word_embeddings)
        else:
            raise ValueError('Invalid algorithm name (pca, tsne).')
                
        scatter_data = [reduced[:, i] for i in range(reduced.shape[-1])]
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        ax.scatter(*scatter_data, edgecolors='k', c='r')
        for word, coord in zip(tokens, reduced):
            initial = word.split('_')[-1][0]  # to use for color coding
            coord = [c + 0.05 for c in coord]
            ax.text(*coord, word, fontsize=4)
            
    plt.tight_layout()
    plt.show()


def compute_closest_token(*args, **kwargs):
    pass
