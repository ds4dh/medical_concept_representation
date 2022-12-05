
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd 
import torch 
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('tkagg')
import numpy as np
import os 
import pickle 
import json 
import models 
import data 


EXPERIMENT_NAME = 'analysis_1__Concept_Visualization'
# PATH = '/home/dproios/work/AIME_DS4DH/data'
ALGORITHM = 'pca' # 'tsne'
# FPATH = os.path.join('/home/dproios/work/automated_phenotyping/data/datasets/autophe/time_categorized')



def compute_reduced_representation(embeddings, algorithm='pca', n_dims=2):
    if algorithm == 'pca':
        return PCA().fit_transform(embeddings)[:, :n_dims]
    elif algorithm == 'tsne':
        params = {'learning_rate': 'auto', 'init': 'pca'}
        return TSNE(n_dims, **params).fit_transform(embeddings)
    else:
        raise ValueError('Invalid algorithm name (pca, tsne).')


def visualize(reduced, dfc, labels_list, cat, logger):
    fig = plt.figure()  # add height / width values
    for i in labels_list:
        to_vis= reduced[:,dfc['label'] == i]
        plt.scatter(*to_vis, s=4)
    plt.legend(labels_list)
    # plt legend right out of the box 
    plt.legend(labels_list, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # plt disable ticks 
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(cat)
    # os.makedirs(os.path.join(os.path.dirname(__file__), f'figures_{EXPERIMENT_NAME}'), exist_ok=True)
    # FIGURE_PATH = os.path.join(os.path.dirname(__file__), f'figures_{EXPERIMENT_NAME}', f'{cat}_{ALGORITHM}.png')
    # plt.savefig(FIGURE_PATH)
    # print(f'Figure saved at {FIGURE_PATH}')
    # return FIGURE_PATH
    fig.canvas.draw()  # I added this to avoid the savefig call
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = torch.from_numpy(data).permute(2, 0, 1)
    logger.experiment.add_image('test', data)

def get_most_populous_labels(words: list, number_of_chars): 
    word_list = [w.split('_')[1] for w in words]
    # get word with min length in_list
    arr = pd.Series([w.split('_')[1][:number_of_chars] for w in words]).value_counts().index.to_list()
    return arr[:10]


def get_categories_per_token(vocabulary, categorization_strategy):
    '''
    unfortunately it's hard to be agnostic to tokens so these rules are based on.
    For different ones you need to define the strategy accordingly
    ('MED' | 'DIA' |  'LAB' |  'LOC' |  'DEM'  |  'PRO')_<NOT_RELEVANT_PART_OF_CODE>
    '''
    all_rows =[] 
    for w in vocabulary:
        for cat in ['MED','DIA', 'LAB', 'LOC', 'DEM' , 'PRO']:
            if ( 
                    w!=(cat+'_')  # filter out category tokens 'MED_' 'DIA_' etc
                    and w.split('_')[0] == cat # exclude words that don't start with 'UNK' 'PAD' 'P' O X  MED_XX DIA_XX etc
            ):
                all_rows.append(        
                    {'category': w.split('_')[0], 'word':w}
                )
    return pd.DataFrame.from_records(all_rows).reset_index(drop=True)


def evaluate(tokenizer, model, logger, categorization_strategy='prefix_codes', dimensionality_reduction_algorithm = 'pca'):
    '''
    @param tokenizer: tokenizer used to assign numerical IDs to words 
    @param model: model to evaluate
    @param categorization_strategy: how to categorize the tokens: Important this depends on the tokenizer
    '''
    vocabulary = tokenizer.get_vocab()
    assert len([ w  for w in tokenizer.get_vocab()]) > 0, 'Vocabulary is empty'
    
    # get category per token
    df = get_categories_per_token(vocabulary, categorization_strategy)
    
    number_of_chars = 2 
    for cat in ['MED','DIA', 'LAB', 
        #'LOC', 
        #'DEM' , 
        #'PRO'
    ]:
        dfc = df[df['category'] == cat].copy()
        words = dfc['word']
        labels_list = get_most_populous_labels(words, number_of_chars)
        dfc['label'] = dfc['word'].apply(
            lambda x: x.split('_')[1][:number_of_chars] 
            if (x.split('_')[1][:number_of_chars] in labels_list) 
            else 'OTHER'
        )
        labels_list = labels_list + ['OTHER']
        tokens = words.apply(lambda x: tokenizer.encode(x))
        embeddings = model.get_token_embeddings(tokens)
        # reduce dimensionality
        reduced = compute_reduced_representation(embeddings, algorithm=dimensionality_reduction_algorithm, n_dims=2).reshape(2,-1)
        visualize(reduced, dfc, labels_list, cat, logger)


if __name__ == '__main__':
    config_path = './config.toml'
    
    modelfn, run_params, data_params, train_params, model_params = models.load_model_and_params_from_config(config_path)
    pipeline = data.DataPipeline(data_params,run_params,train_params,model_params)
    model_params['vocab_sizes'] = pipeline.tokenizer.vocab_sizes
    params_dict = {**model_params, **run_params, **data_params, **train_params}

    model = modelfn(**params_dict)
    evaluate(pipeline.tokenizer, model)
