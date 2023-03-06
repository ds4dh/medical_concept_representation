import os
import sys
import runpy
from pprint import pprint


CONFIG_DIR = os.path.abspath('configs')
BASE_CONFIG_PATH = os.path.join(CONFIG_DIR, 'base_config.toml')
RUN_CONFIG_PATH = os.path.join(CONFIG_DIR, 'run_config.toml')
PARAM_SETS = [

    # {   'exp_id': "'whole_shuffle'",
    #     'gpu_index': "2",
    #     'token_shuffle_mode': "'whole'",
    #     'token_shuffle_prob': "0.5",
    #     'model_used': "'word2vec'",
    #     'ngram_mode': "'word'",
    #     'n_steps': '100_000',
    #     'optimizer': "'hyper-1'",
    #     'hyper_lr': '0.00001',
    #     'lr': '0.001',
    #     'word2vec.d_embed': '512',
    #     'word2vec.n_neg_samples': '0',  # softmax
    # },
    # {   'exp_id': "'whole_shuffle'",
    #     'gpu_index': "2",
    #     'token_shuffle_mode': "'whole'",
    #     'token_shuffle_prob': "0.5",
    #     'model_used': "'fasttext'",
    #     'ngram_mode': "'subword'",
    #     'n_steps': '100_000',
    #     'optimizer': "'hyper-1'",
    #     'hyper_lr': '0.00001',
    #     'lr': '0.001',
    #     'fasttext.d_embed': '512',
    #     'fasttext.n_neg_samples': '0',  # softmax
    # },
    {   'exp_id': "'whole_shuffle'",
        'gpu_index': "2",
        'token_shuffle_mode': "'whole'",
        'token_shuffle_prob': "0.5",
        'model_used': "'glove'",
        'ngram_mode': "'word'",
        'n_steps': '300_000',
        'optimizer': "'hyper-1'",
        'hyper_lr': '0.000001',
        'lr': '0.0001',
        'glove.d_embed': '512',
    },



    # {   'exp_id': "'partial_shuffle'",
    #     'gpu_index': "1",
    #     'token_shuffle_mode': "'partial'",
    #     'token_shuffle_prob': "0.5",
    #     'model_used': "'word2vec'",
    #     'ngram_mode': "'word'",
    #     'n_steps': '100_000',
    #     'optimizer': "'hyper-1'",
    #     'hyper_lr': '0.00001',
    #     'lr': '0.001',
    #     'word2vec.d_embed': '512',
    #     'word2vec.n_neg_samples': '0',  # softmax
    # },
    # {   'exp_id': "'partial_shuffle'",
    #     'gpu_index': "1",
    #     'token_shuffle_mode': "'partial'",
    #     'token_shuffle_prob': "0.5",
    #     'model_used': "'fasttext'",
    #     'ngram_mode': "'subword'",
    #     'n_steps': '100_000',
    #     'optimizer': "'hyper-1'",
    #     'hyper_lr': '0.00001',
    #     'lr': '0.001',
    #     'fasttext.d_embed': '512',
    #     'fasttext.n_neg_samples': '0',  # softmax
    # },
    {   'exp_id': "'partial_shuffle'",
        'gpu_index': "1",
        'token_shuffle_mode': "'partial'",
        'token_shuffle_prob': "0.5",
        'model_used': "'glove'",
        'ngram_mode': "'word'",
        'n_steps': '300_000',
        'optimizer': "'hyper-1'",
        'hyper_lr': '0.000001',
        'lr': '0.0001',
        'glove.d_embed': '512',
    },



    # {   'exp_id': "'full_shuffle'",
    #     'gpu_index': "2",
    #     'token_shuffle_mode': "'whole'",
    #     'token_shuffle_prob': "1.0",
    #     'model_used': "'word2vec'",
    #     'ngram_mode': "'word'",
    #     'n_steps': '100_000',
    #     'optimizer': "'hyper-1'",
    #     'hyper_lr': '0.00001',
    #     'lr': '0.001',
    #     'word2vec.d_embed': '512',
    #     'word2vec.n_neg_samples': '0',  # softmax
    # },
    # {   'exp_id': "'full_shuffle'",
    #     'gpu_index': "2",
    #     'token_shuffle_mode': "'whole'",
    #     'token_shuffle_prob': "1.0",
    #     'model_used': "'fasttext'",
    #     'ngram_mode': "'subword'",
    #     'n_steps': '100_000',
    #     'optimizer': "'hyper-1'",
    #     'hyper_lr': '0.00001',
    #     'lr': '0.001',
    #     'fasttext.d_embed': '512',
    #     'fasttext.n_neg_samples': '0',  # softmax
    # },
    {   'exp_id': "'full_shuffle'",
        'gpu_index': "2",
        'token_shuffle_mode': "'whole'",
        'token_shuffle_prob': "1.0",
        'model_used': "'glove'",
        'ngram_mode': "'word'",
        'n_steps': '300_000',
        'optimizer': "'hyper-1'",
        'hyper_lr': '0.000001',
        'lr': '0.0001',
        'glove.d_embed': '512',
    },



    # {   'exp_id': "'no_shuffle'",
    #     'gpu_index': "3",
    #     'token_shuffle_mode': "'partial'",
    #     'token_shuffle_prob': "0.0",
    #     'model_used': "'word2vec'",
    #     'ngram_mode': "'word'",
    #     'n_steps': '100_000',
    #     'optimizer': "'hyper-1'",
    #     'hyper_lr': '0.00001',
    #     'lr': '0.001',
    #     'word2vec.d_embed': '512',
    #     'word2vec.n_neg_samples': '0',  # softmax
    # },
    # {   'exp_id': "'no_shuffle'",
    #     'gpu_index': "3",
    #     'token_shuffle_mode': "'partial'",
    #     'token_shuffle_prob': "0.0",
    #     'model_used': "'fasttext'",
    #     'ngram_mode': "'subword'",
    #     'n_steps': '100_000',
    #     'optimizer': "'hyper-1'",
    #     'hyper_lr': '0.00001',
    #     'lr': '0.001',
    #     'fasttext.d_embed': '512',
    #     'fasttext.n_neg_samples': '0',  # softmax
    # },
    {   'exp_id': "'no_shuffle'",
        'gpu_index': "3",
        'token_shuffle_mode': "'partial'",
        'token_shuffle_prob': "0.0",
        'model_used': "'glove'",
        'ngram_mode': "'word'",
        'n_steps': '300_000',
        'optimizer': "'hyper-1'",
        'hyper_lr': '0.000001',
        'lr': '0.0001',
        'glove.d_embed': '512',
    },

]


def main():
    sys.argv.append('-c%s' % RUN_CONFIG_PATH)  # no space, for some reason
    for param_set in PARAM_SETS:
        update_run_config_file_with_new_model(param_set)
        runpy.run_module('run', run_name='__main__')
        print('-------------------------------------------------')
        print('Simulation finished for the following parameters:')
        pprint(param_set, width=1)
        print('-------------------------------------------------\n\n')


def update_run_config_file_with_new_model(to_update: dict):
    with open(BASE_CONFIG_PATH, 'r') as f: config_lines = f.readlines()
    for field, value in to_update.items():
        config_lines = update_field_value(config_lines, field, value)
    with open(RUN_CONFIG_PATH, 'w') as f: f.writelines(config_lines)
    

def update_field_value(config_lines: list[str],
                       field_to_update: str,
                       new_value: str
                       ) -> list[str]:
    field_line_index, field_line = identify_field_line(config_lines,
                                                       field_to_update)
    to_replace = field_line.split(' = ')[1]
    new_field_line = field_line.replace(to_replace, new_value + '\n')
    config_lines[field_line_index] = new_field_line
    return config_lines


def identify_field_line(config_lines: list[str],
                        field_to_update: str,
                        ) -> str:
    # Case for model parameter updated
    if '.' in field_to_update:
        model, field_to_update = field_to_update.split('.')
        field_line_indices = [i for i, l in enumerate(config_lines)
                              if field_to_update in l]
        model_line_index = [i for i, l in enumerate(config_lines)
                            if '[models.%s' % model in l][0]
        field_line_index = min([l for l in field_line_indices
                                if l > model_line_index])
        return field_line_index, config_lines[field_line_index]
    # Case for general parameter updated
    else:
        return [(i, l) for i, l in enumerate(config_lines)
                if field_to_update in l.split('#')[0]][0]


if __name__ == '__main__':
    main()
