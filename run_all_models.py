import os
import sys
import runpy
from pprint import pprint


CONFIG_DIR = os.path.abspath('configs')
BASE_CONFIG_PATH = os.path.join(CONFIG_DIR, 'base_config.toml')
RUN_CONFIG_PATH = os.path.join(CONFIG_DIR, 'run_config.toml')
PARAM_SETS = [
    {
        'exp_id': "'test'",
        'model_used': "'glove'",
        'ngram_mode': "'word'",
        'optimizer': "'hyper-1'",
        'lr': '0.001',
        'hyper_lr': '0.0001',
    },
    {
        'exp_id': "'test'",
        'model_used': "'word2vec'",
        'ngram_mode': "'word'",
        'optimizer': "'hyper-1'",
        'lr': '0.001',
        'hyper_lr': '0.0001',
    },
    {
        'exp_id': "'test'",
        'model_used': "'fasttext'",
        'ngram_mode': "'subword'",
        'optimizer': "'hyper-1'",
        'lr': '0.001',
        'hyper_lr': '0.0001',
    }
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
                       new_value: str):
    field_line = [l for l in config_lines if field_to_update in l][0]
    to_replace = field_line.split(' = ')[1]
    new_field_line = field_line.replace(to_replace, new_value + '\n')
    new_config_lines = [l if l != field_line else new_field_line
                        for l in config_lines]
    return new_config_lines


if __name__ == '__main__':
    main()
