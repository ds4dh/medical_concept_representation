import os
import sys
import time
import select
import itertools
import json
import pandas as pd
import zipfile
from absl import app, flags


cwd = os.path.dirname(os.path.abspath(__file__))
flags.DEFINE_string('data_dir', cwd, 'Data directory')
FLAGS = flags.FLAGS


def unzip_data(zip_path, dest_dir):
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest_dir)


def pickle_to_json(pickle_dir, json_dir):
    os.makedirs(json_dir, exist_ok=True)
    for pkl_path in os.listdir(pickle_dir):
        if '.pickle' in pkl_path:
            print(f'Loading {pkl_path}...')
            pkl_path_full = os.path.join(pickle_dir, pkl_path)
            pkl = pd.read_pickle(pkl_path_full)
            json_path = pkl_path.replace('.pickle', '.json')
            json_path_full = os.path.join(json_dir, json_path)
            with open(json_path_full, 'w') as f:
                if type(pkl) is dict:
                    print('\r\t-> writing json file %s' % (json_path))
                    json_line = json.dumps(pkl)
                    f.write(json_line + '\n')
                elif type(pkl) is list:
                    for idx, line in enumerate(pkl):
                        print('\r\t-> writing (line %i/%i) in json file %s' %\
                             (idx, len(pkl), json_path), end='')
                        dict_to_dump = {'text': line}
                        json_line = json.dumps(dict_to_dump)
                        f.write(json_line + '\n')
                    print()
                else:
                    raise TypeError('Pickle type not recognized.')


def timed_input(prompt, timeout, default=None):
    sys.stdout.write(prompt + '\n')
    sys.stdout.flush()
    ready, _, _ = select.select([sys.stdin], [],[], timeout)
    if ready:
        return sys.stdin.readline().rstrip('\n')
    elif default:
        return default
    else:
        class TimeoutExpired(Exception): pass
        raise TimeoutExpired


def build_dataset(_):
    # Define data paths and directories
    data_dir = FLAGS.data_dir  # TODO: this as a parameter and not using absl
    zip_file_path = os.path.join(data_dir, 'pickle.zip')
    pickle_dir = os.path.join(data_dir, 'pickle')
    json_dir = os.path.join(data_dir, 'json')

    # Unzip data .zip file if necessary
    if os.path.exists(zip_file_path):
        if timed_input('Zip file detected. Unzip it? ([y]/n) ', 5, 'n') == 'y':
            print(f'Unzipping the zip file into {data_dir}')
            unzip_data(zip_file_path, data_dir)
        else:
            print(f'No unzipping and assuming data files are in {pickle_dir}')
    
    # Transforms the pickle files (if existing) into json files
    try:
        pickle_to_json(pickle_dir, json_dir)
    except FileNotFoundError as e:
        raise e('No pickle directory or zip file found. \
                 Zip file is available at https://drive.google.com/\
                 file/d/1NFUnnOLFuPIcrBHYVgWN_UClDlf4mXzG/view?usp=sharing')

    # Delete the data .zip file if necessary
    if timed_input('Delete zip file? (y/[n]) ', 5, 'n') == 'y':
        print('Deleting the zip file.')
        os.remove(zip_file_path)
    else:
        print('Not deleting the zip file.')


if __name__ == '__main__':
    app.run(build_dataset)
