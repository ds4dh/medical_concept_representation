import os
import json
import pandas as pd
import zipfile
import shutil

def unzip_data(zip_path, dest_dir):
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest_dir)


def pickle_to_json(pickle_dir, json_dir):
    for pkl_path in os.listdir(pickle_dir):
        if '.pickle' in pkl_path:
            print(f'Writing {pkl_path} to a json file')
            pkl_path_full = os.path.join(pickle_dir, pkl_path)
            pkl = pd.read_pickle(pkl_path_full)
            json_path = pkl_path.replace('.pickle', '.json')
            json_path_full = os.path.join(json_dir, json_path)
            with open(json_path_full, 'w') as f:
                if type(pkl) is dict:
                    json_line = json.dumps(pkl)
                    f.write(json_line + '\n')
                elif type(pkl) is list:
                    for idx, line in enumerate(pkl):
                        print('\r\t---> (line %i/%i)' % (idx, len(pkl)), end='')
                        dict_to_dump = {'text': line}
                        json_line = json.dumps(dict_to_dump)
                        f.write(json_line + '\n')
                    print()
                else:
                    raise TypeError('Pickle type not recognized.')


if __name__ == "__main__":

    zip_file_path = './json.zip'
    pickle_dir = './pickle'
    json_dir = './json'

    unzip_data(zip_file_path, './')
    shutil.rmtree(zip_file_path)
    pickle_to_json(pickle_dir, json_dir)
