import os


def main():
    config_file_path = './config.toml'
    models = ['fasttext', 'glove', 'elmo', 'bert', 'fnet']
    for model in models:
        update_config_file_with_new_model(config_file_path, model)
        os.system('python train.py')
        print('\n' * 2, 'Simulation finished for %s' % model, '\n' * 4)


def update_config_file_with_new_model(config_file_path, model):
    with open(config_file_path, 'r') as f:
        data = f.readlines()
    model_line = [l for l in data if 'model_used' in l][0]
    current_model = model_line.split(' = ')[-1].split("'")[1]
    new_model_line = model_line.replace(current_model, model)
    data = [l if l != model_line else new_model_line for l in data]
    with open(config_file_path, 'w') as f:
        f.writelines(data)


if __name__ == '__main__':
    main()
