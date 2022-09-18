import os
import json
import toml
import warnings
import models
import torch
from torch.optim.lr_scheduler import _LRScheduler


def load_model_and_params_from_config(config_path):
    ''' Load model, training and data parameters, using a config file
    
    Params:
    -------
        config_path (str): path to the config file (.toml)
    
    Returns:
    --------
        model (nn.Module child): the model that is used for this run
        model_name (str): unique identifier for the model
        run_params (dict): general parameters about the simulation
        data_params (dict): parameters about where the data is, etc.
        train_params (dict): learning, optimizers, etc.
        model_params (dict): parameters of the model that is being trained
    
    '''
    # Load configuration parameters
    assert '.toml' in config_path, 'Config path should have .toml extension.'
    config = toml.load(config_path)
    run_params = config['run']
    data_params = config['data']
    train_params = config['train']
    
    # Check ngram length and set unique model name for logs directory
    assert run_params['ngram_min_len'] <= run_params['ngram_max_len'] and \
           run_params['ngram_min_len'] >= 0 and \
           run_params['ngram_max_len'] >= 0, \
        'Invalid ngram min and max length (max < min or any < 0).'
    ngram_str = 'ngram-min%s-max%s' % (run_params['ngram_min_len'],
                                       run_params['ngram_max_len'])
    model_used = run_params['model_used']
    model_name = '_'.join([model_used, ngram_str, run_params['model_id']])
    
    # Retrieve corresponding model code
    assert(model_used) in models.AVAILABLE_MODELS.keys(), 'Selected model ' +\
        f'not available. Select from: {list(models.AVAILABLE_MODELS.keys())}.'
    model = models.AVAILABLE_MODELS[model_used]
    model_params = config['models'][model_used]
    model_params['model_name'] = model_name
    
    # Update bert classifier parameters if used
    if model_used == 'bert_classifier':
        update_class_info(data_params, model_params)
        update_bert_parameters(config, model_name, model_params)
    return model, run_params, data_params, train_params, model_params


def update_class_info(data_params, model_params):
    """ Update data and bert classifier parameters with class information
        TODO: put all this (specific to task, e.g., here, reagent prediction)
        TODO: in data/task/parsing (but hard because model loss_fn need update)
    """
    data_dir = os.path.join(data_params['data_dir'], data_params['data_subdir'])
    with open(os.path.join(data_dir, 'reagent_popularity.json'), 'r') as f:
        
        # Reach reagent information
        dicts = [json.loads(line) for line in f.readlines()]
        model_params['pos_weights'] = [d['weight'] for d in dicts]
        
        # Case where only k most popular reagents are classified
        if model_params['n_classes'] != 0:
            model_params['pos_weights'] = \
                model_params['pos_weights'][:model_params['n_classes']]
                
        # Case where all reagents are classified
        else:
            model_params['n_classes'] = len(dicts)
            

def update_bert_parameters(config, model_name, model_params):
    """ Update parameters of bert classifier with parameters of bert    
    """
    # Try to find the corresponding BERT model ckpt for the BERT classifier
    if model_params['load_pretrained_bert']:
        version = 'version_%s' % config['run']['model_version']
        bert_dir = os.path.join('logs', model_name, version, 'checkpoints')
        bert_dir = bert_dir.replace('bert_classifier', 'bert')
        try:
            bert_ckpt_path = os.path.join(bert_dir, os.listdir(bert_dir)[-1])
            model_params['bert_ckpt_path'] = bert_ckpt_path
        except FileNotFoundError:
            raise FileNotFoundError('No checkpoint found to initialize BERT ' + \
                'model at %s. Note: BERT and BERT classifier ids, ' % bert_dir + \
                'versions and ngram-lengths must match.')
    else:
        model_params['bert_ckpt_path'] = None
    
    # Update BERT classifier hyper-parameters with BERT hyper-parameters
    for k, v in config['models']['bert'].items():
        if k not in model_params.keys():
            model_params[k] = v


def load_checkpoint(model_name, model_version, load_model, **kwargs):
    """ Try to find a checkpoint path for the model from the log directory.
        If no checkpoint is found, None is returned (start from scratch).
    """
    model_dir = os.path.join('logs', model_name, f'version_{model_version}')
    if load_model:
        ckpt_path = find_existing_checkpoint(model_dir)
    else:
        ckpt_path = None
        model_version = initialize_new_checkpoint_dir(model_dir, model_version)
    return ckpt_path, model_version


def find_existing_checkpoint(model_dir):
    """ Try to find checkpoint and initialize a new directory if not found
    """
    try:
        ckpt_dir = os.path.join(model_dir, 'checkpoints')
        ckpt_name = [p for p in os.listdir(ckpt_dir) if 'ckpt' in p][-1]
        print(f'Checkpoint found at {model_dir} and loaded')
        return os.path.join(ckpt_dir, ckpt_name)
    except IndexError:
        print(f'No checkpoint found at {model_dir}, starting from scratch.')
        return None
    except FileNotFoundError:
        print(f'Model folder {model_dir} not found, starting from scratch.')
        os.makedirs(model_dir, exist_ok=True)
        return None
    

def initialize_new_checkpoint_dir(model_dir, model_version):
    """ Try to initialize a new checkpoint directory but avoid overwriting
    """
    directory_not_empty = True
    while directory_not_empty:
        if os.path.exists(os.path.join(model_dir, 'checkpoints')):
            print(f'Data found at {model_dir}. Increasing version number.')
            model_dir = model_dir.replace(f'version_{model_version}',
                                          f'version_{model_version + 1}')
            model_version += 1
        else:
            print(f'Creating folder {model_dir} and starting from scratch.')
            break
    os.makedirs(model_dir, exist_ok=True)
    return model_version


def update_and_save_config(config_path, run_params, model_name, new_model_version):
    """ Update the configuration parameters and save it as a .toml file in the
        model logs folder, so that all parameters are known for that run

    Args:
        config_path (str): path to the original configuration file of the run
        run_params (dict): parameters whose model_version might be updated
        model_name (str): exact name of the model being run
        new_model_version (int): model version that might have been changed to
            avoid erasing an existing model checkpoint
            
    """
    # Find config paths
    old_config_path = config_path
    new_logdir = os.path.join('logs', model_name, f'version_{new_model_version}')
    new_config_path = os.path.join(new_logdir, 'config.toml')
    
    # Find model versions and update run parameters in case of later use
    old_model_version = run_params['model_version']
    run_params['model_version'] = new_model_version
    
    # Set version string to replace in the written config file
    to_replace = f'model_version = {old_model_version}'
    replace_by = f'model_version = {new_model_version}'  # might be the same
    
    # Write config file for this run
    with open(new_config_path, 'w') as new_config_file:
        with open(old_config_path, 'r') as old_config_file:
            for line in old_config_file:
                if to_replace in line:
                    line = line.replace(to_replace, replace_by)
                new_config_file.write(line)
                
            
def set_environment(num_workers):
    """ Update environment if needed and check how many gpu can be used 
    """
    if num_workers > 0:
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    devices = 1 if accelerator == 'gpu' else num_workers
    return accelerator, devices


class NoamLRLambda(_LRScheduler):
    ''' Initialize the NoamLRLambda scheduler.
        
        :param d_model: size of hidden model dimension
        :param factor: multiplicative factor
        :param warmup: number of warmup steps
        :param last_epoch: index of last epoch
        :param verbose: print logs

    '''
    def __init__(self,
                 optimizer,
                 d_embed,
                 n_warmup_steps=8000,
                 last_epoch=-1,
                 verbose=False):
        self.d_embed = d_embed
        self.n_warmup_steps = n_warmup_steps
        self.init_lrs = [group['lr'] for group in optimizer.param_groups]
        super(NoamLRLambda, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn('To get the last learning rate computed by the \
                          scheduler, please use get_last_lr().',
                          UserWarning)
        step = self._step_count
        to_return = []
        for init_lr in self.init_lrs:
            factor = min(step ** (-0.5), step * self.n_warmup_steps ** (-1.5))
            new_lr = init_lr * self.d_embed ** (-0.5) * factor
            to_return.append(new_lr)
        return to_return

    def _get_closed_form_lr(self):
        step = self._step_count
        to_return = []
        for base_lr in self.base_lrs:
            factor = min(step ** (-0.5), step * self.n_warmup_steps ** (-1.5))
            new_lr = base_lr * self.d_embed ** (-0.5) * factor
            to_return.append(new_lr)
        return to_return
