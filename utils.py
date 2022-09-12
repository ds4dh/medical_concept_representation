import os
import shutil
import toml
import warnings
import models
import torch
from torch.optim.lr_scheduler import _LRScheduler


def load_model_and_params_from_config(config_path):
    ''' Load model, training and data parameters
    
    Params:
    -------
        config_path (str): path to the config file (.toml)
    
    Returns:
    --------
        model (nn.Module child): the model that is used for this run
        model_name (str): a unique identifier for the model
        run_params (dict): general parameters about the simulation
        data_params (dict): parameters about where the data is, etc.
        train_params (dict): learning, optimizers, etc.
        model_params (dict): parameters of the model that is being trained
    
    '''
    assert '.toml' in config_path, 'Config path should have .toml extension.'
    config = toml.load(config_path)   
    run_params = config['run']
    data_params = config['data']
    train_params = config['train']
    ngram_str = 'ngram' + str(run_params['ngram_len'])
    model_used = run_params['model_used']
    model_name = '_'.join([model_used, ngram_str, run_params['model_id']])
    assert(model_used) in models.AVAILABLE_MODELS.keys(), f'Selected model \
        not available. Available models: {models.AVAILABLE_MODELS.keys()}'
    model = models.AVAILABLE_MODELS[model_used]
    model_params = config['models'][model_used]
    return model, model_name, run_params, data_params, train_params, model_params


def load_checkpoint(model_name, model_version, load_model, **kwargs):
    """ Try to load a model checkpoint from the log directory
        Note: if checkpoint is not found, returns None and start from scratch
        Note: if a load_model == False, the logs of this model will be erased
        TODO: change this using version numbers, to avoid erasing automatically
    
    Args:
        model_name (str): name that identifies the model uniquely
        load_model (bool): whether the model uses a checkpoint
            or the model is trained from scratch
        version (int): number used to save different runs of a model
        
    Returns:
        str: path to model checkpoint if existing, else None
    """
    model_dir = os.path.join('logs', model_name, f'version_{model_version}')
    if load_model:
        try:
            ckpt_dir = os.path.join(model_dir, 'checkpoints')
            ckpt_name = [p for p in os.listdir(ckpt_dir) if 'ckpt' in p][-1]
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            print(f'Checkpoint found at {model_dir} and loaded')
        except IndexError:
            print(f'No checkpoint found at {model_dir}, starting from scratch')
            ckpt_path = None
        except FileNotFoundError:
            print(f'Model folder {model_dir} not found, starting from scratch')
            os.makedirs(model_dir)
            ckpt_path = None
    else:
        directory_not_empty = True
        while directory_not_empty:
            if os.path.exists(os.path.join(model_dir, 'checkpoints')):
                print(f'Data found at {model_dir}. Increasing version number')
                model_dir = model_dir.replace(f'version_{model_version}',
                                              f'version_{model_version + 1}')
                model_version += 1
            else:
                print(f'Creating folder {model_dir} and starting from scratch')
                break
        os.makedirs(model_dir, exist_ok=True)
        ckpt_path = None
    return ckpt_path, model_version


def update_and_save_config(config_path, run_params, model_name, new_model_version):
    # Find config paths
    old_config_path = config_path
    new_logdir = os.path.join('logs', model_name, f'version_{new_model_version}')
    new_config_path = os.path.join(new_logdir, 'config.toml')
    
    # Find model versions and update run parameters in case of later use
    old_model_version = run_params['model_version']
    run_params['model_version'] = new_model_version
    
    # Set version string to replace in the new config file
    to_replace = f'model_version = {old_model_version}'
    replace_by = f'model_version = {new_model_version}'
    
    # Write config file for this run
    with open(new_config_path, 'w') as new_config_file:
        with open(old_config_path, 'r') as old_config_file:
            for line in old_config_file:
                if to_replace in line:
                    line = line.replace(to_replace, replace_by)
                new_config_file.write(line)
                
            
def set_environment(num_workers):
    """ Update environment if needed and check how many gpu can be used 

    Params:
    -------
        num_workers (int): how many cpu-workers are used in the run

    Returns:
    --------
        n_gpus_to_use (int): how many gpus can be used

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
