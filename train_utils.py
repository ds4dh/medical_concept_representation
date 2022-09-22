import warnings
import torch
from torch.optim.lr_scheduler import _LRScheduler


def select_optimizer(model_weights, train_params):
    optim_params = {'params': model_weights,
                    'lr': train_params['lr'],
                    'betas': train_params['betas'],
                    'weight_decay': train_params['weight_decay']}
    if train_params['optimizer'] == 'adam':
        optim_fn = torch.optim.Adam
    elif train_params ['optimizer'] == 'radam':
        optim_fn = torch.optim.RAdam
    elif train_params['optimizer'] == 'adamw':
        optim_fn = torch.optim.AdamW
    else:
        raise ValueError('Invalid optimizer given to the pipeline.')
    return optim_fn(**optim_params)


def select_scheduler(optimizer, model_params, train_params):
    sched_params = {'optimizer': optimizer,
                    'n_warmup_steps': train_params['n_warmup_steps']}
    if train_params['scheduler'] == 'noam':
        sched_fn = NoamSchedulerWithWarmup
        sched_params.update({'d_embed': model_params['d_embed']})
    elif train_params['scheduler'] == 'linear':
        sched_fn = LinearSchedulerWithWarmup
        sched_params.update({'n_steps': train_params['n_steps']})
    else:
        raise ValueError('Invalid scheduler given to the pipeline.')
    return sched_fn(**sched_params)


class NoamSchedulerWithWarmup(_LRScheduler):
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
        super(NoamSchedulerWithWarmup, self).__init__(optimizer,
                                                      last_epoch,
                                                      verbose)

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


class LinearSchedulerWithWarmup(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, n_steps, n_warmup_steps):
        lr_lambda = lambda s: \
            self._linear_decrease_with_warmup(s, n_warmup_steps, n_steps)
        super().__init__(optimizer=optimizer, lr_lambda=lr_lambda)
    
    def _linear_decrease_with_warmup(self, step, n_warmup_steps, n_steps):
        ramp_up = step / n_warmup_steps if n_warmup_steps != 0 else 1
        fall = (n_steps - step) / (n_steps - n_warmup_steps)
        return min(ramp_up, fall)
