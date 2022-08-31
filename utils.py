import warnings
import matplotlib
matplotlib.use('Agg')
from torch.optim.lr_scheduler import _LRScheduler


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
            warnings.warn("To get the last learning rate computed by the"
                          "scheduler, please use `get_last_lr()`.",
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
