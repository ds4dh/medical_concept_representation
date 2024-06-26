import warnings
import torch
from torch.optim.lr_scheduler import _LRScheduler, OneCycleLR
from gradient_descent_the_ultimate_optimizer import gdtuo
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)


def select_optimizer(model, train_params):
    # Hyper-(hyper-)optimization with gdtuo python package
    if "hyper" in train_params["optimizer"]:
        # Select hyper optimization given tower level (e.g., 2 for "hyper-2")
        hyper_optim = gdtuo.NoOpOptimizer()
        hyper_level = int(train_params["optimizer"].split("hyper-")[-1])
        hyper_lr = train_params["hyper_lr"] / (10 ** hyper_level)
        for _ in range(hyper_level):
            hyper_lr *= 10
            hyper_optim = gdtuo.AdamBaydin(alpha=hyper_lr,
                                           optimizer=hyper_optim)
        
        # Build model optimizer wrapped into hyper-optimization pipeline
        optim = gdtuo.Adam(alpha=train_params["lr"], optimizer=hyper_optim)
        hyper_optim_wrapper = gdtuo.ModuleWrapper(model, optimizer=optim)
        hyper_optim_wrapper.initialize()
        dummy_optim = torch.optim.Adam([torch.empty(0)])
        return hyper_optim_wrapper, dummy_optim

    # Classic optimization
    optim_params = {"params": model.parameters(),
                    "lr": train_params["lr"],
                    "betas": train_params["betas"],
                    "weight_decay": train_params["weight_decay"]}
    if train_params["optimizer"] == "adam":
        optim_fn = torch.optim.Adam
    elif train_params ["optimizer"] == "radam":
        optim_fn = torch.optim.RAdam
    elif train_params["optimizer"] == "adamw":
        optim_fn = torch.optim.AdamW
    else:
        raise ValueError("Invalid optimizer given to the pipeline.")
    return optim_fn(**optim_params)


def select_scheduler(optimizer, train_params):
    sched_params = {"optimizer": optimizer,
                    "n_warmup_steps": train_params["n_sched_warmup_steps"]}
    if train_params["scheduler"] == "noam":
        sched_fn = NoamSchedulerWithWarmup
    elif train_params["scheduler"] == "linear":
        sched_fn = LinearSchedulerWithWarmup
        sched_params.update({"n_steps": train_params["n_sched_steps"]})
    elif train_params["scheduler"] == "onecycle":
        return OneCycleLR(optimizer=optimizer,
                          max_lr=train_params["lr"],
                          total_steps=train_params["n_sched_steps"],
                          pct_start=0.05)
    else:
        raise ValueError("Invalid scheduler given to the pipeline.")
    return sched_fn(**sched_params)


def select_callbacks(train_params, monitored="loss/train"):
    # callbacks = [ModelCheckpoint(
    #     monitor=monitored, mode="min", save_top_k=1, every_n_train_steps=100,
    # )]
    callbacks = [ModelCheckpoint()]
    if train_params["early_stopping_patience"] > 0:
        callbacks.append(
            EarlyStopping(
                monitor=monitored,
                patience=train_params["early_stopping_patience"],
            )
        )
    if train_params["optimizer"] != "hyper":
        callbacks.extend([LearningRateMonitor(logging_interval="step")])
    return callbacks


class NoamSchedulerWithWarmup(_LRScheduler):
    """ Initialize the NoamLRLambda scheduler.
        
        :param d_model: size of hidden model dimension
        :param factor: multiplicative factor
        :param warmup: number of warmup steps
        :param last_epoch: index of last epoch
        :param verbose: print logs

    """
    def __init__(self, optimizer, n_warmup_steps=10000):
        self.n_warmup_steps = n_warmup_steps
        self.init_lrs = [group["lr"] for group in optimizer.param_groups]
        super(NoamSchedulerWithWarmup, self).__init__(optimizer)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the \
                          scheduler, please use get_last_lr().", UserWarning)
        return self._get_lr_fn(self.init_lrs)

    def _get_closed_form_lr(self):
        return self._get_lr_fn(self.base_lrs)
    
    def _get_lr_fn(self, init_lrs):
        step = self._step_count
        to_return = []
        for init_lr in init_lrs:
            factor = self._compute_factor(step)
            to_return.append(init_lr * factor)
        return to_return
    
    def _compute_factor(self, step):
        base_factor = max(self.n_warmup_steps, 1) ** 0.5
        step_factor = min(max(step, 1) ** (-0.5),
                          step * max(self.n_warmup_steps, 1) ** (-1.5))
        return base_factor * step_factor


class LinearSchedulerWithWarmup(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, n_steps, n_warmup_steps):
        lr_lambda = lambda s: \
            self._linear_decrease_with_warmup(s, n_warmup_steps, n_steps)
        super().__init__(optimizer=optimizer, lr_lambda=lr_lambda)
    
    def _linear_decrease_with_warmup(self, step, n_warmup_steps, n_steps):
        ramp_up = step / n_warmup_steps if n_warmup_steps != 0 else 1
        fall = (n_steps - step) / (n_steps - n_warmup_steps)
        return min(ramp_up, fall)
