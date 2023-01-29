import os
import argparse
import models
import data
import metrics
import pytorch_lightning as pl
import train_utils
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import warnings
warnings.filterwarnings('ignore', category=PossibleUserWarning)
warnings.filterwarnings('ignore', category=UserWarning)


DEFAULT_CONFIG_PATH = os.path.join('config', 'run_config.toml')
PARSER = argparse.ArgumentParser(description='Train and test model.')
PARSER.add_argument('--config', '-c', type=str, default=DEFAULT_CONFIG_PATH)
ARGS = PARSER.parse_args()
MODEL, RUN_PARAMS, DATA_PARAMS, TRAIN_PARAMS, MODEL_PARAMS = \
    models.load_model_and_params_from_config(ARGS.config)


class PytorchLightningWrapper(pl.LightningModule):
    def __init__(self):
        """ Initialize a pytorch-lightning that wraps data and model.
            This allows to have data-dependent models (e.g., vocab_size)
        """
        super().__init__()
        # Load data pipeline
        self.pipeline = data.DataPipeline(DATA_PARAMS,
                                          RUN_PARAMS,
                                          TRAIN_PARAMS,
                                          MODEL_PARAMS)
        
        # Update model parameters and load model
        MODEL_PARAMS['vocab_sizes'] = self.pipeline.tokenizer.vocab_sizes
        MODEL_PARAMS['max_seq_len'] = DATA_PARAMS['max_seq_len']
        self.model = MODEL(**MODEL_PARAMS)
        
        # Some useful parameters for the run
        self.automatic_optimization = (TRAIN_PARAMS['optimizer'] != 'gdtuo')
        self.input_keys = set(MODEL_PARAMS['input_keys'])
        self.label_keys = set(MODEL_PARAMS['label_keys'])
    
    def configure_optimizers(self):
        """ Return the optimizer and the scheduler
        """
        optim = train_utils.select_optimizer(self.model, TRAIN_PARAMS)
        if TRAIN_PARAMS['optimizer'] == 'gdtuo':
            self.gdtuo_wrapper, dummy_optimizer = optim
            return [dummy_optimizer]  # for automatisms to be performed
        sched = train_utils.select_scheduler(optim, TRAIN_PARAMS)
        sched_dict = {'scheduler': sched, 'interval': 'step', 'frequency': 1}
        return [optim], [sched_dict]
    
    def step(self, batch, batch_idx, mode):
        """ Proceed forward pass of the mode ('train' or 'val'), compute loss
            Note: the loss function used to compute the loss is model-specific       
        """            
        # Retrieve inputs and labels
        inputs = {k: batch[k] for k in batch.keys() & self.input_keys}
        labels = {k: batch[k] for k in batch.keys() & self.label_keys}
        
        # Compute model loss
        outputs = self.model(inputs)
        loss = self.model.loss_fn(outputs, **labels)
        
        # Log loss and return it to the pl-module
        btch_sz = inputs[list(inputs.keys())[0]].size(0)
        self.log('%s_loss' % mode, loss.cpu().detach(), batch_size=btch_sz)
        return {'loss': loss}
    
    def gdtuo_step(self, batch, batch_idx):
        """ Proceed hyper-optimization training step with gdtu-optimizer
        """
        # Retrieve inputs and labels
        inputs = {k: batch[k] for k in batch.keys() & self.input_keys}
        labels = {k: batch[k] for k in batch.keys() & self.label_keys}
        
        # Initialize model wrapper and compute model loss
        self.gdtuo_wrapper.begin()
        outputs = self.gdtuo_wrapper.forward(inputs)  # input_dict
        loss = self.model.loss_fn(outputs, **labels)
        
        # Perform hyper-optimization with gdtuo
        self.gdtuo_wrapper.zero_grad()
        self.manual_backward(loss, create_graph=True)
        self.gdtuo_wrapper.step()
        
        # Dummy optimization step to update some pytorch lightning variables
        self.optimizer_step(self.current_epoch, batch_idx, self.optimizers())
        
        # Log loss and return it to the pl-module
        btch_sz = inputs[list(inputs.keys())[0]].size(0)
        for k, v in self.gdtuo_wrapper.optimizer.parameters.items():
            self.log('gdtuo-%s' % k, v)
        self.log('train_loss', loss.cpu().detach(), batch_size=btch_sz)
        return {'loss': loss}

    def training_step(self, batch, batch_idx):
        """ Perform training step and return loss (see step)
        """
        if TRAIN_PARAMS['optimizer'] == 'gdtuo':
            return self.gdtuo_step(batch, batch_idx)
        else:
            return self.step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        """ Perform validation step and return loss (see step)
        """
        return self.step(batch, batch_idx, 'val')
        
    def validation_epoch_end(self, outputs):
        """ Log metrics from the output of the last validation step
        """
        pass  # TODO: see if we implement anything here
    
    def test_step(self, batch, batch_idx):
        """ Perform a testing step with the trained model
        """
        return self.step(batch, batch_idx, 'test')

    def test_epoch_end(self, output):
        print('Evaluating trained model')
        metrics.visualization_task_ehr(self.pipeline.tokenizer,
                                       self.model,
                                       self.logger)
        # metrics.clustering_task_ehr(self.model, self.pipeline.tokenizer)
        # metrics.prediction_task_ehr(self.model, test_data_dir)

    def get_dataloaders(self, split, shuffle):
        """ Generic function to initialize and return a dataloader
        """
        pl = self.pipeline.get_pipeline(MODEL_PARAMS['task'], split, shuffle)
        return DataLoader(dataset=pl,
                          batch_size=None,  # batch_size is set by pipeline
                          num_workers=RUN_PARAMS['num_workers'],
                          pin_memory=RUN_PARAMS['pin_memory'])

    def train_dataloader(self):
        """ Return the training dataloader
        """
        return self.get_dataloaders('train', shuffle=True)
    
    def val_dataloader(self):
        """ Return the validation dataloader
        """
        return self.get_dataloaders('val', shuffle=False)

    def test_dataloader(self):
        """ Return the testing dataloader
        """
        return self.get_dataloaders('test', shuffle=False)


def main():
    """ Wrap a pytorch-ligthning module around a model and the corresponding
        data, and train the model to perform a model-specific task
    """
    # Load checkpoint path if needed (set to None if no checkpoint)
    ckpt_path, log_dir, new_model_version = \
        models.load_checkpoint(MODEL_PARAMS['model_name'], **RUN_PARAMS)
    
    # Update params if model_version changed and save config file to model logs
    models.update_and_save_config(ARGS.config,
                                  RUN_PARAMS,
                                  MODEL_PARAMS['model_name'],
                                  new_model_version)
    
    # Load pytorch lightning model-data wrapper
    model_data_wrapper = PytorchLightningWrapper()
    
    # Set environment
    accelerator, devices = models.set_environment(RUN_PARAMS['num_workers'])
    
    # Callbacks for logging and checkpointing
    callbacks = [ModelCheckpoint(every_n_train_steps=100),
                 EarlyStopping(monitor='val_loss', patience=4)]
    if TRAIN_PARAMS['optimizer'] != 'gdtuo':
        callbacks.extend([LearningRateMonitor(logging_interval='step')])
        
    # Set a logger to monitor progress on tensorboard
    logger = pl.loggers.TensorBoardLogger(save_dir=log_dir,
                                          name=MODEL_PARAMS['model_name'],
                                          version=new_model_version)
    
    # Set a trainer to train the model
    trainer = pl.Trainer(default_root_dir=log_dir,
                         accelerator=accelerator,
                         devices=devices,
                         num_sanity_val_steps=0,
                         accumulate_grad_batches=
                            TRAIN_PARAMS['accumulate_grad_batches'],
                         gradient_clip_val=0.0,
                         max_epochs=TRAIN_PARAMS['n_epochs'],
                         max_steps=TRAIN_PARAMS['n_steps'],
                         check_val_every_n_epoch=None,
                         val_check_interval=TRAIN_PARAMS['n_steps_check_val'],
                         log_every_n_steps=10,
                         callbacks=callbacks,
                         logger=logger)
    
    # Train, then test model
    trainer.fit(model_data_wrapper, ckpt_path=ckpt_path)
    trainer.test(model_data_wrapper, ckpt_path=ckpt_path)  # 'last')


if __name__ == '__main__':
    time_with_profiler = False
    if time_with_profiler:
        import cProfile
        import pstats
        with cProfile.Profile() as pr:
            main()
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME) 
        stats.print_stats()  # stats.dump_stats(filename='profiling.prof')
    else:
        main()
