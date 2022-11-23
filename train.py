import argparse
import models
import data
import metrics
import pytorch_lightning as pl
import train_utils
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import warnings
warnings.filterwarnings('ignore', category=PossibleUserWarning)


parser = argparse.ArgumentParser(description='Train and test model.')
parser.add_argument('--config_path', '-c', type=str, default='./config.toml')
args = parser.parse_args()
model, run_params, data_params, train_params, model_params = \
    models.load_model_and_params_from_config(args.config_path)


class PytorchLightningWrapper(pl.LightningModule):
    def __init__(self):
        """ Initialize a pytorch-lightning wrapper to train any model
        """
        super().__init__()
        # Load data pipeline
        self.pipeline = data.DataPipeline(data_params,
                                          run_params,
                                          train_params,
                                          model_params)
        
        # Update model parameters and load model
        model_params['vocab_sizes'] = self.pipeline.tokenizer.vocab_sizes
        model_params['max_seq_len'] = data_params['max_seq_len']
        self.model = model(**model_params)
        
        # Some useful parameters for the run
        self.input_keys = set(model_params['input_keys'])
        self.label_keys = set(model_params['label_keys'])
        
        # # THIS IS JUST TO TEST MY METRICS WITHOUT TRAINING A MODEL, WILL REMOVE
        # import os
        # metrics.clustering_task_ehr(self.model, self.pipeline.tokenizer)
        # metrics.prediction_task_ehr(self.model,
        #                             os.path.join(data_params['data_dir'],
        #                                          data_params['data_subdir']),
        #                             self.pipeline.tokenizer)
        # exit()
        
    def step(self, batch, batch_idx, mode):
        """ Proceed forward pass of the mode ('train' or 'val'), compute loss
            Note: the loss function used to compute the loss is model-specific       
        """
        # Retrieve inputs and labels and run the model
        inputs = {k: batch[k] for k in batch.keys() & self.input_keys}
        labels = {k: batch[k] for k in batch.keys() & self.label_keys}
        outputs = self.model(**inputs)
        
        # Compute loss and other metrics that may be defined in the model
        loss = self.model.loss_fn(outputs, **labels)
        returned = {'loss': loss}
        if hasattr(self.model, '%s_metric' % mode):  # should remove this
            metric_fn = getattr(self.model, '%s_metric' % mode)
            returned.update(metric_fn(logger=self.logger.experiment,
                                      batch_idx=batch_idx,
                                      step=self.global_step,
                                      outputs=outputs,
                                      **labels))
                
        # Log loss and other metrics, and return them to the pl-module
        btch_sz = inputs[list(inputs.keys())[0]].size(0)  # outputs.size(0)
        for k, v in returned.items():
            self.log('%s_%s' % (mode, k), v.cpu().detach(), batch_size=btch_sz)
        return returned
    
    def training_step(self, batch, batch_idx):
        """ Perform training step and return loss (see step)
        """
        return self.step(batch, batch_idx, 'train')
            
    def validation_step(self, batch, batch_idx):
        # USEFULENESS OF VALIDATION SET? GOOD TO AVOID FITTING HYPERPARAMETERS ON THE TEST SET
        # BUT APART FROM THAT, ONCE WE HAVE VALIDATED THE HYPERPARAMETERS, WE CAN USE VALIDATION + TRAINING AS TRAINING
        # OR AT LEAST WE NEED TO EMBED ALL THE CONCEPTS (SO NOT LOOSING SOME OF THEM JUST BECAUSE WE WANT VALIDATION)
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
        pass
        # metrics.clustering_task_ehr(self.model, self.pipeline.tokenizer)
        # metrics.prediction_task_ehr(self.model, test_data_dir)

    def get_dataloaders(self, split, shuffle):
        """ Generic function to initialize and return a dataloader
        """
        pl = self.pipeline.get_pipeline(model_params['task'], split, shuffle)
        return DataLoader(dataset=pl,
                          batch_size=None,  # batch_size is set by pipeline
                          num_workers=run_params['num_workers'],
                          pin_memory=run_params['pin_memory'])

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

    def configure_optimizers(self):
        """ Return the optimizer and the scheduler
        """
        optim = train_utils.select_optimizer(self.parameters(), train_params)
        sched = train_utils.select_scheduler(optim, train_params)
        sched_dict = {'scheduler': sched, 'interval': 'step', 'frequency': 1}
        return [optim], [sched_dict]


def main():
    """ Wrap a pytorch-ligthning module around a model and the corresponding
        data, and train the model to perform a model-specific task
    """

    import cProfile
    import pstats

    # Load checkpoint path if needed (set to None if no checkpoint)
    ckpt_path, log_dir, new_model_version = \
        models.load_checkpoint(model_params['model_name'], **run_params)
    
    # Update params if model_version changed and save config file to model logs
    models.update_and_save_config(args.config_path,
                                  run_params,
                                  model_params['model_name'],
                                  new_model_version)
    
    # Load pytorch lightning model-data wrapper
    model_data_wrapper = PytorchLightningWrapper()

    # Set environment
    accelerator, devices = models.set_environment(run_params['num_workers'])
    
    # Callbacks for logging and checkpointing
    callbacks = [LearningRateMonitor(logging_interval='step'),
                 ModelCheckpoint(monitor=None,
                                 mode='min',
                                 every_n_train_steps=0,
                                 every_n_epochs=1,
                                 train_time_interval=None,
                                 save_on_train_epoch_end=None)]
    
    # Set a logger to monitor progress on tensorboard
    logger = pl.loggers.TensorBoardLogger(save_dir=log_dir,
                                          name=model_params['model_name'],
                                          version=new_model_version)
    
    # Set a trainer to train the model
    trainer = pl.Trainer(default_root_dir=log_dir,
                         accelerator=accelerator,
                         devices=devices,
                         num_sanity_val_steps=0,
                         accumulate_grad_batches= \
                            train_params['accumulate_grad_batches'],
                         gradient_clip_val=0.0,
                         max_epochs=train_params['n_epochs'],
                         max_steps=train_params['n_steps'],
                         log_every_n_steps=10,
                         callbacks=callbacks,
                         logger=logger)
    
    # Train, then test model
    with cProfile.Profile() as pr:
        trainer.fit(model_data_wrapper, ckpt_path=ckpt_path)
        # trainer.test(ckpt_path=ckpt_path)  # trainer.test(ckpt_path='last')

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename='profiling.prof')

if __name__ == '__main__':
    main()
