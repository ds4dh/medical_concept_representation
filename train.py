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
warnings.filterwarnings('ignore', category=UserWarning)


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
        self.automatic_optimization = (train_params['optimizer'] != 'gdtuo')
        self.input_keys = set(model_params['input_keys'])
        self.label_keys = set(model_params['label_keys'])
        
    def step(self, batch, batch_idx, mode):
        """ Proceed forward pass of the mode ('train' or 'val'), compute loss
            Note: the loss function used to compute the loss is model-specific       
        """
        # Initialize hyper-optimization if needed
        if mode == 'train' and train_params['optimizer'] == 'gdtuo':
            self.mv.begin()
            
        # Retrieve inputs and labels and run the model
        inputs = {k: batch[k] for k in batch.keys() & self.input_keys}
        labels = {k: batch[k] for k in batch.keys() & self.label_keys}
        outputs = self.model(**inputs)
        
        # Compute loss and other metrics that may be defined in the model
        loss = self.model.loss_fn(outputs, **labels)
        returned = {'loss': loss}
        
        # Perform optimization (or hyper-optimization if needed)
        if mode == 'train' and train_params['optimizer'] == 'gdtuo':
            self.mv.zero_grad()
            self.manual_backward(loss, create_graph=True)
            self.mv.step()
            
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
        metrics.evaluate(self.pipeline.tokenizer,
                         self.model,
                         self.logger,
                         categorization_strategy='prefix_codes')
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
        optim = train_utils.select_optimizer(self.model, train_params)
        if train_params['optimizer'] == 'gdtuo':
            self.mv = optim
            return None  # will not use pytorch-lightning logic in this case
        sched = train_utils.select_scheduler(optim, train_params)
        sched_dict = {'scheduler': sched, 'interval': 'step', 'frequency': 1}
        return [optim], [sched_dict]


def main():
    """ Wrap a pytorch-ligthning module around a model and the corresponding
        data, and train the model to perform a model-specific task
    """
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
    trainer.fit(model_data_wrapper, ckpt_path=ckpt_path)
    trainer.test(ckpt_path=ckpt_path)  # trainer.test(ckpt_path='last')


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
