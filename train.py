import argparse
import utils
import data
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore', category=pl.utilities.warnings.PossibleUserWarning)


parser = argparse.ArgumentParser(description='Train and test model.')
parser.add_argument('--config_path', '-c', type=str, default='./config.toml')
args = parser.parse_args()
model, model_name, run_params, data_params, train_params, model_params = \
    utils.load_model_and_params_from_config(args.config_path)


class PytorchLightningWrapper(pl.LightningModule):
    def __init__(self):
        ''' Initialize a pytorch-lightning wrapper to train any model '''
        super().__init__()       
        # Load data pipeline
        self.pipeline = data.DataPipeline(data_params['data_dir'],
                                          data_params['data_subdir'],
                                          data_params['max_seq_len'],
                                          run_params['debug_mode'],
                                          run_params['ngram_len'],
                                          train_params['max_tokens_per_batch'],
                                          model_params['special_tokens'])
        
        # Update model parameters and load model
        model_params['vocab_size'] = len(self.pipeline.tokenizer.encoder)
        model_params['max_seq_len'] = data_params['max_seq_len']
        self.model = model(**model_params)
        
        # Some useful parameters for the run
        self.input_keys = set(model_params['input_keys'])
        self.label_keys = set(model_params['label_keys'])
        self.learning_rate = train_params['lr']
        
    def step(self, batch, mode):
        ''' Proceed forward pass of the mode ('train' or 'val'), compute loss
            Note: the loss function used to compute the loss is model-specific
                
        Params:
        -------
        batch: dict
            Dictionary with the batch data
        mode: str
            Whether the step is a 'train' or 'val' step

        Returns:
        --------
        dict: dictionary containing the loss (could contain more stuff)
        
        '''
        # import pdb; pdb.set_trace()
        inputs = {k: batch[k] for k in batch.keys() & self.input_keys}
        labels = {k: batch[k] for k in batch.keys() & self.label_keys}
        outputs = self.model(**inputs)
        loss = self.model.loss_fn(outputs, **labels)
        self.log('%s_loss' % mode, loss.cpu().detach())
        return {'loss': loss}  #, 'output': output.cpu().detach()}

    def training_step(self, batch, batch_idx):
        """ Perform training step and return loss (see step) """
        return self.step(batch, 'train')
            
    def validation_step(self, batch, batch_idx):
        # USEFULENESS OF VALIDATION SET? GOOD TO AVOID FITTING HYPERPARAMETERS ON THE TEST SET
        # BUT APART FROM THAT, ONCE WE HAVE VALIDATED THE HYPERPARAMETERS, WE CAN USE VALIDATION + TRAINING AS TRAINING
        # OR AT LEAST WE NEED TO EMBED ALL THE CONCEPTS (SO NOT LOOSING SOME OF THEM JUST BECAUSE WE WANT VALIDATION)
        """ Perform validation step and return loss (see step) """
        return self.step(batch, 'val')
        
    def validation_epoch_end(self, outputs):
        ''' Log metrics from the output of the last validation step
        
        Params:
        -------
        outputs: list
            List of dicts containing the outputs of the last validation step
        
        '''
        pass  # TODO: see if we implement anything here
    
    def test_step(self, batch, batch_idx):
        ''' Perform a testing step with the trained model
        
        Params:
        -------
        batch: dict
            Dictionary with the batch data of the testing dataset
        batch_idx: int
            Index of the batch
                    
        '''
        pass  # TODO: implemement the pca visualization as in glove code

    # def test_epoch_end(output):
    #     model.export_as_gensim(output)

    def get_dataloaders(self, split, shuffle):
        ''' Generic function to initialize and return a dataloader
                
        Params:
        -------
        split: str
            Split of the dataset to return. Can be 'train', 'val', or 'test'
        shuffle: bool
            Whether to shuffle the data or not (typically for training)

        Returns:
        --------
        DataLoader
            Dataloader for the specified data type
        
        '''
        pl = self.pipeline.get_pipeline(model_params['task'], split, shuffle)
        return DataLoader(dataset=pl,
                          batch_size=None,  # batch_size is set by pipeline
                          num_workers=run_params['num_workers'],
                          pin_memory=run_params['pin_memory'])

    def train_dataloader(self):
        ''' Return the training dataloader '''
        return self.get_dataloaders('train', shuffle=True)
    
    def val_dataloader(self):
        ''' Return the validation dataloader '''
        return self.get_dataloaders('val', shuffle=False)

    def test_dataloader(self):
        ''' Return the testing dataloader '''
        return self.get_dataloaders('test', shuffle=False)

    def configure_optimizers(self):
        ''' Return the optimizer and the scheduler '''
        optim_params = {'params': self.parameters(),
                        'lr': self.learning_rate,
                        'betas': train_params['adam_betas']}
        optim = torch.optim.Adam(**optim_params)
        sched_params = {'optimizer': optim,
                        'd_embed': model_params['d_embed'],
                        'n_warmup_steps': train_params['n_warmup_steps']}
        sched = utils.NoamLRLambda(**sched_params)
        sched_dict = {'scheduler': sched, 'interval': 'step', 'frequency': 1}
        return [optim], [sched_dict]


def main():
    # Load pytorch lightning model-data wrapper
    model_data_wrapper = PytorchLightningWrapper()
    
    # Load checkpoint path if needed (and set to None if not found)
    ckpt_path = utils.load_checkpoint(model_name, run_params['load_model'])

    # Set environment
    accelerator, devices = utils.set_environment(run_params['num_workers'])
    
    # Callbacks for logging and checkpointing
    callbacks = [LearningRateMonitor(logging_interval='step'),
                 ModelCheckpoint(monitor=None,
                                 mode='min',
                                 every_n_train_steps=0,
                                 every_n_epochs=1,
                                 train_time_interval=None,
                                 save_on_train_epoch_end=None)]
    
    # Set a logger to monitor progress on tensorboard
    logger = pl.loggers.TensorBoardLogger(save_dir='logs/',
                                          name=model_name,
                                          version=0)

    # Set a trainer to train the model
    trainer = pl.Trainer(default_root_dir='logs',
                         accelerator=accelerator,
                         devices=devices,
                         accumulate_grad_batches= \
                            train_params['accumulate_grad_batches'],
                         gradient_clip_val=0.0,
                         num_sanity_val_steps=0,
                         log_every_n_steps=10,
                         max_steps=train_params['n_steps'],
                         callbacks=callbacks,
                         logger=logger)
    
    # Train model
    trainer.fit(model_data_wrapper, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
