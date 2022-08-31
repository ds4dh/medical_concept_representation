import os
import toml
import shutil
import utils
import data
import models
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


config = toml.load('config.toml')
general_params = config['general']
model_used = config['model']['model_used']
model_name = model_used + '_' + config['model']['model_id']
model_params = config['model'][model_used]
data_params = config['data']
training_params = config['training']
available_models = {'fasttext': models.FastText,
                    'my_bert': models.MyBert,
                    'pytorch_bert': models.PyTorchBert}


class PytorchLightningWrapper(pl.LightningModule):
    def __init__(self):
        ''' Initialize a lightning version of Transformer

        Params:
        -------
        model_params: dict
            Dictionary with all parameters the model needs

        '''
        super().__init__()
        # Load dataset and get data parameters
        self.data_holder = data.DataHolder(data_params['data_dir'],
                                           data_params['data_subdir'],
                                           data_params['data_keys'],
                                           data_params['special_tokens'],
                                           training_params['max_tokens'],
                                           data_params['max_len'],
                                           general_params['load_tokenizer'])
        self.pad_id = self.data_holder.special_ids['pad_id']
        self.vocab_size = len(self.data_holder.vocab.keys())
        model_params['vocab_size'] = self.vocab_size
        model_params['pad_id'] = self.pad_id
        
        # Load model, learning rate and loss function
        self.model = available_models[model_used](**model_params)
        self.learning_rate = training_params['lr']
        # self.loss_fn = nn.NLLLoss(ignore_index=self.pad_id)
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, tokens):
        ''' Forward pass of the model
        
        Params:
        -------
        tokens: torch.Tensor of shape (batch_size, max_tokens)
            Input tokens
        
        Returns:
        --------
        logits: torch.Tensor of shape (batch_size, seq_len, vocab_size)
            Output tokens of the target language (logits for any word)
        
        '''
        logits = self.model(tokens)
        return logits

    def step(self, batch, mode):
        ''' Compute loss and hit_rate and log it
        
        Params:
        -------
        batch: dict
            Dictionary with the batch data
        mode: str
            Whether the step is a 'train' or 'val' step

        Returns:
        --------
        loss
            Loss of the batch
        hit_rate
            Hit rate of the batch
        
        '''
        tokens = batch['text']
        label = batch['text'][:, 0]  # just for the example
        logits = self.forward(tokens)  # teacher forcing        
        loss = self.loss_fn(logits, label)
        acc = (logits.argmax(-1) == label).float().mean()
        self.log('%s_loss' % mode, loss.cpu().detach())
        self.log('%s_acc' % mode, acc.cpu().detach())
        return loss, logits, acc

    def training_step(self, batch, batch_idx):
        ''' Compute training loss
        
        Params:
        -------
        batch: dict
            Dictionary with the batch data
        batch_idx: int
            Index of the batch

        Returns:
        --------
        dict
            Dictionary with the training loss used for backpropagation
        
        '''
        loss, _, acc = self.step(batch, 'train')
        return {'loss': loss, 'acc': acc}
    
    def validation_step(self, batch, batch_idx):
        ''' Compute validation loss and correctly predicted tokens
        
        Params:
        -------
        batch: dict
            Dictionary with the batch data
        batch_idx: int
            Index of the batch

        Returns:
        --------
        outputs: dict
            Features that are used at the end of the validation step
        
        '''
        _, logits, _ = self.step(batch, 'val')
        tokens = batch['text']
        predictions = logits.argmax(-1)
        outputs = {'tokens': tokens, 'predictions': predictions}
        return outputs
    
    def validation_epoch_end(self, outputs):
        ''' Log true reactions against corresponding generated reactions
        
        Params:
        -------
        outputs: list
            List of dicts with the output of the validation steps
        
        '''
        if outputs:
            pass

    def get_dataloaders(self, split, shuffle):
        ''' Generic function to return initialize and return a dataloader
            Note that a custom batch sampler is used for adaptive batch size
        
        Params:
        -------
        split: str
            Split of the dataset to return. Can be 'train', 'val', or 'test'
        shuffle: bool
            Whether to shuffle the data (typically for training) or not

        Returns:
        --------
        DataLoader
            Dataloader for the specified data type
        
        '''
        pipeline = self.data_holder.pipeline(split, shuffle)
        return DataLoader(pipeline,
                          batch_size=None,
                          num_workers=training_params['num_workers'],
                          pin_memory=training_params['pin_memory'])

    def train_dataloader(self):
        ''' Return the training dataloader '''
        split = 'train' if not general_params['debug'] else 'val'
        return self.get_dataloaders(split, shuffle=True)
    
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
                        'betas': training_params['adam_betas']}
        optim = torch.optim.AdamW(**optim_params)
        sched_params = {'optimizer': optim,
                        'd_embed': model_params['d_embed'],
                        'n_warmup_steps': training_params['n_warmup_steps']}
        sched = utils.NoamLRLambda(**sched_params)
        sched_dict = {'scheduler': sched, 'interval': 'step', 'frequency': 1}
        return [optim], [sched_dict]


def main():    
    # Load pytorch lightning model-data wrapper and deal with checkpointing
    model_data_wrapper = PytorchLightningWrapper()
    model_dir = os.path.join('logs', model_used)
    if general_params['load_model']:
        try:
            ckpt_dir = os.path.join(model_dir, 'version_0/checkpoints')  # TODO: account for checkpoint versions
            ckpt_name = [p for p in os.listdir(ckpt_dir) if 'ckpt' in p][-1]
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        except (IndexError, FileNotFoundError):
            print('No checkpoint found in logs/, starting from scratch')
            ckpt_path = None
    else:
        print('Reseting logs/ and starting from scratch')
        shutil.rmtree(path=model_dir, ignore_errors=True)
        os.makedirs('logs', exist_ok=True)
        ckpt_path = None
    
    # Set environment
    if training_params['num_workers'] > 0:
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    n_gpus_to_use = 1 if torch.cuda.is_available() else 0
    callbacks = [LearningRateMonitor(logging_interval='step'),
                 ModelCheckpoint(monitor=None,
                                 mode='min',
                                 every_n_train_steps=0,
                                 every_n_epochs=1,
                                 train_time_interval=None,
                                 save_on_train_epoch_end=None)]
    
    # Set a logger to monitor progress on tensorboard
    logger = pl.loggers.TensorBoardLogger(save_dir='logs/',
                                          name=model_used,
                                          version=0)

    # Set a trainer to train the model
    trainer = pl.Trainer(default_root_dir='logs',
                         gpus=n_gpus_to_use,
                         auto_lr_find=general_params['find_best_lr'],
                         accumulate_grad_batches=4,
                         gradient_clip_val=0.0,
                         num_sanity_val_steps=0,
                         log_every_n_steps=10,
                         max_steps=training_params['n_steps'],
                         callbacks=callbacks,
                         logger=logger)
    
    # Train model
    trainer.tune(model_data_wrapper)
    trainer.fit(model_data_wrapper, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
