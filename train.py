# This is just an example of a pytorch lightning module that we could wrap
# around any of our models. We would have flags to choose the model, etc.


import os
import shutil
import data
import models
import pytorch_lightning as pl
import torch
from torch.nn import CrossEntropyLoss as loss_fn
from torch.optim import Adam as optim_fn
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from transformers import get_cosine_schedule_with_warmup as sched_fn
from absl import app, flags


flags.DEFINE_boolean('debug', False, 'Debug mode')
flags.DEFINE_string('data_dir', data.DEFAULT_DATA_DIR, 'Data directory')
flags.DEFINE_string('tokenizer_type', 'opennmt', 'Define type of vocabulary')
flags.DEFINE_boolean('load_model', True, 'Try to load last model ckpt')
flags.DEFINE_string('data_format', 'smiles', 'Data format: smiles or selfies')
flags.DEFINE_integer('num_pop_reag', -1, 'How many reagents used in training')
flags.DEFINE_integer('n_steps', 500000, 'Number of steps for training')
flags.DEFINE_integer('max_tokens', 4096, 'Max tokens in a batch')
flags.DEFINE_integer('num_workers', 4, 'Number of workers')
flags.DEFINE_float('lr', 1e-2, 'Initial learning rate (after warm-up)')
flags.DEFINE_integer('n_warmup_steps', 8000, 'Number of warmup steps')
flags.DEFINE_integer('d_embed', 256, 'Dimension of transformer embeddings')
flags.DEFINE_integer('d_ff', 2048, 'Dimension used in feedforward layers')
flags.DEFINE_integer('n_heads', 8, 'Number of heads used in Transformer')
flags.DEFINE_integer('n_enc_layers', 4, 'Number of encoder layers')
flags.DEFINE_integer('n_dec_layers', 4, 'Number of decoder layers')
flags.DEFINE_float('dropout', 0.1, 'Dropout probability')
flags.DEFINE_boolean('find_lr', False, 'Find learning rate automatically')
FLAGS = flags.FLAGS


class LightningTransformer(pl.LightningModule):
    def __init__(self, model_params):
        ''' Initialize a lightning version of Transformer

        Params:
        -------
        model_params: dict
            Dictionary with all parameters the model needs

        '''
        super().__init__()
        # Load dataset and get data parameters
        self.datasets = data.SmilesDataset(FLAGS.data_dir,
                                           FLAGS.tokenizer_type,
                                           FLAGS.debug)
        self.pad_idx = self.datasets.tokenizer.vocab['[PAD]']
        vocab_size = len(self.datasets.tokenizer.vocab.keys())
        max_tokens = self.datasets.max_tokens

        # Update model parameters with data parameters
        model_params['src_pad_idx'] = self.pad_idx
        model_params['tgt_pad_idx'] = self.pad_idx
        model_params['src_vocab_size'] = vocab_size
        model_params['tgt_vocab_size'] = vocab_size
        model_params['max_tokens'] = max_tokens

        # Load model, learning rate and loss function
        self.model = models.PyTorchTransformer(**model_params)
        self.learning_rate = FLAGS.lr
        self.loss_fn = loss_fn

        # Load translator for validation and testing
        self.translator = data.Translator(beam_size=5, max_len=max_tokens)
        
    def forward(self, src_tokens, tgt_tokens):
        ''' Forward pass of the model
        
        Params:
        -------
        src_tokens: torch.Tensor of shape (batch_size, max_tokens)
            Source tokens
        tgt_tokens: torch.Tensor of shape (batch_size, max_tokens)
            Target tokens

        Returns:
        --------
        logits: torch.Tensor of shape (batch_size, seq_len, vocab_size)
            Output tokens of the target language (logits for any word)
        
        '''
        logits = self.model(src_tokens, tgt_tokens)
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
        loss: torch.Tensor of shape (1,)
            Loss of the batch
        logits: torch.Tensor of shape (batch_size, seq_len, vocab_size)
            Logits of the batch
        
        '''
        src_tokens = batch['src']['input_ids']
        tgt_tokens = batch['tgt']['input_ids'][:, :-1]
        gold_tokens = batch['tgt']['input_ids'][:, 1:]
        logits = self.forward(src_tokens, tgt_tokens)
        loss = self.loss_fn(logits.transpose(-2, -1), gold_tokens)
        hits = (logits.argmax(-1) == gold_tokens)
        hit_rate = hits[gold_tokens != self.pad_idx].float().mean()
        self.log('%s_loss' % mode, loss.cpu().detach())
        self.log('%s_hit_rate' % mode, hit_rate.cpu().detach())
        return loss, logits

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
        loss, _ = self.step(batch, 'train')
        return {'loss': loss}
    
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
        _, logits = self.step(batch, 'val')
        src_ids = batch['src']['input_ids']
        tgt_ids = batch['tgt']['input_ids']
        prd_ids = logits.argmax(-1)
        outputs = {'src_ids': src_ids, 'tgt_ids': tgt_ids, 'prd_ids': prd_ids}
        return outputs
    
    def validation_epoch_end(self, outputs):
        ''' Log true reactions against corresponding generated reactions
        
        Params:
        -------
        outputs: list
            List of dicts with the output of the validation steps
        
        '''
        if outputs:
            self.datasets.log_reactions(ids=outputs[0],
                                        model=self.model,
                                        translator=self.translator,
                                        writer=self.logger.experiment,
                                        global_step=self.global_step)

    def get_dataloaders(self, data_type, shuffle):
        ''' Generic function to return initialize and return a dataloader
            Note that a custom batch sampler is used for adaptive batch size
        
        Params:
        -------
        data_type: str
            Type of data to return. Can be 'train', 'val', or 'test'
        shuffle: bool
            Whether to shuffle the data or not (typically for training)

        Returns:
        --------
        DataLoader
            Dataloader for the specified data type
        
        '''
        dataset = self.datasets.get_datasets(data_type)
        collate_fn = self.datasets.get_collate_fn()
        batch_sampler = data.AdaptiveBatchSampler(dataset,
                                                  FLAGS.max_tokens,
                                                  shuffle)
        dataloader_params = {'batch_sampler': batch_sampler,
                             'num_workers': FLAGS.num_workers,
                             'collate_fn': collate_fn}
        return DataLoader(dataset=dataset, **dataloader_params)

    def train_dataloader(self):
        ''' Return the training dataloader '''
        return self.get_dataloaders('train', shuffle=True)
    
    def val_dataloader(self):
        ''' Return the validation dataloader '''
        return self.get_dataloaders('val', shuffle=True)  # False

    def test_dataloader(self):
        ''' Return the testing dataloader '''
        return self.get_dataloaders('test', shuffle=False)

    def configure_optimizers(self):
        ''' Return the optimizer and the scheduler '''
        optim_params = {'params': self.parameters(),
                        'lr': self.learning_rate,
                        'betas': (0.9, 0.998)}
        optim = optim_fn(**optim_params)
        sched_params = {'optimizer': optim,
                        'num_warmup_steps': FLAGS.n_warmup_steps,
                        'num_training_steps': FLAGS.n_steps}
        sched = sched_fn(**sched_params)
        sched_dict = {'scheduler': sched, 'interval': 'step', 'frequency': 1}
        return [optim], [sched_dict]


def main(_):
    ''' Main function to train a transformer model using pytorch-lightning
    
    Params:
    -------
    _: None
        Unused. Simply there for compatibility with the absl package.
    
    '''
    # Create the model
    model_name = f'transformer_lr_{FLAGS.lr}_tok_{FLAGS.tokenizer_type}'
    model_params = {
        'n_enc_layers': FLAGS.n_enc_layers,
        'n_dec_layers': FLAGS.n_dec_layers,
        'd_embed': FLAGS.d_embed,
        'd_ff': FLAGS.d_ff,
        'n_heads': FLAGS.n_heads,
        'dropout': FLAGS.dropout,
        'share_embeddings': True
    }
    model = LightningTransformer(model_params)

    # Deal with checkpointing
    model_dir = os.path.join('logs', model_name)
    if FLAGS.load_model:
        try:
            ckpt_dir = os.path.join(model_dir, 'version_0/checkpoints')
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
    os.environ["TOKENIZERS_PARALLELISM"] = "true"  # for num_workers > 1
    n_gpus_to_use = 1 if torch.cuda.is_available() else 0
    callbacks = [LearningRateMonitor(logging_interval='step'),
                 ModelCheckpoint(monitor=None,
                                 mode='min',
                                 every_n_train_steps=0,
                                 every_n_epochs=1,
                                 train_time_interval=None,
                                 save_on_train_epoch_end=None)]
    
    # Set trainer object
    trainer = pl.Trainer(default_root_dir='logs',
                         gpus=n_gpus_to_use,
                         auto_lr_find=FLAGS.find_lr,
                         accumulate_grad_batches=4,
                         gradient_clip_val=0.0,
                         limit_train_batches=20000,
                         limit_val_batches=2000,
                         num_sanity_val_steps=2,
                         val_check_interval=0.5,
                         log_every_n_steps=10,
                         max_steps=FLAGS.n_steps,
                         callbacks=callbacks,
                         logger=pl.loggers.TensorBoardLogger(save_dir='logs/',
                                                             name=model_name,
                                                             version=0))
    
    # Train model
    trainer.tune(model)
    trainer.fit(model, ckpt_path=ckpt_path)


if __name__ == '__main__':
    app.run(main)
