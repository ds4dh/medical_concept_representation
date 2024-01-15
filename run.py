import os
import argparse
import models
import data
import metrics
import train_utils
import wandb
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import warnings
warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings("ignore", category=UserWarning)


DEFAULT_CONFIG_PATH = os.path.join("configs", "base_config.toml")
PARSER = argparse.ArgumentParser(description="Train and test model.")
PARSER.add_argument("-c", "--config", type=str, default=DEFAULT_CONFIG_PATH)
PARSER.add_argument("-t", "--test_only", action="store_true")
ARGS = PARSER.parse_args()
TEST_MODE = ARGS.test_only
MODEL, RUN_PARAMS, DATA_PARAMS, TRAIN_PARAMS, MODEL_PARAMS = \
    models.load_model_and_params_from_config(ARGS.config)


class PytorchLightningWrapper(pl.LightningModule):
    def __init__(self):
        """ Initialize a pytorch-lightning that wraps data and model.
            This allows to have data-dependent models (e.g., vocab_size)
        """
        super().__init__()
        # Load data pipeline
        self.pipeline = data.DataPipeline(
            DATA_PARAMS, RUN_PARAMS, TRAIN_PARAMS, MODEL_PARAMS,
        )
        
        # Update model parameters and load model
        MODEL_PARAMS["vocab_sizes"] = self.pipeline.tokenizer.vocab_sizes
        MODEL_PARAMS["max_seq_len"] = DATA_PARAMS["max_seq_len"]
        self.model = MODEL(**MODEL_PARAMS)
        
        # Some useful parameters for the run
        self.automatic_optimization = ("hyper" not in TRAIN_PARAMS["optimizer"])
        self.input_keys = set(MODEL_PARAMS["input_keys"])
        self.label_keys = set(MODEL_PARAMS["label_keys"])
        
    def configure_optimizers(self):
        """ Return the optimizer and the scheduler
        """
        optim = train_utils.select_optimizer(self.model, TRAIN_PARAMS)
        if "hyper" in TRAIN_PARAMS["optimizer"]:
            self.hyper_optim_wrapper, dummy_optimizer = optim
            return [dummy_optimizer]  # for lightning automatisms to be performed
        sched = train_utils.select_scheduler(optim, TRAIN_PARAMS)
        sched_dict = {"scheduler": sched, "interval": "step", "frequency": 1}
        return [optim], [sched_dict]
    
    def step(self, batch, batch_idx, mode):
        """ Proceed forward pass of the mode ("train" or "val"), compute loss
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
        self.log("loss/%s" % mode, loss.cpu().detach(), batch_size=btch_sz)
        return {"loss": loss}
    
    def hyper_optim_step(self, batch, batch_idx):
        """ Proceed hyper-optimization training step with gdtu-optimizer
        """
        # Retrieve inputs and labels
        inputs = {k: batch[k] for k in batch.keys() & self.input_keys}
        labels = {k: batch[k] for k in batch.keys() & self.label_keys}
        
        # Initialize model wrapper and compute model loss
        self.hyper_optim_wrapper.begin()
        outputs = self.hyper_optim_wrapper.forward(inputs)  # input_dict
        loss = self.model.loss_fn(outputs, **labels)
        
        # Perform hyper-optimization with gdtuo python package
        self.hyper_optim_wrapper.zero_grad()
        self.manual_backward(loss, create_graph=True)
        self.hyper_optim_wrapper.step()
        
        # Dummy optimization step to update some pytorch lightning variables
        self.optimizer_step(self.current_epoch, batch_idx, self.optimizers())
        
        # Log loss and return it to the pl-module
        btch_sz = inputs[list(inputs.keys())[0]].size(0)
        for k, v in self.hyper_optim_wrapper.optimizer.parameters.items():
            self.log("hyper/%s" % k, v)
        self.log("loss/train", loss.cpu().detach(), batch_size=btch_sz)
        return {"loss": loss}
    
    def training_step(self, batch, batch_idx):
        """ Perform training step and return loss (see step)
        """
        if "hyper" in TRAIN_PARAMS["optimizer"]:
            return self.hyper_optim_step(batch, batch_idx)
        else:
            return self.step(batch, batch_idx, "train")
        
    def validation_step(self, batch, batch_idx):
        """ Perform validation step and return loss (see step)
        """
        return self.step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        """ Perform a testing step with the trained model
        """
        return self.step(batch, batch_idx, "test")
        
    def on_test_epoch_start(self):
        metric_params = {
            "model": self.model,
            "pipeline": self.pipeline,
            "logger": self.logger,
            "global_step": self.global_step
        }
        metrics.visualization_task(**metric_params)  # for Figure 4
        metrics.hierachization_task(**metric_params)  # for Figure 5
        metrics.outcomization_task(**metric_params)  # for Figure 7
        metrics.prediction_task(**metric_params)  # for Figure 6 and 8
        metrics.trajectorization_task(**metric_params)  # for Figure 9
        
    def get_dataloaders(self, split, shuffle):
        """ Generic function to initialize and return a dataloader
        """
        pl = self.pipeline.get_pipeline(MODEL_PARAMS["task"], split, shuffle)
        if split == "test":  # test set loaded during "on_test_epoch_start" tasks
            for sample in pl:
                pl = [sample]
                break
        return DataLoader(
            dataset=pl,
            batch_size=None,  # batch_size is set by pipeline
            num_workers=RUN_PARAMS["num_workers"],
            pin_memory=RUN_PARAMS["pin_memory"],
        )
                
    def train_dataloader(self):
        """ Return the training dataloader
        """
        return self.get_dataloaders("train", shuffle=True)
    
    def val_dataloader(self):
        """ Return the validation dataloader
        """
        return self.get_dataloaders("val", shuffle=False)
    
    def test_dataloader(self):
        """ Return the testing dataloader
        """
        return self.get_dataloaders("test", shuffle=False)


def setup():
    """ Wrap a pytorch-ligthning module around a model and training data, then
        define a trainer object and a checkpoint directory to train and test
    """
    # Set save directory and logger
    save_dir = os.path.join(
        RUN_PARAMS["log_dir"], RUN_PARAMS["exp_id"], MODEL_PARAMS["model_name"],
    )
    logger = pl.loggers.WandbLogger(
        save_dir=save_dir, name=MODEL_PARAMS["model_name"],
    )
    
    # Load PL model-data wrapper, set environment and callbacks
    model_data_wrapper = PytorchLightningWrapper()
    accelerator, devices = models.set_environment(RUN_PARAMS)
    callbacks = train_utils.select_callbacks(TRAIN_PARAMS)
    
    # Set a trainer to train the model
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        num_sanity_val_steps=0,
        accumulate_grad_batches=TRAIN_PARAMS["accumulate_grad_batches"],
        gradient_clip_val=0.0,
        max_epochs=TRAIN_PARAMS["n_epochs"],
        max_steps=TRAIN_PARAMS["n_steps"],
        check_val_every_n_epoch=None,
        val_check_interval=TRAIN_PARAMS["n_steps_check_val"],
        log_every_n_steps=10,
        callbacks=callbacks,
        logger=logger,
    )
    
    # Return all objects required for training
    return {
        "trainer": trainer,
        "model_data_wrapper": model_data_wrapper,
        "save_dir": save_dir,
    }


def train(
    trainer: pl.Trainer,
    model_data_wrapper: pl.LightningModule,
    save_dir: str,
):
    """ Train a model and save the best model to a checkpoint
    """
    print("Training model (will be saved to %s)" % save_dir)
    trainer.fit(model_data_wrapper)
    print("Trained model saved to %s" % save_dir)
    

def test(
    trainer: pl.Trainer,
    model_data_wrapper: pl.LightningModule,
    save_dir: str,
):
    """ Test a model using checkpoint available in save_dir
    """
    print("Testing model at %s" % save_dir)
    logs_dir = os.path.join(save_dir, "lightning_logs")
    ckpt_dirs = os.listdir(logs_dir); assert len(ckpt_dirs) == 1
    ckpt_dir = os.path.join(logs_dir, ckpt_dirs[0], "checkpoints")
    ckpt_paths = os.listdir(ckpt_dir); assert len(ckpt_paths) == 1
    ckpt_path = os.path.join(ckpt_dir, ckpt_paths[0])
    trainer.test(model_data_wrapper, ckpt_path=ckpt_path)
    

if __name__ == "__main__":
    """ Train or test a model, depending on TEST_MODE argument
    """
    setup_output = setup()
    if not TEST_MODE:
        train(**setup_output)
    else:
        test(**setup_output)
    wandb.finish()  # still useful?
    