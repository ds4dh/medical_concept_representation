import os
import toml
import torch
import models


def load_model_and_params_from_config(config_path: str):
    """ Load model, training and data parameters, using a config file
    
    Params:
    -------
        config_path (str): path to the config file (.toml)
    
    Returns:
    --------
        model (nn.Module child): the model that is used for this run
        run_params (dict): general parameters about the simulation
        data_params (dict): parameters about where the data is, etc.
        train_params (dict): learning, optimizers, etc.
        model_params (dict): parameters of the model that is being trained
    
    """
    # Load configuration parameters
    assert ".toml" in config_path, "Config path should have .toml extension."
    config = toml.load(config_path)
    run_params = config["run"]
    data_params = config["data"]
    train_params = config["train"]

    # Check if n_epochs or n_steps is used
    if train_params["n_epochs"] > 0:
        train_params["n_steps"] = -1
    else:
        train_params["n_epochs"] = None  # for pytorch lightning
    
    # Check ngram length and set unique model name for logs directory
    if run_params["ngram_mode"] == "subword":
        assert (run_params["ngram_min_len"] > 0 and \
                run_params["ngram_max_len"] >= run_params["ngram_min_len"]), \
                "Invalid ngram min and max length (must have max > min > 0)."
        ngram_str = "ngram-min%s-max%s" % (run_params["ngram_min_len"],
                                        run_params["ngram_max_len"])
    else:
        ngram_str = "ngram-%s" % run_params["ngram_mode"]
    model_used = run_params["model_used"]
    model_name = "_".join([model_used, ngram_str])
    
    # Retrieve corresponding model code
    assert model_used in models.AVAILABLE_MODELS.keys(),\
        "Model not found (available: %s)" % list(models.AVAILABLE_MODELS.keys())
    model = models.AVAILABLE_MODELS[model_used]
    model_params = config["models"][model_used]
    model_params["model_name"] = model_name
    
    # Send the model and all parameterss to the main script
    return model, run_params, data_params, train_params, model_params


# def load_checkpoint(model_name: str,
#                     do_test_only: bool,
#                     log_dir: str,
#                     exp_id: str,
#                     model_version: int,
#                     load_model: bool,
#                     **kwargs):
#     """ Try to find a checkpoint path for the model from the log directory.
#         If no checkpoint is found, None is returned (start from scratch).
#     """
#     log_dir = os.path.join(log_dir, exp_id)
#     model_dir = os.path.join(log_dir, model_name, f"version_{model_version}")
#     if load_model:
#         ckpt_path = find_existing_checkpoint(model_dir, do_test_only)
#     else:
#         ckpt_path = None
#         model_version = initialize_new_checkpoint_dir(model_dir, model_version)
#     return ckpt_path, log_dir, model_version


# def find_existing_checkpoint(model_dir: str, do_test_only: bool):
#     """ Try to find checkpoint and initialize a new directory if not found
#     """
#     try:
#         ckpt_dir = os.path.join(model_dir, "checkpoints")
#         ckpt_name = [p for p in os.listdir(ckpt_dir) if "ckpt" in p][-1]
#         print(f"Checkpoint found at {model_dir} and loaded")
#         return os.path.join(ckpt_dir, ckpt_name)
#     except IndexError:
#         print(f"No checkpoint folder found at {model_dir}, starting from scratch.")
#         return None
#     except FileNotFoundError as e:
#         if do_test_only:
#             print(f"No checkpoint folder found in {model_dir}, but test_only mode.")
#             raise e
#         print(f"Checkpoint not found in {model_dir}, starting from scratch.")
#         os.makedirs(model_dir, exist_ok=True)
#         return None
    

# def initialize_new_checkpoint_dir(model_dir: str, model_version: int):
#     """ Try to initialize a new checkpoint directory but avoid overwriting
#     """
#     directory_not_empty = True
#     while directory_not_empty:
#         if os.path.exists(os.path.join(model_dir, "checkpoints")):
#             print(f"Data found at {model_dir}. Increasing version number.")
#             model_dir = model_dir.replace(f"version_{model_version}",
#                                           f"version_{model_version + 1}")
#             model_version += 1
#         else:
#             print(f"Creating folder {model_dir} and starting from scratch.")
#             break
#     os.makedirs(model_dir, exist_ok=True)
#     return model_version


# def update_and_save_config(config_path: str,
#                            run_params: dict,
#                            model_name: str,
#                            new_model_version: int):
#     """ Update the configuration parameters and save it as a .toml file in the
#         model logs folder, so that all parameters are known for that run

#     Args:
#         config_path (str): path to the original configuration file of the run
#         run_params (dict): parameters whose model_version might be updated
#         model_name (str): exact name of the model being run
#         new_model_version (int): model version that might have been changed to
#             avoid erasing an existing model checkpoint
            
#     """
#     # Find config paths
#     old_config_path = config_path
#     new_config_path = os.path.join(run_params["log_dir"],
#                                    run_params["exp_id"],
#                                    model_name,
#                                    "version_%s" % new_model_version,
#                                    "run_config.toml")
    
#     # Find model versions and update run parameters in case of later use
#     old_model_version = run_params["model_version"]
#     run_params["model_version"] = new_model_version
    
#     # Set version string to replace in the written config file
#     to_replace = f"model_version = {old_model_version}"
#     replace_by = f"model_version = {new_model_version}"  # might be the same
    
#     # Write config file for this run
#     with open(new_config_path, "w") as new_config_file:
#         with open(old_config_path, "r") as old_config_file:
#             for line in old_config_file:
#                 if to_replace in line:
#                     line = line.replace(to_replace, replace_by)
#                 new_config_file.write(line)


def set_environment(run_params: int):
    """ Update environment if needed and check how many gpu can be used 
    """
    if run_params["num_workers"] > 0:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
    if torch.cuda.is_available():
        if run_params["gpu_index"] >= torch.cuda.device_count()\
        or run_params["gpu_index"] < 0:
            raise ValueError("Invalid GPU index")
        accelerator = "gpu"
        devices = [run_params["gpu_index"]]
    else:
        accelerator = "cpu"
        devices = "auto"
    return accelerator, devices


# def update_class_info_for_bert_classifier(data_params: dict,
#                                           model_params: dict):
#     """ Update data and bert classifier parameters with class information
#         Usage in load_model_and_params_from_config, just before return
#         if model_used == "bert_classifier":
#             update_class_info_for_bert_classifier(data_params, model_params)
#             update_params_for_bert_classifier(config, model_name, model_params)
    
#     """
#     data_dir = os.path.join(data_params["data_dir"], data_params["data_subdir"])
#     with open(os.path.join(data_dir, "reagent_popularity.json"), "r") as f:
        
#         # Reach reagent information
#         dicts = [json.loads(line) for line in f.readlines()]
#         model_params["pos_weights"] = [d["weight"] for d in dicts]
        
#         # Case where only k most popular reagents are classified
#         if model_params["n_classes"] != 0:
#             model_params["pos_weights"] = \
#                 model_params["pos_weights"][:model_params["n_classes"]]
                
#         # Case where all reagents are classified
#         else:
#             model_params["n_classes"] = len(dicts)
            

# def update_params_for_bert_classifier(config: dict,
#                                       model_name: str,
#                                       model_params: dict):
#     """ Update parameters of bert classifier with parameters of bert    
#     """
#     # Try to find the corresponding BERT model ckpt for the BERT classifier
#     if model_params["load_pretrained_bert"]:
#         bert_path = model_params["bert_path"]
#         if bert_path not in ["", "none"]:
#             bert_dir = os.path.join(bert_path, "checkpoints")
#             bert_config = toml.load(os.path.join(bert_path, "config.toml"))
#             bert_params = bert_config["models"]["bert"]
#         else:
#             log_dir = os.path.join("logs", config["run_params"]["exp_dir"])
#             version = "version_%s" % config["run"]["model_version"]
#             bert_dir = os.path.join(log_dir, model_name, version, "checkpoints")
#             bert_dir = bert_dir.replace("bert_classifier", "bert")
#             bert_params = config["models"]["bert"]
#         try:
#             bert_ckpt_path = os.path.join(bert_dir, os.listdir(bert_dir)[-1])
#             model_params["bert_ckpt_path"] = bert_ckpt_path
#         except FileNotFoundError:
#             raise FileNotFoundError("No checkpoint found to initialize BERT " +\
#                 "model at %s. Note: BERT and BERT classifier " % bert_dir +\
#                 "ids, versions and ngram-lengths must match.")
#     else:
#         model_params["bert_ckpt_path"] = None
#         bert_params = config["models"]["bert"]
    
#     # Update BERT classifier hyper-parameters with BERT hyper-parameters
#     bert_params = config["models"]["bert"]
#     for k, v in bert_params.items():
#         if k not in model_params.keys():
#             model_params[k] = v