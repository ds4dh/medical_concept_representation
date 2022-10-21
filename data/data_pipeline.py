import os
import data
import data.tasks as tasks
from tqdm import tqdm
from torchdata.datapipes.iter import Shuffler
from itertools import chain


class DataPipeline():
    """ General pipeline for a dataset of word / code sequences
    """
    def __init__(self, data_params, run_params, train_params, model_params):
        # Data parameters
        self.data_dir = data_params['data_dir']
        self.data_subdir = data_params['data_subdir']
        self.data_fulldir = os.path.join(self.data_dir, self.data_subdir)
        self.max_seq_len = data_params['max_seq_len']
        self.debug = run_params['debug']  # smaller dataset for training
        if 'n_classes' in model_params.keys():
            self.n_classes = model_params['n_classes']
        
        # Load tokenizer and train it
        self.max_tokens = train_params['max_tokens_per_batch']
        self.tokenizer = self.get_tokenizer(model_params, run_params)
    
    def get_pipeline(self, task, split, shuffle=False):
        """ General pipeline common to all models. Specificities include how
            data is parsed after being read in the json file and how the task
            is built for the model, once the data is encoded by the tokenizer
        """
        # Print information about which pipeline and dataset is used
        print(f'Building {split} pipeline for {task} task.')
        if self.debug and split == 'train':
            print(' - Using validation data for training in debug mode.')
            split = 'val'
        
        # Build task-specific pipeline        
        dp = data.JsonReader(self.data_fulldir, split)
        dp = self.select_parse_pipeline(dp, task)
        dp = data.Encoder(dp, self.tokenizer)
        dp = self.select_task_pipeline(dp, task, split)
        if shuffle: dp = Shuffler(dp)
        dp = data.DynamicBatcher(dp, self.max_tokens, self.max_seq_len)
        dp = data.DictUnzipper(dp)
        if shuffle: dp = Shuffler(dp)
        dp = data.TorchPadder(dp, self.tokenizer)
        return dp
    
    def select_parse_pipeline(self, dp, task):
        """ Set how data is parsed for the model after being read
        """
        if task in ['skipgram', 'cooc', 'lm', 'mlm']:
            return dp
        elif task == 'reagent_pred_mt':
            return tasks.ReagentPredParser(dp, task)
        elif task == 'reagent_pred_mlm':
            return tasks.ReagentPredParser(dp, task)
        elif task == 'reagent_pred_cls':
            return tasks.ReagentPredParser(dp,
                                           task=task,
                                           data_dir=self.data_fulldir,
                                           n_classes=self.n_classes)
        else:
            raise Exception('Invalid task given to the pipeline %s' % task)
        
    def select_task_pipeline(self, dp, task, split):
        """ Set the pipeline specific to the task of the model
        """
        if task == 'skipgram':
            return tasks.SkipGramMaker(dp, self.tokenizer, self.data_fulldir, split)
        elif task == 'cooc':
            return tasks.CoocMaker(dp, self.tokenizer, self.data_fulldir, split)
        elif task in ['mt', 'reagent_pred_mt']:
            return tasks.BosEosAdder(dp, self.tokenizer)
        elif task == 'lm':
            return tasks.ElmoSetter(dp, self.tokenizer)
        elif task in ['mlm', 'reagent_pred_mlm']:
            return tasks.DynamicMasker(dp, self.tokenizer)
        elif task == 'reagent_pred_cls':
            return dp
        else:
            raise Exception('Invalid task given to the pipeline %s' % task)
        
    def get_tokenizer(self, model_params, run_params):
        """ Load and train a tokenizer with / without ngrams
        """
        # Load the tokenizer
        if run_params['ngram_mode'] == 'word':
            tokenizer = data.Tokenizer(model_params['special_tokens'])
        elif run_params['ngram_mode'] in ['subword', 'icd', 'char']:
            tokenizer = data.SubWordTokenizer(run_params['ngram_min_len'],
                                              run_params['ngram_max_len'],
                                              run_params['ngram_mode'],
                                              run_params['ngram_base_prefixes'],
                                              run_params['ngram_base_suffixes'],
                                              model_params['special_tokens'])
        else:
            raise Exception('Invalid ngram mode given to the pipeline.')
        
        # Figure out which type of input the tokenizer should encode
        if 'tokenizer_task' in model_params:
            tokenizer_task = model_params['tokenizer_task']
        else:
            tokenizer_task = model_params['task']

        # Build a pipeline for the tokenizer, given the task of the model
        split = 'val' if self.debug else 'train'
        dp = data.JsonReader(self.data_fulldir, split)
        dp = self.select_parse_pipeline(dp, tokenizer_task)
        
        # Train the tokenizer with the training data (validation if debug mode)
        tokenizer_training_batches = []
        for sample in tqdm(dp, desc='Building data to train tokenizer'):
            if isinstance(sample, dict):  # avoid taking label indices as tokens
                sample = {k: v for k, v in sample.items() if 'label' not in k}
                sample = list(chain(*list(sample.values())))
            tokenizer_training_batches.extend(sample)
        tokenizer.fit(tokenizer_training_batches)

        return tokenizer
