import os
import pickle
import data
import data.tasks as tasks
import torchdata.datapipes.iter


class DataPipeline():
    def __init__(
        self,
        data_params: dict,
        run_params: dict,
        train_params: dict,
        model_params: dict,
    ) -> None:
        """ Initialize a general pipeline for a dataset of word/code sequences
        """
        # Data and model parameters
        self.data_fulldir = os.path.join(
            data_params['data_dir'], data_params['data_subdir'],
        )
        self.max_seq_len = data_params['max_seq_len']
        self.debug = run_params['debug']  # smaller dataset for training
        self.data_params = data_params
        self.model_params = model_params
        self.run_params = run_params

        # Load tokenizer and train it
        self.max_tokens = train_params['max_tokens_per_batch']
        self.tokenizer = self.get_tokenizer()
    
    def get_pipeline(
        self,
        task: str,
        split: str,
        shuffle: bool=False,
    ) -> torchdata.datapipes.iter.IterDataPipe:
        """ General pipeline common to all models. Specificities include how
            data is parsed after being read in the json file and how the task
            is built for the model, once the data is encoded by the tokenizer
        """
        # Print information about which pipeline and dataset is used
        print('Building %s pipeline for %s task.' % (split, task))
        if self.debug and split == 'train':
            print(' - Using validation data for training in debug mode.')
            split = 'val'
        
        # Build task-specific pipeline
        dp = data.JsonReader(self.data_fulldir, split)
        if self.data_params['subsample_mimic']:
            dp = data.MimicSubsampler(dp)
        dp = data.TokenFilter(
            dp,
            self.run_params['ngrams_to_remove'],
            self.run_params['ngrams_to_reinsert'],
        )
        dp = data.Encoder(dp, self.tokenizer)
        dp = data.TokenShuffler(
            dp,
            self.run_params['token_shuffle_prob'],
            self.run_params['token_shuffle_mode'],
        )
        dp = self.select_task_pipeline(dp, task, split)
        dp = data.CustomBatcher(dp, self.max_tokens, self.max_seq_len, shuffle)
        dp = data.TorchPadder(dp, self.tokenizer)
        return dp
        
    def select_task_pipeline(
        self,
        dp: torchdata.datapipes.iter.IterDataPipe,
        task: str,
        split: str,
    ) -> torchdata.datapipes.iter.IterDataPipe:
        """ Set the pipeline specific to the task of the model
        """
        if task == 'skipgram':
            return tasks.SkipGramMaker(dp, self.tokenizer, self.model_params)
        elif task == 'cooc':
            return tasks.CoocMaker(
                dp, self.tokenizer, self.data_fulldir, split, self.model_params,
            )
        elif task in ['lm', 'mt', 'reagent_pred_mt']:
            return tasks.LMSetter(dp, self.tokenizer)
        elif task in ['mlm', 'reagent_pred_mlm']:
            return tasks.DynamicMasker(dp, self.tokenizer)
        elif task == 'reagent_pred_cls':
            return dp
        else:
            raise Exception('Invalid task given to the pipeline %s' % task)
        
    def get_tokenizer(self):
        """ Load and train a tokenizer with / without ngrams
        """
        # Initialize the correct tokenizer
        valid_ngram_modes = ['word', 'subword', 'icd', 'char']
        assert self.run_params['ngram_mode'] in valid_ngram_modes,\
            'Invalid ngram mode given to the pipeline %s.' % valid_ngram_modes
        print('Creating %s tokenizer' % self.run_params['ngram_mode'])
        tokenizer = self.initialize_tokenizer()
        
        # Try to load the tokenizer, only if wanted, and return it if found
        if self.run_params['load_tokenizer']:
            print(' - Loading tokenizer from %s' % tokenizer.path)
            try:
                with open(tokenizer.path, 'rb') as tokenizer_file:
                    tokenizer = pickle.load(tokenizer_file)
                    print(' - Loaded tokenizer - voc: %s' % tokenizer.vocab_sizes)
                    return tokenizer
            except:
                print(' - Tokenizer not found, retraining it')
        
        # If tokenizer was not loaded, train the tokenizer using train dataset
        print(' - Training tokenizer using the training dataset')
        dp = data.JsonReader(self.data_fulldir, 'train')
        # if self.data_params['subsample_mimic']:  # didn't worke well with this
        #     dp = data.MimicSubsampler(dp)        # didn't worke well with this
        dp = data.TokenFilter(dp, self.run_params['ngrams_to_remove'])
        list_with_all_data = [token for sentence in dp for token in sentence]
        tokenizer.fit(list_with_all_data)
        print(' - Trained tokenizer - voc: %s' % tokenizer.vocab_sizes)
        
        # Save the trained tokenizer and return it
        if self.run_params['debug']: tokenizer.path
        os.makedirs(os.path.split(tokenizer.path)[0], exist_ok=True)
        with open(tokenizer.path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(' - Saved tokenizer at %s' % tokenizer.path)
        return tokenizer
        
    def initialize_tokenizer(self):
        """ Initialize correct tokenizer, given simulation and model parameters
        """
        if self.run_params['ngram_mode'] == 'word':
            return data.Tokenizer(
                data_dir=self.data_fulldir,
                special_tokens=self.model_params['special_tokens']
            )
        elif self.run_params['ngram_mode'] in ['subword', 'icd', 'char']:
            return data.SubWordTokenizer(
                data_dir=self.data_fulldir,
                special_tokens=self.model_params['special_tokens'],
                **self.run_params,
            )
        
