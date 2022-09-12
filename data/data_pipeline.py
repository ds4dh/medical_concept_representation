import os
import data
from tqdm import tqdm
from torchdata.datapipes.iter import Batcher, Shuffler


class DataPipeline():
    """ General pipeline for a dataset of word / code sequences """
    def __init__(self, data_dir, data_subdir, max_seq_len, debug,
                 ngram_len, max_tokens_per_batch, special_tokens):
        # Data parameters
        self.data_dir = data_dir
        self.data_subdir = data_subdir
        self.data_fulldir = os.path.join(data_dir, data_subdir)
        self.max_seq_len = max_seq_len
        self.debug = debug  # will take a smaller dataset for training
        
        # Load tokenizer and train it
        self.use_subwords = ngram_len > 0
        self.max_tokens = max_tokens_per_batch
        self.special_tokens = special_tokens
        self.tokenizer = self.get_tokenizer(ngram_len, special_tokens)
    
    def skipgram_pipeline(self, split, shuffle=False):
        """ Pipeline for skipgram task (center to context token prediction).
        
            1) Read the correct file and yield lines
            2) Encode each sample word / code using the specified tokenizer
            3) Compute and yield center-context pairs from the downsampled file
            4) Shuffle sample order if wanted
            5) Batch samples
                - for ngram-tokenization, dynamic batch size and padded samples
                - for normal tokenization, fixed batch size is fixed
            6) Shuffle batch order if wanted
            7) Build and yield torch.tensor from each batch
            
        """
        dp = data.JsonReader(self.data_fulldir, split)
        dp = data.Encoder(dp, self.tokenizer)
        dp = data.SkipGramMaker(dp, self.tokenizer, self.data_fulldir, split)
        if shuffle: dp = Shuffler(dp)
        if self.use_subwords:
            dp = data.DynamicBatcher(dp, self.max_tokens, self.max_seq_len)
        else:
            dp = Batcher(dp, batch_size=self.max_tokens//2)  # center, context
        dp = data.DictUnzipper(dp)
        if shuffle: dp = Shuffler(dp)
        dp = data.TorchPadder(dp, self.tokenizer)
        return dp
    
    def cooc_pipeline(self, split, shuffle=False):
        """ Pipeline for glove task (token co-occurrence prediction).
        
            1) Read the correct file and yield lines
            2) Encode each sample word / code using the specified tokenizer
            3) Compute and format co-occurrence matrix from the file
            4) Shuffle sample order if wanted
            5) Batch samples
                - for ngram-tokenization, dynamic batch size and padded samples
                - for normal tokenization, batch size is fixed
            6) Shuffle batch order if wanted
            7) Build and yield torch.tensor from each batch
            
        """
        dp = data.JsonReader(self.data_fulldir, split)
        dp = data.Encoder(dp, self.tokenizer)
        dp = data.CoocMaker(dp, self.tokenizer, self.data_fulldir, split)
        if shuffle: dp = Shuffler(dp)
        if self.use_subwords:
            dp = data.DynamicBatcher(dp, self.max_tokens, self.max_seq_len)
        else:
            dp = Batcher(dp, batch_size=self.max_tokens//3)  # left, right, cooc
        dp = data.DictUnzipper(dp)
        if shuffle: dp = Shuffler(dp)
        dp = data.TorchPadder(dp, self.tokenizer)
        return dp
        

    def mlm_pipeline(self, split, shuffle=False):
        """ Pipeline for mlm task (information retrieval task, using masking).
        
            1) Read the correct file and yield lines
            2) Encode each sample word / code using the specified tokenizer
            3) Mask a given proportion of input tokens, replace them by mask_id
            4) Shuffle sample order if wanted
            5) Batch samples dynamically (always needed)
            6) Shuffle batch order if wanted
            7) Build and yield torch.tensor from each batch
            
        """
        dp = data.JsonReader(self.data_fulldir, split)
        dp = data.Encoder(dp, self.tokenizer)
        dp = data.DynamicMasker(dp, self.tokenizer)
        if shuffle: dp = Shuffler(dp)
        dp = data.DynamicBatcher(dp, self.max_tokens, self.max_seq_len)
        dp = data.DictUnzipper(dp)
        if shuffle: dp = Shuffler(dp)
        dp = data.TorchPadder(dp, self.tokenizer)
        return dp
        
    def get_pipeline(self, task, split, shuffle=False):
        """ Function that selects the wanted pipeline for the correct split """
        # Information about which pipeline and dataset is used
        print(f'Building {split} pipeline for {task} task.')
        if self.debug and split == 'train':
            print(' - Using validation data for training in debug mode.')
            split = 'val'

        # Send the correct pipeline
        if task == 'skipgram':
            return self.skipgram_pipeline(split, shuffle)
        elif task == 'cooc':
            return self.cooc_pipeline(split, shuffle)
        elif task == 'mlm':
            return self.mlm_pipeline(split, shuffle)
        else:
            raise Exception('Invalid task given to the pipeline.')
    
    def get_tokenizer(self, ngram_len, special_tokens):
        """ Function that loads and train a tokenizer with / without ngrams """
        # Load the tokenizer
        if ngram_len == 0:
            tokenizer = data.Tokenizer(special_tokens)
        elif ngram_len > 0:
            tokenizer = data.SubWordTokenizer(ngram_len, special_tokens)
        else:
            raise Exception('Invalid ngram length given to the pipeline.')
        
        # Train the tokenizer with the training data (validation if debug mode)
        tokenizer_training_batches = []
        split = 'val' if self.debug else 'train'
        for batch in tqdm(data.JsonReader(self.data_fulldir, split),
                          desc='Building data to train tokenizer'):
            tokenizer_training_batches.extend(batch)
        tokenizer.fit(tokenizer_training_batches)

        return tokenizer
