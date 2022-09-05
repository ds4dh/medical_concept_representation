import os
import data
from tqdm import tqdm
from torchdata.datapipes.iter import Batcher, Shuffler


class DataPipeline():
    """ Pipeline for a dataset consisting in a collection of word sequences """
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
        dp = data.JsonReader(self.data_fulldir, split)
        dp = data.Encoder(dp, self.tokenizer)
        dp = data.SkipGramMaker(dp, self.tokenizer)
        if self.use_subwords:
            dp = data.DynamicBatcher(dp, self.max_tokens)
            dp = data.Padder(dp, self.special_tokens, self.max_seq_len)
        else:
            dp = Batcher(dp, batch_size=self.max_tokens//2)  # center, context
            dp = data.DictUnzipper(dp)
        if shuffle: dp = Shuffler(dp)
        return data.Torcher(dp)
    
    def cooc_pipeline(self, split, shuffle=False):
        dp = data.JsonReader(self.data_fulldir, split)
        dp = data.Encoder(dp, self.tokenizer)
        dp = data.GloveMaker(dp)
        if self.use_subwords:
            dp = data.DynamicBatcher(dp, self.max_tokens)
            dp = data.Padder(dp, self.special_tokens, self.max_seq_len)
        else:
            dp = Batcher(dp, batch_size=self.max_tokens//3)  # left, right, cooc
            dp = data.DictUnzipper(dp)
        if shuffle: dp = Shuffler(dp)
        return data.Torcher(dp)

    # TODO: MAKE THIS AN MLM PIPELINE
    def mlm_pipeline(self, split, shuffle=False):
        dp = data.JsonReader(self.data_fulldir, split)
        dp = data.DynamicBatcher(dp, max_tokens=1e5)
        dp = data.Encoder(dp, self.tokenizer)
        dp = data.DynamicBatcher(dp, max_tokens=self.max_tokens)
        # dp = DynamicMasker(dp) --> FOR THE TODO: TYPICALLY HERE
        if shuffle: dp = Shuffler(dp)
        return data.Torcher(dp)

    def get_pipeline(self, task, split, shuffle=False):
        print(f'Building {split} pipeline for {task} task.')
        if self.debug and split == 'train':
            print('Using validation data for training in debug mode.')
            split = 'val'
        if task == 'skipgram':
            return self.skipgram_pipeline(split, shuffle)
        elif task == 'cooc':
            return self.cooc_pipeline(split, shuffle)
        elif task == 'mlm':
            return self.mlm_pipeline(split, shuffle)
        else:
            raise Exception('Invalid task given to the pipeline.')
    
    def get_tokenizer(self, ngram_len, special_tokens):
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
