import os
import data
from tqdm import tqdm
from torchdata.datapipes.iter import Batcher, Shuffler


class DataPipeline():
    """ Pipeline for a dataset consisting in a collection of word sequences """
    def __init__(self, data_dir, data_subdir, data_keys, debug,
                 max_tokens_per_batch, special_tokens, encoding):
        # Data parameters       
        self.data_dir = data_dir
        self.data_subdir = data_subdir
        self.data_fulldir = os.path.join(data_dir, data_subdir)
        self.data_keys = data_keys
        
        # Train and load tokenizer
        self.max_tokens = max_tokens_per_batch
        self.tokenizer = self.get_tokenizer(encoding, special_tokens, debug)
    
    # TODO: MAKE THIS A SKIPGRAM PIPELINE
    def skipgram_pipeline(self, split, shuffle=False):
        dp = data.JsonReader(self.data_fulldir, split)
        if shuffle: dp = Shuffler(dp)
        return data.Torcher(dp)
    
    def cooc_pipeline(self, split, shuffle=False):
        dp = data.GloveJsonReader(self.data_fulldir, split, self.tokenizer)
        dp = Batcher(dp, batch_size=self.max_tokens)
        dp = data.DictUnzipper(dp)
        if shuffle: dp = Shuffler(dp)
        return data.Torcher(dp)

    # TODO: MAKE THIS AN MLM PIPELINE
    def mlm_pipeline(self, split, shuffle=False):
        dp = data.JsonReader(self.data_fulldir, split)
        dp = data.DynamicBucketBatcher(dp, max_tokens=1e5)
        dp = data.Encoder(dp, self.tokenizer, self.data_keys)
        dp = data.DynamicBucketBatcher(dp, max_tokens=self.max_tokens)
        # dp = DynamicMasker(dp) --> FOR THE TODO: TYPICALLY HERE
        dp = data.Padder(dp, self.special_ids, self.max_len, self.data_keys)
        if shuffle: dp = Shuffler(dp)
        return data.Torcher(dp, self.data_keys)

    def get_pipeline(self, task, split, shuffle=False):
        if task == 'skipgram':
            return self.skipgram_pipeline(split, shuffle)
        if task == 'cooc':
            return self.cooc_pipeline(split, shuffle)
        elif task == 'mlm':
            return self.mlm_pipeline(split, shuffle)
        else:
            raise Exception('Invalid task given to the pipeline.')
    
    def get_tokenizer(self, encoding, special_tokens, debug=False):
        # Load the tokenizer
        if encoding == 'word':
            tokenizer = data.Tokenizer(special_tokens)
        elif encoding == 'subword':
            tokenizer = data.SubWordTokenizer()
        else:
            raise Exception('Invalid encoding scheme given to the pipeline.')
        
        # Train the tokenizer with the training data (validation if debug mode)
        print('Training tokenizer')
        tokenizer_training_batches = []
        for batch in tqdm(data.JsonReader(self.data_fulldir,
                                          split='train' if debug else 'val')):
            tokenizer_training_batches.extend(batch)
        tokenizer.fit(tokenizer_training_batches)

        return tokenizer
