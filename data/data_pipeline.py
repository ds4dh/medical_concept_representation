import os
import data
from tqdm import tqdm
from torchdata.datapipes.iter import (
    UnBatcher,
    Shuffler,
)


class DataPipeline():
    """ Pipeline for a dataset consisting in a collection of token sequences.
        Each sequence can be {'src': [...], 'tgt': [...]}, or be standalone.
    """
    def __init__(self,
                 data_dir, data_subdir, data_keys, special_tokens, encoding):
        # Data parameters       
        self.data_dir = data_dir
        self.data_subdir = data_subdir
        self.data_fulldir = os.path.join(data_dir, data_subdir)
        self.data_keys = data_keys
        
        # Trained tokenizer
        self.tokenizer = self.get_tokenizer(encoding, special_tokens)
        tokenizer_training_batches = []
        for batch in tqdm(data.JsonReader(self.data_fulldir, 'val')):  # 'train'
            tokenizer_training_batches.extend(batch)
        self.tokenizer.fit(tokenizer_training_batches)
    
    def skipgram_pipeline(self, split, shuffle=False):
        # TODO: MAKE THIS A SKIPGRAM PIPELINE!!!
        # Extract samples from the selected file of the data directory
        dp = data.JsonReader(self.data_fulldir, split)
        
        # Return the final pipeline
        return dp
    
    def cooc_pipeline(self, split, shuffle=False):
        # Extract samples from the selected file of the data directory       
        dp = data.JsonReader(self.data_fulldir, split)
        dp = data.Encoder(dp, self.tokenizer, self.data_keys)
        dp = data.Glover(dp, self.data_fulldir)
        
        # # Encoding and batching (grouped by similar length, dynamic batch size)
        # dp = data.Encoder(dp, self.tokenizer, self.data_keys)
        # dp = data.DynamicBucketBatcher(dp, max_tokens=self.max_tokens)

        # # Shuffling and sending to tensors
        # if shuffle: dp = Shuffler(dp)
        # dp = data.Torcher(dp, self.data_keys)
        
        # Return the final pipeline
        return dp
    
    def mlm_pipeline(self, split, shuffle=False):
        """ Data pipeline for masked language modelling. The pipeline extracts
            data from a given split, encodes samples and batches efficiently.
            
        Params:
        -------
        split (str): split of the data to take ('train', 'val', 'test')
        shuffle (bool): whether batches are shuffled
        
        Returns:
        --------
        iterable: provide the batched token ids
        
        """
        # TODO: MAKE THIS AN MLM PIPELINE!!!
        # Extract samples from the selected file of the data directory
        dp = data.JsonReader(self.data_fulldir, split)
        dp = data.DynamicBucketBatcher(dp, max_tokens=1e5)
        
        # Encoding and batching (grouped by similar length, dynamic batch size)
        dp = data.Encoder(dp, self.tokenizer, self.data_keys)
        dp = data.DynamicBucketBatcher(dp, max_tokens=self.max_tokens)

        # Padding, shuffling and sending to tensors
        # dp = DynamicMasker(dp) --> FOR THE TODO: TYPICALLY HERE!
        dp = data.Padder(dp, self.special_ids, self.max_len, self.data_keys)
        if shuffle: dp = Shuffler(dp)
        dp = data.Torcher(dp, self.data_keys)
        
        # Return the final pipeline
        return dp

    def get_pipeline(self, task, split, shuffle=False):
        if task == 'skipgram':
            return self.skipgram_pipeline(split, shuffle)
        if task == 'cooc':
            return self.cooc_pipeline(split, shuffle)
        elif task == 'mlm':
            return self.mlm_pipeline(split, shuffle)
        else:
            raise Exception('Invalid task given to the pipeline.')
    
    def get_tokenizer(self, encoding, special_tokens):
        if encoding == 'word':
            return data.Tokenizer(special_tokens)
        elif encoding == 'subword':
            return data.SubWordTokenizer()
        else:
            raise Exception('Invalid encoding scheme given to the pipeline.')
        