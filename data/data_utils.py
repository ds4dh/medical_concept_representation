import json
import pickle
import torch
from tqdm import tqdm
from functools import partial
from itertools import zip_longest
from torchdata.datapipes.iter import (
    IterDataPipe,
    FileLister,
    Filter,
    FileOpener,
    LineReader,
    UnBatcher,
    BucketBatcher,
)


def save_dp(dp, save_path):
    print('THIS FUNCTION IS NOT WORKING YET')
    with open(save_path, 'wb') as handle:
        pickle.dump(dp, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_dp(load_path):
    print('THIS FUNCTION IS NOT WORKING YET')
    with open(load_path, 'rb') as file:
        saved_dp = pickle.load(file)
    return saved_dp


class JsonReader(IterDataPipe):
    """ Combined pipeline to list, select, open, read and parse a json file
        This pipeline should generate lists of words to send to the encoder
    """
    def __init__(self, data_dir, split):
        dp = FileLister(data_dir)
        dp = Filter(dp, partial(self.filter_fn, split=split))
        dp = FileOpener(dp)
        dp = LineReader(dp)
        self.dp = dp
    
    def __iter__(self):
        for _, stream in self.dp:
            yield self.parse_fn(json.loads(stream))
    
    @staticmethod
    def filter_fn(filename, split):
        """ Return whether a string filename contains the given split """
        return split in filename

    def parse_fn(self, sample):
        """ This function defines how the data is parsed from the json file
            You should write this function, which depends on your data.
            The output should be a list of token words (or other)
        """
        # return sample
        return self.parse_chemical_reaction(sample)
    
    @staticmethod
    def parse_chemical_reaction(sample):
        """ Parse a dict of src (reactant(s) and product) and tgt (reagent(s))
            strings into a list of tokens representing the smiles reaction
        """
        product = sample['src'].split('.')[-1]
        reactants = sample['src'].replace(product, '')[:-1]
        reagents = sample['tgt']
        reaction = ' > '.join([reactants, reagents, product])
        return reaction.replace('  ', ' ').split(' ')        


class Encoder(IterDataPipe):
    """ Encode lists of words to lists of tokens or ngrams with a tokenizer """
    def __init__(self, dp, tokenizer):
        self.dp = dp
        self.tokenizer = tokenizer
        
    def __iter__(self):
        """ Source sample format: list of strings
            Output sample format (yielded by __iter__):
            - For encoding == 'word': list of token ids
            - For encoding == 'subword': list of lists of token ids, where each
                first token is the word itself, but inside angular brackets
        """
        for sample in self.dp:
            if isinstance(sample, dict):
                yield {k: sample[k] if 'label' in k  # avoid tokenizing labels
                       else self.encode_fn(sample[k]) for k in sample.keys()}
            else:
                yield self.encode_fn(sample)
    
    def encode_fn(self, sample):
        """ Tokenize a list of words using the tokenizer """
        assert isinstance(sample, list), 'Bad input type %s' % type(sample)
        return [self.tokenizer.encode(word) for word in sample]
        

class DynamicBatcher(IterDataPipe):
    """ Combine bucket-batching (batching by groups of sequences with similar
        length) with token-batching (batching based on number of input tokens).
        
        - BucketBatcher creates batches taken from buckets of sequences sorted
            by length, while keeping some randomness using many buckets
        - UnBatcher undo the batching but keeps the order of the sentences
        - DynamicBatcher implements dynamic batching by setting the batch size
            of each batch so that n_tokens per batch < max_tokens
        - DynamicBatcher also trims sequences longer than max_seq_len
        
    """
    def __init__(self, dp, max_tokens, max_len, drop_last=True):
        # Pipeline parameters
        self.max_tokens = max_tokens
        self.max_len = max_len
        self.drop_last = drop_last
        bucket_params = {'batch_size': 1024,
                         'batch_num': 128,
                         'bucket_num': 8,
                         'use_in_batch_shuffle': False,
                         'sort_key': self.sort_fn}

        # Pre-processing (samples ordered by length before dynamic batching)
        dp = BucketBatcher(dp, **bucket_params)
        dp = UnBatcher(dp)
        self.dp = dp
        
    def __iter__(self):
        batch = []
        sample_len_max = 0
        n_tokens_in_padded_batch = 0
        for sample in tqdm(self.dp,
                           desc='Pre-computing batch sizes for this epoch',
                           leave=False):
            
            # Trim the sample in case it is longer than self.max_len
            if isinstance(sample, dict):
                sample = {k: self.trim_fn(v) for k, v in sample.items()}
            else:
                sample = self.trim_fn(sample)
            
            # Compute number of tokens in the batch if this sample were added
            sample_len = self.len_fn(sample, method='sum')
            sample_len_max = max(sample_len_max, sample_len)
            n_tokens_in_padded_batch = sample_len_max * len(batch)
                       
            # Yield the batch if this number is over the max number of tokens
            if n_tokens_in_padded_batch > self.max_tokens:
                yield batch
                batch = []
                sample_len_max = sample_len
                n_tokens_in_padded_batch = sample_len
                
            # Add the sample to the batch (add to empty list if just yielded)
            batch.append(sample)
        
        # Yield the last batch (or not)
        if not self.drop_last:
            yield batch
    
    def trim_fn(self, list_or_int):
        """ Make sure a sequence does not go over the max number of tokens """
        if isinstance(list_or_int, list):
            return list_or_int[:self.max_len]
        else:
            return list_or_int  # sometimes, part of a sample is a label
            
    def sort_fn(self, bucket):
        """ Sort samples using a length metric given by len_fn """
        return sorted(bucket, key=self.len_fn)  # takes default args of len_fn
    
    def len_fn(self, sample, method='first'):
        """ Compute sample length given a method to compute it. The method
            'first' uses only the first dict value to compute length.
        """
        length = 0
        if isinstance(sample, dict):
            for key in sample.keys():
                length += self.compute_len(sample[key])
                if method == 'first':
                    break  # note: dicts are insertion-ordered for python 3.7+
        else:
            length += self.compute_len(sample)
        return length
    
    @staticmethod
    def compute_len(sample):
        """ Length metric that depends on input type """
        if isinstance(sample, str):
            return sample.count(' ') + 1  # e.g., 'i am alban'
        elif isinstance(sample, list):
            return len(sample)  # e.g., [2, 23, 203, 3] or ['i', 'am', 'alban']
        elif isinstance(sample, (int, float)):
            return 1  # e.g., 2 (for instance, a label)
        else:
            raise TypeError(f'Bad input type {type(sample)}')


class DictUnzipper(IterDataPipe):
    """ Take a batch of dicts and unzip it to the corresponding dict of batches
        E.g.: [{'src': id_1, 'tgt': id_a}, {'src': id_2, 'tgt': id_b}, ...]
           -> {'src': [id_1, id_2, ...], 'tgt': [id_a, id_b, ...]}
    """
    def __init__(self, dp):
        self.dp = dp
        self.data_keys = None
    
    def __iter__(self):
        for batch in self.dp:
            if self.data_keys == None:
                self.data_keys = batch[0].keys()
            yield {key: [s[key] for s in batch] for key in self.data_keys}


class TorchPadder(IterDataPipe):
    """ Transform a batch of iterables in a tensor of the same dimensions.
        - Input pipe can consist of dicts of batched lists or batched lists.
        - Each batch is either:
            - A list of ints (e.g., list of labels for all batch samples)
            - A list of list of ints (e.g., ngrams or token sequences)
            - A list of lists of lists (e.g., ngram sequences)
        - Within each nested_list, each sequence can have variable lengths, in
            which case each sequence is padded to the length of the longest.
    """
    def __init__(self, dp, tokenizer):
        self.dp = dp
        self.pad_id = tokenizer.encode('[PAD]')
        self.pad_int = self.pad_id
        if isinstance(self.pad_id, list):
            self.pad_int = self.pad_id[0]
        
    def __iter__(self):
        for batch in self.dp:
            if isinstance(batch, dict):
                yield {key: batch[key] if 'label' in key  # labels untouched
                       else torch.tensor(self.pad_fn(batch[key]))
                       for key in batch.keys()}
            else:
                yield torch.tensor(self.pad_fn(batch))

    def pad_fn(self, nested_list):
        """ Insert a nested list in a torch tensor with the correct dimensions.
            - The nesting of the list can go up to len(list_dims) = 3.
            - The first dimension is always the batch dimension.
            - Tensor dimensions are set to the longest sequence in each dim.
        """
        # Case of unique values (e.g., labels, cooc values, unique tokens)
        if isinstance(nested_list[0], (int, float)):
            return nested_list
        
        # Case of token sequences or ngrams (batch = list of lists)
        elif isinstance(nested_list[0], list):
            if isinstance(nested_list[0][0], int):
                return self.pad_to_longest(nested_list, self.pad_int)
                
            # Case of ngram sequences (batch = list of lists of lists)
            # Note 1: for this case, self.pad_id is of the kind "[0]" (not "0")
            # Note 2: this could definitely be done in a more robust way!
            elif isinstance(nested_list[0][0], list):
                max_ngram = max(max(len(ss) for ss in s) for s in nested_list)
                first_pad = self.pad_id * (max_ngram - len(nested_list[0][0]))
                nested_list[0][0] += first_pad
                padded_list = [self.pad_to_longest(s, self.pad_id[0])
                               for s in nested_list]
                second_pad = (self.pad_id * max_ngram)
                return self.pad_to_longest(padded_list, second_pad)
                
    @staticmethod
    def pad_to_longest(nested_list, pad_elem):
        """ Pad sublists to the length of the longest. The padding element can
            be an int (e.g., 0) or a list of ints (e.g., [0, 0, ..., 0])
        """
        return list(zip(*zip_longest(*nested_list, fillvalue=pad_elem)))
    