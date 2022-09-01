import os
import tempfile
import json
import random
import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from functools import partial
from itertools import zip_longest
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.iter import (
    FileLister,
    FileOpener,
    Filter,
    LineReader,
    UnBatcher,
    BucketBatcher,
    MaxTokenBucketizer
)

# def key_value_fn(batch):
#     return (batch[key] for key in batch.keys())

def sort_fn(bucket):
    """ Sort samples in a bucket using a length metric given by sort_len_fn.
    
    Args:
        bucket (iterable of dicts): set of samples to sort
    
    Returns:
        iterable: set of sorted samples
    
    """
    return sorted(bucket, key=len_fn)  # takes default argument of len_fn


def compute_len(sample):
    """ Length metric depending on input type.
    
    Args:
        sample (str or list): element whose length is computed
    
    Returns:
        int: length of the sample
    
    """
    if type(sample) is str:
        return sample.count(' ') + 1
    if type(sample) is list:
        return len(sample)
    else:
        raise TypeError(f'Bad input type {type(sample)}')


def len_fn(sample, unique=False, method='first'):
    """ Compute length of a sample following a method to compute length
    
    Args:
        sample (dict of sequences or sequence): element whose length to compute
        unique (bool): tag any length value by adding a small random number
        method (str): which method is used to compute the length of the sample
            - 'first': {'src': [...], 'tgt': [...]}, -> len(src)
            - 'sum': {'src': [...], 'tgt': [...]}, -> len(src) + len(tgt)
            Note: if the sample is just a sequence, len(sequence) is computed
    
    Returns:
        int: length of the sample - note: sample format is tokens separated by
            spaces, hence sample length = number of white spaces + 1
    
    """
    length = 0
    if type(sample) is dict:
        for key in sample.keys():
            length += compute_len(sample[key])
            if method == 'first':
                break  # note: dict keys are insertion-ordered for python 3.7+
    else:
        length += compute_len(sample)
    if unique:
        length += (0.01 * random.random()) - 0.005
    return length


def path_fn(filename):
    """ Generate a path name for a cache to be stored in the tmp folder
    
    Args:
        filename (str): name of the file to cache
    
    Returns:
        str: path to the cached file
        
    """
    temp_dir = tempfile.gettempdir()
    return os.path.join(temp_dir, os.path.basename(filename))


def encode_fn(batch, tokenizer):
    """ Encode input tokens with a tokenizer
    
    Args:
        batch_or_sample (set of sequences or sequence): sequence(s) to encode
        tokenizer (tokenizers.Tokenizer): tokenizer used to encode tokens
    
    Returns:
        iterable of lists: tokenized batch
    
    """
    # CHECK HERE BECAUSE ITS LIST BUT NOT BATCH!!
    if type(batch) is str:
        return tokenizer.encode(batch).ids
    else:
        return [e.ids for e in tokenizer.encode_batch(batch)]


def encapsulate_fn(sequence, bos_id, eos_id):
    """ Add start and end tokens to one sequence of tokens
    
    Args:
        bos_id (int): id of the "beginning of sentence" token
        eos_id (int): id of the "end of sentence" token
        sequence (list of ints): sequence of token ids
    
    """
    sequence.append(bos_id); sequence.insert(0, eos_id)


def pad_fn(batch, bos_id, eos_id, pad_id, max_len):
    """ Pad each sequence of a batch to the length of the longest
    
    Args:
        bos_id (int): id of the "beginning of sentence" token
        eos_id (int): id of the "end of sentence" token
        pad_id (int): id of the padding token
        batch (iterable of lists of ints): already tokenized batch
    
    Returns:
        iterable of lists of ints: same batch but with added padding
        
    """
    [encapsulate_fn(s, bos_id, eos_id) for s in batch]
    batch = list(zip(*zip_longest(*batch, fillvalue=pad_id)))
    if len(batch[0]) > max_len:
        batch = [s[:max_len] for s in batch]  # trim to max_len if too long
    return batch


def flatten(t):
    return [item for sublist in t for item in sublist]


def create_cooc(document):
    """ Create co-occurences matrix

    Args:
        document (list of lists of strs): training dataset samples

    Returns:
        list of lits: co-occurences matrix
        
    """
    names = np.unique(flatten(document)).tolist()
    occurrences = OrderedDict((name, OrderedDict((name, 0)
                              for name in names)) for name in names)

    # Find the co-occurrences:
    for l in tqdm(document, leave=False):
        for i in range(len(l)):
            for item in l[:i] + l[i + 1:]:
                occurrences[l[i]][item] += 1

    return occurrences


def filter_cooc(occurrences, min_cooc):
    assert min_cooc > 0, 'Tokens must co-occur at least once \
                          because GloVe is a log-regression model.'
    # Filter based on min_cooc:
    occ = occurrences.copy()
    for k, v in tqdm(occ.items(), leave=False):
        occ[k] = {x: y for x, y in v.items() if y >= min_cooc}
    return occ


def format_cooc(cooc):
    left, right, cooc_int = list(), list(), list()
    for l_dict_str, r_dict in cooc.items():
        for r_dict_str, cooc_i in r_dict.items():
            left.append(str(l_dict_str))
            right.append(str(r_dict_str))
            cooc_int.append(int(cooc_i))
    return left, right, cooc_int


class JsonReader(IterDataPipe):
    """ Combined pipeline to list, select, open, read and parse a json file """
    def __init__(self, data_dir, split):
        dp = FileLister(data_dir)
        dp = Filter(dp, lambda t: split in t)
        dp = FileOpener(dp)
        dp = LineReader(dp)
        self.dp = dp
    
    def __iter__(self):
        for _, stream in self.dp:
            yield json.loads(stream)
            
            
class DictUnzipper(IterDataPipe):
    """ Take an iterable of dicts and unzip it to a dict of iterables """
    def __init__(self, dp, data_keys):
        self.dp = dp
        self.data_keys = data_keys
    
    def __iter__(self):
        for batch in self.dp:
            yield {key: [s[key] for s in batch] for key in self.data_keys}


class DynamicBucketBatcher():
    """ Combine bucket-batching (batching by groups of sequences with similar
        length) with token-batching (batching based on number of input tokens).
    """
    def __init__(self, dp, max_tokens):
        dp = BucketBatcher(dp, batch_size=1024, bucket_num=8, sort_key=sort_fn)
        dp = UnBatcher(dp)
        length_fn = partial(len_fn, unique=True, method='sum')
        dp = MaxTokenBucketizer(dp, max_tokens, len_fn=length_fn)
        self.dp = dp
                
    def __iter__(self):
        for batch in self.dp:
            yield batch


class _Encoder(IterDataPipe):
    """ Encode tokens to token ids using the given encoder function.
        Input pipe can consist of a dict of batched lists or a batched list.
    """
    def __init__(self, dp, tokenizer, data_keys=[]):
        self.dp = dp
        self.encode_fn = partial(encode_fn, tokenizer=tokenizer)
        self.data_keys = data_keys
    
    def __iter__(self):
        for batch in tqdm(self.dp, desc='Tokenizing dataset'):
            if len(self.data_keys) > 0:
                ids = {key: self.encode_fn([s[key] for s in batch])
                            for key in self.data_keys}
                yield [dict(zip(ids, t)) for t in zip(*ids.values())]
            else:
                yield self.encode_fn(batch)


class Encoder():
    """ Take the _Encoder output and unbatch it """
    def __init__(self, dp, tokenizer, data_keys=[]):
        dp = _Encoder(dp, tokenizer, data_keys=[])
        dp = UnBatcher(dp)
        self.dp = dp
                
    def __iter__(self):
        for batch in self.dp:
            yield batch


class Padder(IterDataPipe):
    """ Pad each element of a batch, so that it can be put in a tensor.
        Input pipe can consist of a dict of batched lists or a batched list.
    """
    def __init__(self, dp, special_ids, max_len, data_keys=[]):
        self.dp = dp
        self.pad_fn = partial(pad_fn, **special_ids, max_len=max_len)
        self.data_keys = data_keys
    
    def __iter__(self):
        for batch in self.dp:
            if len(self.data_keys) > 0:
                yield {key: self.pad_fn(batch[key]) for key in self.data_keys}
            else:
                yield self.pad_fn(batch)


class DynamicMasker(IterDataPipe):
    def __init__(self, dp, special_ids):
        self.dp = dp
        # TODO: implement this (dynamic masking)
        

class Torcher(IterDataPipe):
    """ Transform a batch of iterables in a tensor of the same dimensions.
        Input pipe can consist of a dict of batched lists or a batched list.
    """
    def __init__(self, dp, data_keys=[]):
        self.dp = dp
        self.data_keys = data_keys
    
    def __iter__(self):
        for batch in self.dp:
            if len(self.data_keys) > 0:
                yield {key: torch.tensor(batch[key]) for key in self.data_keys}
            else:
                yield torch.tensor(batch)


class Glover(IterDataPipe):
    """ Transform an item into a set of left/right context and co-occurence """
    def __init__(self, dp, data_dir, load_cooc=False):
        self.dp = dp
        raw_cooc = create_cooc(JsonReader(data_dir, 'val'))
        self.left, self.right, self.cooc = format_cooc(raw_cooc)
        # Ttokenized beforehand!!! (with data.Encoder)
        # self.tokenizer = tokenizer
        # self.l_e = [self.tokenizer.encode(i) for i in left]
        # self.r_e = [self.tokenizer.encode(i) for i in right]
        # self.c = [float(i) for i in cooc]

    # def __len__(self):
    #     return len(self.cooc)

    def __iter__(self):
        for item in self.dp:
            yield {
                'left': self.l_e[item],
                'right': self.r_e[item],
                'cooc': self.c[item]
            }

