import json
import random
import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from functools import partial
from itertools import zip_longest
from torchdata.datapipes.iter import (
    IterDataPipe,
    FileLister,
    FileOpener,
    Filter,
    LineReader,
    UnBatcher,
    BucketBatcher,
    MaxTokenBucketizer
)


def filter_fn(filename, split):
    """ Filter function that selects the correct file given a split
    
    Args:
        filename (str): name of the candidate file
        split (str): split ('train', 'val', or 'test')

    Returns:
        (bool): wether the given file corresponds to the given split.

    """
    return split in filename


def sort_fn(bucket):
    """ Sort samples in a bucket using a length metric given by sort_len_fn
    
    Args:
        bucket (iterable of dicts): set of samples to sort
    
    Returns:
        iterable: set of sorted samples
    
    """
    return sorted(bucket, key=len_fn)  # takes default argument of len_fn


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


def compute_len(sample):
    """ Length metric depending on input type
    
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
    print('Creating co-occurrence matrix')
    names = np.unique(flatten(document)).tolist()
    occurrences = OrderedDict((name, OrderedDict((name, 0) for name in names))
                              for name in names)
    # occurrences = {name: {name: 0 for name in names} for name in names}

    # Find the co-occurrences:
    for sample in tqdm(document, desc='Filling co-occurrence matrix'):
        for i, center in enumerate(sample):
            for context in sample[:i] + sample[i + 1:]:
                occurrences[center][context] += 1
    return occurrences


def filter_cooc(occurrences, min_cooc):
    assert min_cooc > 0, 'Tokens must co-occur at least once \
                          because GloVe is a log-regression model.'
    # Filter based on min_cooc:
    occ = occurrences.copy()
    for k, v in tqdm(occ.items(), desc='Filtering co-occurence matrix'):
        occ[k] = {x: y for x, y in v.items() if y >= min_cooc}
    return occ


def format_cooc(cooc):
    samples = list()
    for l_dict_str, r_dict in cooc.items():
        for r_dict_str, cooc_i in r_dict.items():
            samples.append({'left': l_dict_str,
                            'right': r_dict_str,
                            'cooc': float(cooc_i)})
    return samples


def create_skipgram(document, context_size):
    ''' Generate a set of (context, target) pairs and build the batch
        as a batched tensor for the continuous bag of words task.
    '''
    sample_pairs = list()
    for sentence in tqdm(document, desc='Creating skipgram sample pairs'):
        for center_pos, center_word in enumerate(sentence):
            for i in range(-context_size, context_size + 1):
                context_pos = center_pos + i
                if context_pos < 0 \
                or context_pos >= len(sentence) \
                or i == 0:  # same word
                    continue
                context_word = sentence[context_pos]
                sample_pairs.append({'center': center_word,
                                     'context': context_word})
    return sample_pairs
                
                
class JsonReader(IterDataPipe):
    """ Combined pipeline to list, select, open, read and parse a json file """
    def __init__(self, data_dir, split):
        dp = FileLister(data_dir)
        dp = Filter(dp, partial(filter_fn, split=split))
        dp = FileOpener(dp)
        dp = LineReader(dp)
        self.dp = dp
    
    def __iter__(self):
        for _, stream in self.dp:
           yield json.loads(stream)


class Encoder(IterDataPipe):
    def __init__(self, dp, tokenizer, encoding='word'):
        self.dp = dp
        self.tokenizer = tokenizer
        assert encoding in ('word', 'subword')
        self.use_ngrams = (encoding == 'subword')
        
    def __iter__(self):
        for sample in self.dp:
            if self.use_ngrams:
                yield [[self.tokenizer.encode(subword) \
                        for subword in subwords] for subwords in sample]
            else:
                yield [self.tokenizer.encode(word) for word in sample]
        

class GloveMaker(IterDataPipe):
    """ Compute co-occurence matrix from the source pipeline, and load it to
        memory as a flat array, then get ready to yield the computed samples
    """
    def __init__(self, dp):
        # Compute the co-occurence matrix
        raw_cooc = create_cooc(list(dp))
        filtered_cooc = filter_cooc(raw_cooc, min_cooc=10)  # good value for min_cooc?
        self.dp = format_cooc(filtered_cooc)
        
    def __iter__(self):
        # Sample format: {'left': token_id, 'right': token_id, 'cooc': float}
        for sample in self.dp:
            yield sample


class SkipGramMaker(IterDataPipe):
    """ Compute all possible skipgram pairs from the source pipeline and load
        them to memory, then get ready to yield the computed sample pairs
    """
    def __init__(self, dp):
        self.dp = create_skipgram(dp, context_size=2)
    
    def __iter__(self):
        # Sample format: {'center': token_id, 'context': token_id}
        for sample in self.dp:
            yield sample


class DynamicMasker(IterDataPipe):
    def __init__(self, dp, special_ids):
        self.dp = dp
        self.mask_id = special_ids['[MASK]']
        # TODO: implement dynamic masking
    
    def __iter__(self):
        # Sample format: {'masked': list of token_ids, 'target': same}
        for sample in self.dp:
            yield sample

            
class DictUnzipper(IterDataPipe):
    """ Take a batch of dicts and unzip it to the corresponding dict of batches
        Example: [{'src': ..., 'tgt': ...}, ...] -> {'src': [...], 'tgt': [...]}
    """
    def __init__(self, dp):
        self.dp = dp
        self.data_keys = None
    
    def __iter__(self):
        for batch in self.dp:
            if self.data_keys == None:
                self.data_keys = batch[0].keys()
            yield {key: [s[key] for s in batch] for key in self.data_keys}


class DynamicBucketBatcher(IterDataPipe):
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


class Padder(IterDataPipe):
    """ Pad each element of a batch, so that it can be put in a tensor.
        Input pipe can consist of a dict of batched lists or a batched list.
    """
    def __init__(self, dp, special_ids, max_len):
        self.dp = dp
        self.pad_fn = partial(pad_fn, **special_ids, max_len=max_len)
        self.data_keys = None
    
    def __iter__(self):
        for batch in self.dp:
            if type(batch) is dict:
                if self.data_keys is None:
                    self.data_keys = batch[0].keys()  # not sure if working
                yield {key: self.pad_fn(batch[key]) for key in self.data_keys}
            else:
                yield self.pad_fn(batch)
                

class Torcher(IterDataPipe):
    """ Transform a batch of iterables in a tensor of the same dimensions.
        Input pipe can consist of a dict of batched lists or a batched list.
    """
    def __init__(self, dp):
        self.dp = dp

    def __iter__(self):
        for batch in self.dp:
            if type(batch) is dict:
                yield {key: torch.tensor(batch[key]) for key in batch.keys()}
            else:
                yield torch.tensor(batch)
