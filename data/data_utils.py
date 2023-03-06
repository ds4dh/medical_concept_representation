import json
import torch
import random
from functools import partial
from itertools import zip_longest
from torchdata.datapipes.iter import (
    IterDataPipe,
    FileLister,
    Filter,
    FileOpener,
    LineReader,
    Shuffler,
    MaxTokenBucketizer,
)


class JsonReader(IterDataPipe):
    """ Combined pipeline to list, select, open, read and parse a json file
    """
    def __init__(self, data_dir, split, parse_id=None):
        super().__init__()
        self.data_dir = data_dir
        self.parse_id = parse_id
        dp = FileLister(data_dir)
        dp = Filter(dp, partial(self.filter_fn, split=split))
        dp = FileOpener(dp)
        dp = LineReader(dp)
        self.dp = dp
    
    def __iter__(self):
        for _, stream in self.dp:
            yield json.loads(stream)
    
    @staticmethod
    def filter_fn(filename, split):
        """ Return whether a string filename contains the given split
        """
        return split in filename


class Encoder(IterDataPipe):
    """ Encode lists of words to lists of tokens or ngrams with a tokenizer 
    """
    def __init__(self, dp, tokenizer):
        super().__init__()
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
        """ Tokenize a list of words using the tokenizer
        """
        assert isinstance(sample, list), 'Bad input type %s' % type(sample)
        return [self.tokenizer.encode(word) for word in sample]
    

class TokenFilter(IterDataPipe):
    def __init__(self, dp, to_remove=[], to_split=[]):
        """ Clean samples with token filters and/or split by matching tokens
        """
        super().__init__()
        self.dp = dp
        self.to_remove = to_remove
        self.to_split = to_split
        
    def __iter__(self):
        for sample in self.dp:
            if isinstance(sample, dict):
                yield {k: sample[k] if 'label' in k  # avoid acting on labels
                       else self.filter_fn(sample[k]) for k in sample.keys()}
            else:
                yield self.filter_fn(sample)
                
    def filter_fn(self, sample):
        """ Filter out tokens that contains tokens to remove and, if required,
            split the sample by matching split tokens
        """
        sample = [w for w in sample if not any(s in w for s in self.to_remove)]
        if len(self.to_split) == 0:
            return sample
        else:
            return (
                [w for w in sample if not any(s in w for s in self.to_split)],
                [w for w in sample if any(s in w for s in self.to_split)],
            )
            

class TokenShuffler(IterDataPipe):
    """ Shuffle tokens inside sample sequences with some probability
    """
    def __init__(self, dp, shuffle_prob=0.0, shuffle_mode='partial'):
        super().__init__()
        self.dp = dp
        self.shuffle_prob = shuffle_prob
        assert shuffle_mode in ['whole', 'partial'],\
            'Invalid shuffle mode [whole, partial]'
        if shuffle_mode == 'whole':
            self.shuffle_fn = self.full_shuffle_fn
        elif shuffle_mode == 'partial':
            self.shuffle_fn = self.partial_shuffle_fn
    
    def __iter__(self):
        for sample in self.dp:
            if isinstance(sample, dict):
                yield {k: sample[k] if 'label' in k  # avoid shuffling labels
                       else self.shuffle_fn(sample[k]) for k in sample.keys()}
            else:
                yield self.shuffle_fn(sample)
    
    def full_shuffle_fn(self, sample):
        """ Shuffle a list of tokens with some probability
        """
        if random.random() < self.shuffle_prob: random.shuffle(sample)
        return sample
    
    def partial_shuffle_fn(self, sample):
        """ Insert a fraction of the tokens of a list at random positions
        """
        if self.shuffle_prob == 0.0: return sample
        n_elem_shuffled = int(len(sample) * self.shuffle_prob)
        to_shuffle = random.sample(sample, n_elem_shuffled)
        sample = [x for x in sample if x not in to_shuffle]
        random.shuffle(to_shuffle)
        for elem in to_shuffle:
            idx = random.randint(0, len(sample))
            sample.insert(idx, elem)
        return sample


class Trimer(IterDataPipe):
    def __init__(self, dp, max_len):
        self.dp = dp
        self.max_len = max_len
    
    def __iter__(self):
        for sample in self.dp:
            if isinstance(sample, dict):
                yield {k: self.trim_fn(v) for k, v in sample.items()}
            else:
                yield self.trim_fn(sample)
    
    def trim_fn(self, list_or_int):
        """ Make sure a sequence does not go over the max number of tokens
        """
        if isinstance(list_or_int, list):
            return list_or_int[:self.max_len]  # or random ordered selection?
        else:
            return list_or_int  # sometimes, part of a sample is a label 


class DictCustomizer(IterDataPipe):
    """ Transorm a dictionary samples into custom dictionary objects
    """
    def __init__(self, dp):
        self.dp = dp
    
    def __iter__(self):
        for sample in self.dp:
            if isinstance(sample, dict):
                yield self.CustomDict(sample)
            else:
                yield sample
    
    class CustomDict(dict):
        """ Subclass of dict that implements dummy comparison operators
            This avoids a bug in the MaxTokenBucketizer bucketizer.
        """
        def __lt__(self, _):
            return False
        
        def __gt__(self, _):
            return True


class CustomBatcher(IterDataPipe):
    """ Batche samples dynamically, so that each batch size < max_tokens
        Also trim sequences > max_len and provide variability by shuffling
        The unique logic is there to prevent a bug (?) in MaxTokenBucketizer
    """
    def __init__(self, dp, max_tokens, max_len, shuffle=True):
        dp = Trimer(dp, max_len)
        if shuffle: dp = Shuffler(dp)
        dp = DictCustomizer(dp)  # useful to fix a bug in MaxTokenBucketizer
        dp = MaxTokenBucketizer(datapipe=dp,
                                max_token_count=max_tokens,
                                len_fn=self.len_fn,
                                buffer_size=96,  # not too large for variability
                                include_padding=True)
        if shuffle: dp = Shuffler(dp)
        dp = DictUnzipper(dp)
        self.dp = dp
    
    def __iter__(self):
        for sample in self.dp:
            yield sample
            
    def len_fn(self, sample):
        """ Compute sample length (given data type, apply different method)
            Note that the length is made unique to avoid an issue in torchdata
        """
        if isinstance(sample, dict):
            return sum([self.compute_len(sample[k]) for k in sample.keys()])
        else:
            return self.compute_len(sample)
    
    @staticmethod
    def compute_len(sample):
        """ Length metric that depends on input type
        """
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
        super().__init__()
        self.dp = dp
        self.data_keys = None
    
    def __iter__(self):
        for batch in self.dp:
            if self.data_keys == None:
                self.data_keys = list(batch[0].keys())
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
        super().__init__()
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
