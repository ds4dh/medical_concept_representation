import os
import json
import pickle
import random
import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from functools import partial
from itertools import zip_longest, compress
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
    """ Combined pipeline to list, select, open, read and parse a json file """
    def __init__(self, data_dir, split):
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
        return split in filename


class Encoder(IterDataPipe):
    """ Encode strings or lists of strings to tokens using a tokenizer """
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
            yield [self.tokenizer.encode(word) for word in sample]
            
class CoocMaker(IterDataPipe):
    """ Compute co-occurence matrix from the source pipeline, and load it to
        memory as a flat array, then get ready to yield the computed samples
    """
    def __init__(self, dp, tokenizer, data_dir, split, load_data=False):
        save_or_load_path = os.path.join(data_dir, f'cooc_{split}')
        if load_data:
            self.dp = load_dp(save_or_load_path)
        else:
            raw_cooc = self.create_cooc(list(dp))
            filtered_cooc = self.filter_cooc(raw_cooc)
            self.dp = self.format_cooc(filtered_cooc, tokenizer)
            if 0:
                save_dp(self.dp, save_or_load_path)
        
    def __iter__(self):
        """ Sample format: {'left': token_id (int) or ngram (list of ints),
                            'right': token_id (int) or ngram (list of ints),
                            'cooc': float}
        """
        for sample in self.dp:
            yield sample
    
    @staticmethod
    def create_cooc(document):
        """ Compute co-occurrences between tokens in document sentences """
        # Initialize cooccurrence matrix using document vocabulary
        print(' - Initializing co-occurrence matrix')
        if isinstance(document[0][0], list):  # for ngram encoding
            document = [[ids[0] for ids in sample] for sample in document]
        flattened_document = [id_ for sample in document for id_ in sample]
        token_ids = np.unique(flattened_document).tolist()
        cooc = OrderedDict((id_, OrderedDict((id_, 0) for id_ in token_ids))
                            for id_ in token_ids)
        
        # For each token pair, fill-in how many times they co-occur
        for sample in tqdm(document, desc=' - Filling co-occurrence matrix'):
            for i, center in enumerate(sample):
                for context in sample[:i] + sample[i+1:]:
                    cooc[center][context] += 1
        
        return cooc

    @staticmethod
    def filter_cooc(cooc_, min_cooc=1):
        """ Filters out token co-occurences that are below a threshold """
        assert min_cooc > 0, 'Tokens must co-occur at least once because \
                              GloVe is a log-regression model.'
        
        # Keep only co-occurrences that are above a given threshold
        cooc = cooc_.copy()
        for k, v in tqdm(cooc.items(),
                         desc=' - Filtering co-occurrence array'):
            cooc[k] = {x: y for x, y in v.items() if y >= min_cooc}
        
        return cooc
    
    @staticmethod
    def format_cooc(cooc, tokenizer):
        """ Format co-occurrence dict to left/right inputs and cooc target """
        # Take care of the case where token encoding uses ngrams
        ngram_map = {}
        for token in cooc.keys():
            ngram_map[token] = tokenizer.encode(tokenizer.decode(token))
        
        # Format the co-occurrence matrix as requested by the models
        samples = list()
        for l_token, r_dict in tqdm(cooc.items(),
                                    desc=' - Formatting co-occurrence array'):
            for r_token, cooc_i in r_dict.items():
                samples.append({'left': ngram_map[l_token],
                                'right': ngram_map[r_token],
                                'cooc': float(cooc_i)})
        
        return samples


class SkipGramMaker(IterDataPipe):
    """ Compute all possible skipgram pairs from the source pipeline and load
        them to memory, then get ready to yield the computed sample pairs
    """
    def __init__(self, dp, tokenizer, data_dir, split, load_data=False):
        save_or_load_path = os.path.join(data_dir, f'skipgram_{split}')
        if load_data:
            self.dp = load_dp(save_or_load_path)
        else:
            subsample_probs = self.compute_subsample_probs(tokenizer)
            subsampled_dp = self.subsample_document(dp, subsample_probs)
            self.dp = self.create_skipgram(subsampled_dp)
            if 0:
                save_dp(self.dp, save_or_load_path)
            
    def __iter__(self):
        """ Sample format: {'center': token_id (int) or ngrams (list of ints),
                            'context': token_id (int) or ngrams (list of ints)}
        """
        for sample in self.dp:
            yield sample
    
    @staticmethod
    def compute_subsample_probs(tokenizer, thresh=1e-4):
        """ Compute the subsample probability for each token id """
        word_counts = tokenizer.word_counts
        sum_of_all_word_counts = sum(word_counts.values())
        subsample_probs = {}
        for token_id, word_occurence in word_counts.items():
            word_fraction = word_occurence / sum_of_all_word_counts
            keep_score = (thresh / word_fraction) ** 0.5
            subsample_probs[token_id] = min(keep_score, 1.0)
        
        return subsample_probs
    
    @staticmethod
    def subsample_document(document, subsample_probs):
        """ Subsample tokens in a document (to skip common tokens) """
        subsampled_document = []
        for sentence in tqdm(document, desc=' - Subsampling document'):
            subsampled_sentence = []
            if isinstance(sentence[0], list):
                check_sentence = [token_id[0] for token_id in sentence]
            else:
                check_sentence = sentence
            probs = [subsample_probs[token_id] for token_id in check_sentence]
            selection = [p > random.random() for p in probs]
            subsampled_sentence = list(compress(sentence, selection))
            if subsampled_sentence:
                subsampled_document.append(subsampled_sentence)

        return subsampled_document

    @staticmethod
    def create_skipgram(document, max_context_size=5):
        """ Generate a set of (context, target) pairs from sentence list """
        sample_pairs = list()
        for sentence in tqdm(document, desc=' - Creating skipgram pairs'):
            for center_pos, center_token_id in enumerate(sentence):
                context_size = random.randint(1, max_context_size)
                for i in range(-context_size, context_size + 1):
                    # Find context word position and skip if outside sentence
                    context_pos = center_pos + i
                    if not 0 < context_pos < len(sentence) or i == 0:
                        continue

                    # Retrieve context word
                    context_token_id = sentence[context_pos]
                    if isinstance(center_token_id, list):  # for ngram encoding
                        context_token_id = context_token_id[0]
                    
                    # Update the sample pair list
                    sample_pairs.append({'center': center_token_id,
                                         'context': context_token_id})
                    
        return sample_pairs


class DynamicMasker(IterDataPipe):
    """ Take samples and replace a given proportion of token_ids by mask_id """
    def __init__(self, dp, tokenizer):
        self.dp = dp
        self.mask_id = tokenizer.encode('[MASK]')
        self.bos_id = tokenizer.encode('[CLS]')
        self.eos_id = tokenizer.encode('[END]')
        
    def __iter__(self):
        """ Sample format: {'masked': list of token_ids (ints)
                                           or ngrams (lists of ints),
                            'target': list of token_ids (only whole words)}
        """
        for sample in self.dp:
            yield {k: v for k, v in self.mask_fn(sample).items()}

    def beos_fn(self, sample):
        """ Add start and end tokens to a sequence of tokens """
        sample.insert(0, self.bos_id); sample.append(self.eos_id)
        return sample

    def mask_fn(self, sample):
        """ Implement dynamic masking (i.e., called at yield-time) by replacing
            a percentage of the sample tokens by the id of '[MASK]'.
            - Note: '[BOS]' and '[EOS]' tokens are added to the sequences, but
                are not subject to masking! (added after)
            - Note: for the ngram case, tgt sequences only contain whole words
        """
        assert isinstance(sample, list), 'Bad sample type %s' %  type(sample)
        
        # Mask tokens and add bos and eos ids afterwards, to avoid masking them
        msk = list(map(self.replace_fn, sample))
        msk = self.beos_fn(msk)

        # Add bos and eos ids, then collapse to whole words only for ngram case
        tgt = self.beos_fn(sample)
        tgt = [s[0] for s in sample] if isinstance(sample[0], list) else sample
        
        return {'masked': msk, 'target': tgt}

    def replace_fn(self, token, mask_prob=0.15):
        """ Replace a token by mask_id given a probability to mask """
        if mask_prob > random.random():
            return self.mask_id
        else:
            return token
    
        
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
        """ Compute sample length following a given method to compute it """
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
                yield {key: torch.tensor(self.pad_fn(batch[key]))
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
    