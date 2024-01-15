import os
import hashlib
import numpy as np
from nltk.util import ngrams as ngram_base_fn


class Tokenizer():
    def __init__(self,
                 data_dir: str,
                 special_tokens: dict,
                 min_freq: int=5):
        """ Initialize word-level tokenizer.
            
        Parameters
        ----------
            special_tokens (list of str)
                tokens used in the model for unknown words, padding, masking,
                starting/closing sentences, etc.
            min_freq (int)
                minimum frequency at which a word should occur in the corpus to have
                its own token_id (else: token_id of '[UNK]')
                
        """
        self.data_dir = data_dir
        self.encoder = dict(special_tokens)
        self.special_tokens = dict(special_tokens)
        self.min_freq = min_freq
        self.unique_id = self.create_unique_id()
        self.path = os.path.join(self.data_dir, 'tokenizer', self.unique_id)
        
    def create_unique_id(self):
        unique_str = str(vars(self))
        return hashlib.sha256(unique_str.encode()).hexdigest()

    def fit(self, words: list):
        
        # Compute, filter and sort vocabulary by term frequency        
        word_vocab, word_counts = np.unique(words, return_counts=True)
        if self.min_freq > 0:  # remove rare words
            word_vocab = word_vocab[word_counts >= self.min_freq]
            word_counts = word_counts[word_counts >= self.min_freq]
        inds = word_counts.argsort()[::-1]
        word_vocab = word_vocab[inds]
        
        # Generate word level encoder
        self.encoder.update({
            i: idx + len(self.special_tokens) for idx, i in enumerate(word_vocab)
        })
        
        # Store word count for every word (useful for skipgram dataset)
        self.word_counts = {
            self.encoder[word]: count  # note that word_counts was not sorted yet
            for word, count in zip(word_vocab, sorted(word_counts)[::-1])
        }
        self.word_counts.update({k: 1 for k in self.special_tokens.values()})
        
        # Build decoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        # Store and print tokenizer vocabulary information
        self.vocab_sizes = {'total': len(self.encoder),
                            'special': len(self.special_tokens),
                            'word': len(word_vocab)}
        
    def encode(self, word: str):
        try:
            return self.encoder[word]
        except:
            return self.encoder['[UNK]']
    
    def decode(self, token_id: int):
        return self.decoder[token_id]
    
    def get_vocab(self):
        return self.encoder.keys()


class SubWordTokenizer():
    def __init__(self,
                 data_dir: str,
                 special_tokens: dict,
                 ngram_min_len: int,
                 ngram_max_len: int,
                 ngram_mode: str,
                 ngram_base_prefixes: list,
                 ngram_base_suffixes: list,
                 min_freq: int=5,
                 brackets: list=['<', '>'],
                 *args, **kwargs):
        """ Initialize Subword-level tokenizer.

        Parameters
        ----------
            ngram_min_len (int)
                minimum length of ngrams in tokenized subwords
            ngram_max_len (int)
                maximum length of ngrams in tokenized subwords
            special_tokens (list of str)
                tokens used in the model for unknown words, padding, masking,
                starting/closing sentences, etc.
            min_freq (int)
                minimum frequency at which a word should occur in the corpus to have
                its own token_id (else: token_id of '[UNK]')
            brackets (list of 2 str)
                special characters used to differentiate similar words and subwords
                (e.g., word '<her>' vs subword 'her' subword word <where>)
            mode (str)
                how ngrams are computed ('subword' for classic ngrams, 'icd' for
                ngrams suited to icd codes (forward-only ngrams))
        
        """
        self.data_dir = data_dir
        self.encoder = dict(special_tokens)  # will be updated in fit
        self.special_tokens = dict(special_tokens)  # will stay the same
        assert ngram_min_len >= 0 and ngram_max_len >= ngram_min_len
        self.ngram_len = list(range(ngram_min_len, ngram_max_len + 1))
        self.min_freq = min_freq
        self.brackets = brackets
        self.forbidden_ngrams = brackets + list(special_tokens.keys())
        self.ngram_mode = ngram_mode
        self.ngram_base_prefixes = ngram_base_prefixes
        self.ngram_base_suffixes = ngram_base_suffixes
        self.unique_id = self.create_unique_id()
        self.ngram_fn = self._select_ngram_fn(ngram_mode)
        self.path = os.path.join(self.data_dir, 'tokenizer', self.unique_id)
        
    def create_unique_id(self):
        unique_str = str(vars(self))
        return hashlib.sha256(unique_str.encode()).hexdigest()
        
    def _select_ngram_fn(self, ngram_mode):
        if ngram_mode == 'subword':
            return self._generate_multi_ngrams
        elif ngram_mode == 'icd':
            return self._generate_icd_ngrams
        else:
            raise ValueError('Unknown ngramization mode %s' % self.ngram_mode)

    @staticmethod
    def _flatten(t):
        return [item for sublist in t for item in sublist]

    @staticmethod
    def _generate_ngrams(word, n, forbidden_ngram):
        return [ngram for ngram in [''.join(i) for i in ngram_base_fn(word, n)]
                if (ngram not in forbidden_ngram)]  # and (word[1:-1] not in ngram)]
    
    def _initialize_ngrams(self, word):
        base_ngrams = []
        for base_voc in [self.ngram_base_prefixes, self.ngram_base_suffixes]:
            for base_token in base_voc:
                if base_token in word:
                    word = word.replace(base_token, '')
                    base_ngrams.append(base_token)
                    break
        return word, base_ngrams
    
    def _generate_icd_ngrams(self, word):
        # TODO: do this only for appropriate codes (DIA, PRO, MED?, LAB?)
        word, ngrams = self._initialize_ngrams(word)
        ngrams.extend([word[:i] for i in range(1, len(word))])
        return ngrams

    def _generate_multi_ngrams(self, word):
        word, ngrams = self._initialize_ngrams(word)
        ngrams.extend(self._flatten([self._generate_ngrams(
            word, i, self.forbidden_ngrams) for i in self.ngram_len]))
        return ngrams
    
    def _add_brackets(self, word):
        return self.brackets[0] + word + self.brackets[1]
    
    def _compute_and_sort_vocabulary(self, words_or_ngrams):
        vocab, counts = np.unique(words_or_ngrams, return_counts=True)
        if self.min_freq > 0:  # remove rare words
            vocab = vocab[counts >= self.min_freq]
            counts = counts[counts >= self.min_freq]
        inds = counts.argsort()[::-1]  # most occurent first
        return vocab[inds], counts[inds]
        
    def fit(self, words):
        print('Training tokenizer')
        assert isinstance(words, list)
        
        # Compute and sort word and ngram vocabularies
        word_vocab, word_counts = self._compute_and_sort_vocabulary(words)
        if self.ngram_mode != 'icd':  # differentiate whole words from ngrams
            word_vocab = [self._add_brackets(w) for w in word_vocab]
        ngrams = self._flatten([self.ngram_fn(word) for word in word_vocab])
        ngram_vocab, _ = self._compute_and_sort_vocabulary(ngrams)
        
        # Populate word level encoder (if not in char mode)
        len_so_far = len(self.encoder)
        self.encoder.update({
            word: idx + len_so_far for idx, word in enumerate(word_vocab)
        })

        # Store word count for every word (if not in char mode)
        self.word_counts = {
            self.encoder[word]: count  # note that word_counts was not sorted yet
            for word, count in zip(word_vocab, sorted(word_counts)[::-1])
        }
        self.word_counts.update({k: 1 for k in self.special_tokens.values()})

        # Update encoder with ngram level vocabulary
        len_so_far = len(self.encoder)
        self.encoder.update({i: (idx + len_so_far)
                             for idx, i in enumerate(ngram_vocab)})

        # Decoder
        self.decoder = {v: k if k not in word_vocab else k[1:-1]
                        for k, v in self.encoder.items()}

        # Store and print tokenizer vocabulary information
        self.vocab_sizes = {'total': len(self.encoder),
                            'special': len(self.special_tokens),
                            'word': len(word_vocab),
                            'ngram': len(ngram_vocab)}
        print(' - Vocabulary sizes: %s' % self.vocab_sizes)
        
    def encode(self, word):
        # Generate n-grams and add them to the word (if not in char mode)
        seq = []
        if word in self.special_tokens:
            seq = [word]
        elif self.ngram_mode == 'subword':
            seq = [self._add_brackets(word)] + self.ngram_fn(word)
        elif self.ngram_mode == 'icd':
            seq = [word] + self.ngram_fn(word)
            
        # Encode the word and the ngram
        indices = []
        for i, word_or_ngram in enumerate(seq):  # first in list is word itself
            try:
                indices.append(self.encoder[word_or_ngram])
            except:  # add unkown token only for word, not for ngram
                if i == 0: indices.append(self.encoder['[UNK]'])
        return indices
    
    def decode(self, token_id_or_ids):
        """ Decode a token or a list of tokens.
            - If the input is a list of tokens, this means that the first token
                correspond to the word and the subsequent ones correspond to
                the subword.
            - In this case, the tokenizer only returns the decoded word.    
        """
        if type(token_id_or_ids) == list:
            return self.decoder[token_id_or_ids[0]]
        elif type(token_id_or_ids) == int:
            return self.decoder[token_id_or_ids]
        else:
            raise TypeError('Invalid token format: %s' % type(token_id_or_ids))

    def get_vocab(self):
        voc = self.encoder.keys()  # still has brackets around each word
        return [w.strip(self.brackets[0]).strip(self.brackets[1]) for w in voc]
        