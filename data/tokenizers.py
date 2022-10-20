import numpy as np
from nltk.util import ngrams as ngram_base_fn


class Tokenizer():
    """
    Word-level tokenizer.
    """
    def __init__(self, special_tokens, min_freq=0):
        self.encoder = dict(special_tokens)
        self.special_tokens = dict(special_tokens)
        self.min_freq = min_freq

    def fit(self, words):
        print('Training tokenizer')
        assert isinstance(words, list)
        
        # Compute and sort vocabulary
        word_vocab, word_counts = np.unique(words, return_counts=True)
        if self.min_freq > 0:  # remove rare words
            word_vocab = word_vocab[word_counts > self.min_freq]
            word_counts = word_counts[word_counts > self.min_freq]
        inds = word_counts.argsort()[::-1]
        word_vocab = word_vocab[inds]

        # Generate word level encoder
        self.encoder.update({i: (idx + len(self.special_tokens))
                             for idx, i in enumerate(word_vocab)})
                
        # Store word count for every word (useful for skipgram dataset)
        self.word_counts = {self.encoder[word]: count for word, count in \
                            zip(word_vocab, sorted(word_counts)[::-1])}

        # Decoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        # Store useful parameters of the tokenizer
        self.vocab_sizes = {'total': len(self.encoder),
                            'special': len(self.special_tokens),
                            'word': len(word_vocab)}
        
    def encode(self, word):
        try:
            return self.encoder[word]
        except:
            return self.encoder['[UNK]']
    
    def decode(self, token_id):
        return self.decoder[token_id]


class SubWordTokenizer():
    """ Subword-level tokenizer.

        Parameters
        ----------
        - ngram_min_len (int)
            minimum length of ngrams in tokenized subwords
        - ngram_max_len (int)
            maximum length of ngrams in tokenized subwords
        - special_tokens (list of str)
            tokens used in the model for unknown words, padding, masking,
            starting/closing sentences, etc.
        - min_freq (int)
            minimum frequency at which a word should occur in the corpus to have
            its own token_id (else: token_id of '[UNK]')
        - brackets (list of 2 str)
            special characters used to differentiate similar words and subwords
            (e.g., word '<her>' vs subword 'her' subword word <where>)
        - mode (str)
            how ngrams are computed ('subword' for classic ngrams, 'icd' for
            ngrams suited to icd codes (forward-only ngrams))
    
    """
    def __init__(self,
                 ngram_min_len, ngram_max_len, ngram_mode, ngram_base_voc,
                 special_tokens, min_freq=0, brackets=['<', '>']):
        self.encoder = dict(special_tokens)  # will be updated in fit
        self.special_tokens = dict(special_tokens)  # will stay the same
        assert ngram_min_len >= 0 and ngram_max_len >= ngram_min_len
        self.ngram_len = list(range(ngram_min_len, ngram_max_len + 1))
        self.min_freq = min_freq
        self.brackets = brackets
        self.forbidden_ngrams = brackets + list(special_tokens.keys())
        self.ngram_mode = ngram_mode
        self.ngram_base_voc = ngram_base_voc
        self.ngram_fn = self._select_ngram_fn(ngram_mode)
        
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
        for basic_token in self.ngram_base_voc:
            if basic_token in word:
                return word.replace(basic_token, ''), [basic_token]
        return word, []
    
    def _generate_icd_ngrams(self, word):
        # TODO: do this only for appropriate codes (DIA, PRO, MED?, LAB?)
        word, ngrams = self._initialize_ngrams(word)
        ngrams.extend([word[:i] for i in range(1, len(word))])
        return ngrams

    def _generate_multi_ngrams(self, word):
        word, ngrams = self._initialize_ngrams(word)
        ngrams.extend(self._flatten([self._generate_ngrams(
            word, i, self.forbidden_ngrams) for i in self.ngram_len]))
        import pdb; pdb.set_trace()
        return ngrams
    
    def _add_brackets(self, word):
        return self.brackets[0] + word + self.brackets[1]
    
    def _compute_and_sort_vocabulary(self, words_or_ngrams):
        vocab, counts = np.unique(words_or_ngrams, return_counts=True)
        if self.min_freq > 0:  # remove rare words
            vocab = vocab[counts > self.min_freq]
            counts = counts[counts > self.min_freq]
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
        self.encoder.update({word: (idx + len_so_far)
                             for idx, word in enumerate(word_vocab)})
            
        # Store word count for every word (if not in char mode)
        self.word_counts = {self.encoder[word]: count for word, count in \
                            zip(word_vocab, sorted(word_counts)[::-1])}

        # Update encoder with ngram level vocabulary
        len_so_far = len(self.encoder)
        self.encoder.update({i: (idx + len_so_far)
                             for idx, i in enumerate(ngram_vocab)})

        # Decoder
        self.decoder = {v: k if k not in word_vocab else k[1:-1]
                        for k, v in self.encoder.items()}

        # Store useful parameters of the tokenizer
        self.vocab_sizes = {'total': len(self.encoder),
                            'special': len(self.special_tokens),
                            'word': len(word_vocab),
                            'ngram': len(ngram_vocab)}
        
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
        for word_or_ngram in seq:  # first in the list is the word itself
            try:
                indices.append(self.encoder[word_or_ngram])
            except:  # this will still take the ngrams for an unknown word
                indices.append(self.encoder['[UNK]'])
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
        
