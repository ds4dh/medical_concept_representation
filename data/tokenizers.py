import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from nltk.util import ngrams


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
        unique_words, word_counts = np.unique(words, return_counts=True)
        if self.min_freq > 0:  # remove rare words
            unique_words = unique_words[word_counts > self.min_freq]
            word_counts = word_counts[word_counts > self.min_freq]
        inds = word_counts.argsort()[::-1]
        unique_words = unique_words[inds]

        # Generate word level encoder
        self.encoder.update({i: (idx + len(self.special_tokens))
                             for idx, i in enumerate(unique_words)})
                
        # Store word count for every word (useful for skipgram dataset)
        self.word_counts = {self.encoder[word]: count for word, count in \
                            zip(unique_words, sorted(word_counts)[::-1])}

        # Decoder
        self.decoder = {v: k for k, v in self.encoder.items()}

    def encode(self, word):
        try:
            return self.encoder[word]
        except:
            return self.encoder['[UNK]']
    
    def decode(self, token_id):
        return self.decoder[token_id]


# IDEA: USE A SPECIFIC TOKENIZATION SCHEME THAT TAKES INTO ACCOUNT THE SCRUCTURE OF ICD (OR OTHER) CODES
# [XX-16-AV] -> [[X] [XX] [XX-1] [XX-16] [XX-16-A] [XX-16-AV]]


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
    def __init__(self, ngram_min_len, ngram_max_len, ngram_mode, special_tokens,
                 min_freq=0, brackets=['<', '>']):
        self.encoder = dict(special_tokens)  # will be updated in fit
        self.special_tokens = dict(special_tokens)  # will stay the same
        assert ngram_min_len <= ngram_max_len
        assert ngram_min_len >= 0
        assert ngram_max_len >= 0
        self.ngram_len = list(range(ngram_min_len, ngram_max_len + 1))
        self.min_freq = min_freq
        self.brackets = brackets
        self.forbidden_ngrams = brackets + list(special_tokens.keys())
        self.ngram_mode = ngram_mode

    @staticmethod
    def _flatten(t):
        return [item for sublist in t for item in sublist]

    @staticmethod
    def _generate_ngrams(word, length, forbidden_ngram):
        return [ngram for ngram in ["".join(i) for i in ngrams(word, length)]
                if (ngram not in forbidden_ngram) and (word[1:-1] not in ngram)]
    
    @staticmethod
    def _generate_icd_ngrams(txt):
        # TODO: do this only for appropriate codes (DIA, PRO, MED?, LAB?)
        return [txt[:i] for i in range(1, len(txt))]

    def _generate_multi_ngrams(self, word):
        return self._flatten([self._generate_ngrams(
            word, i, self.forbidden_ngrams) for i in self.ngram_len])
    
    def _add_brackets(self, word):
        return self.brackets[0] + word + self.brackets[1]
        
    def fit(self, words):
        print('Training tokenizer')
        assert isinstance(words, list)
        
        # Compute and sort vocabulary
        unique_words, word_counts = np.unique(words, return_counts=True)
        if self.min_freq > 0:  # remove rare words
            unique_words = unique_words[word_counts > self.min_freq]
            word_counts = word_counts[word_counts > self.min_freq]
        inds = word_counts.argsort()[::-1]  # most occurent first
        unique_words = unique_words[inds]

        # Compute ngram vocabulary
        if self.ngram_mode == 'subword':
            unique_whole_words = [self._add_brackets(w) for w in unique_words]
            unique_ngrams = self._flatten([self._generate_multi_ngrams(word)
                                           for word in unique_whole_words])
        elif self.ngram_mode == 'icd':
            unique_whole_words = unique_words
            unique_ngrams = self._flatten([self._generate_icd_ngrams(word)
                                           for word in unique_whole_words])
        else:
            raise ValueError('Unknown ngramization mode %s' % self.ngram_mode)

        # Sort ngram vocabulary
        unique_ngrams, ngram_counts = np.unique(
            unique_ngrams, return_counts=True)
        inds = ngram_counts.argsort()[::-1]
        unique_ngrams = unique_ngrams[inds]

        # Generate word level encoder
        self.encoder.update({word: (i + len(self.special_tokens))
                            for i, word in enumerate(unique_whole_words)})

        # Store word count for every word (useful for skipgram dataset)
        self.n_words = len(self.encoder)  # might be useful for elmo (let's see)
        self.word_counts = {self.encoder[word]: count for word, count in \
                            zip(unique_whole_words, sorted(word_counts)[::-1])}

        # Update encoder with ngram level vocabulary
        self.encoder.update({i: (idx + self.n_words)
                            for idx, i in enumerate(unique_ngrams)})

        # Decoder
        self.decoder = {v: k if k in self.special_tokens else k[1:-1]
                        for k, v in self.encoder.items()}

    def encode(self, word):
        # Generate n-grams and add them to the word
        if word not in self.special_tokens:
            bracket_word = self._add_brackets(word)
            seq = [bracket_word] + self._generate_multi_ngrams(bracket_word)
        else:
            seq = [word]

        # Encode the word and the ngram.
        indices = []
        for word_or_ngram in seq:  # first in the list is the word itself
            try:
                indices.append(self.encoder[word_or_ngram])
            except:  # this will still take the ngrams for an unknown word
                indices.append(self.encoder['[UNK]'])
                
        return indices

    def decode(self, token_id_or_ids):
        if type(token_id_or_ids) == list:
            return self.decoder[token_id_or_ids[0]]
        elif type(token_id_or_ids) == int:
            return self.decoder[token_id_or_ids]
        else:
            raise TypeError('Invalid token format: %s' % type(token_id_or_ids))


# TODO: put this at a better place (data_utils?)
def display_pca_scatterplot(model, words=None, sample=0):
    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.key_to_index), sample)
        else:
            words = [word for word in model.vocab]

    word_vectors = np.array([model[w] for w in words])
    twodim = PCA().fit_transform(word_vectors)[:, :2]

    plt.figure(figsize=(6, 6))
    plt.scatter(twodim[:, 0], twodim[:, 1], edgecolors='k', c='r')
    for word, (x, y) in zip(words, twodim):
        plt.text(x+0.05, y+0.05, word)
