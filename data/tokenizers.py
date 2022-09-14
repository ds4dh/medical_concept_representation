import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from nltk.util import ngrams


class Tokenizer():
    """
    Word-level tokenizer.
    """
    def __init__(self, special_tokens, min_freq=0):
        self.encoder = None
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
        self.encoder = self.special_tokens
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
    """
    Subword-level tokenizer.
    """

    def __init__(self,
                 ngram_min_len,
                 ngram_max_len,
                 special_tokens,
                 min_freq=0,
                 forbidden_ngram=["<", ">"]):

        self.encoder = None
        assert ngram_min_len <= ngram_max_len
        assert ngram_min_len >= 0
        assert ngram_max_len >= 0
        self.ngram_len = list(range(ngram_min_len, ngram_max_len + 1))
        self.special_tokens = special_tokens
        self.min_freq = min_freq
        self.forbidden_ngram = forbidden_ngram

    @staticmethod
    # TODO: see what type of ngram we want (Dimitris had an idea)
    def _generate_ngram(text, length, forbidden_ngram):
        return [ngram for ngram in ["".join(i) for i in ngrams(text, length)] if (ngram not in forbidden_ngram) and (text[1:-1] not in ngram)]

    @staticmethod
    def _flatten(t):
        return [item for sublist in t for item in sublist]

    def _generate_ngrams(self, text):
        return self._flatten([self._generate_ngram(text, i, self.forbidden_ngram) for i in self.ngram_len])

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

        # Add angular brackets to indicate whole words
        unique_whole_words = ['<' + i + '>' for i in unique_words]

        # Compute ngram vocabulary
        unique_ngrams = self._flatten(
            [self._generate_ngrams(word) for word in unique_whole_words])

        # Sort ngram vocabulary
        unique_ngrams, ngram_counts = np.unique(
            unique_ngrams, return_counts=True)
        inds = ngram_counts.argsort()[::-1]
        unique_ngrams = unique_ngrams[inds]

        # Generate word level encoder (using '<...>' words!)
        self.encoder = dict(self.special_tokens)  # copy
        self.encoder.update({word: (i + len(self.special_tokens))
                            for i, word in enumerate(unique_whole_words)})

        # Store word count for every word (useful for skipgram dataset)
        self.word_counts = {self.encoder[word]: count for word, count in zip(
            unique_whole_words, sorted(word_counts)[::-1])}

        # Update encoder with ngram level vocabulary
        len_so_far = len(self.encoder)
        self.encoder.update({i: (idx + len_so_far)
                            for idx, i in enumerate(unique_ngrams)})

        # Decoder
        self.decoder = {
            v: k if k in self.special_tokens else k[1:-1] for k, v in self.encoder.items()}

    def encode(self, word):
        # Generate n-grams and add them to the word
        if word not in self.special_tokens:
            angular_word = '<' + word + '>'
            seq = [angular_word] + self._generate_ngrams(angular_word)
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
