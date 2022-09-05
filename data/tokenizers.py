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


class SubWordTokenizer():
    """
    Subword-level tokenizer.
    """
    def __init__(self, ngram_len, special_tokens, min_freq=0):
        self.encoder = None
        self.ngram_len = ngram_len
        self.special_tokens = dict(special_tokens)  # copy
        self.min_freq = min_freq
        
    @staticmethod
    # TODO: see what type of ngram we want (Dimitris had an idea)
    def _generate_ngram(text, ngram_len):
        return [''.join(i) for i in ngrams(text, ngram_len)]
    
    @staticmethod
    def _flatten(t):
        return [item for sublist in t for item in sublist]
    
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
        unique_ngrams = [self._generate_ngram(word, self.ngram_len)
                         for word in unique_whole_words]
        unique_ngrams = self._flatten(unique_ngrams)
        unique_ngrams = [ngram for ngram in unique_ngrams
                         if not (ngram[0] == '<' and ngram[-1] == '>')]

        # Sort ngram vocabulary
        unique_ngrams, ngram_counts = np.unique(unique_ngrams,
                                                return_counts=True)
        inds = ngram_counts.argsort()[::-1]
        unique_ngrams = unique_ngrams[inds]

        # Generate word level encoder (using '<...>' words!)
        self.encoder = self.special_tokens
        self.encoder.update({word: (i + len(self.special_tokens))
                             for i, word in enumerate(unique_whole_words)})
        
        # Store word count for every word (useful for skipgram dataset)
        self.word_counts = {self.encoder[word]: count for word, count in \
                            zip(unique_whole_words, sorted(word_counts)[::-1])}

        # Update encoder with ngram level vocabulary
        len_so_far = len(self.encoder)
        self.encoder.update(
            {i: (idx + len_so_far) for idx, i in enumerate(unique_ngrams)})

        # Decoder
        self.decoder = {v: k for k, v in self.encoder.items()}

    def encode(self, word):
        # Generate n-grams and add them to the word
        angular_word = '<' + word + '>'
        seq = [angular_word]
        if len(word) > 1:
            seq += self._generate_ngram(angular_word, self.ngram_len)
        # seq = lst + ['[PAD]'] * (self.max_len_ngram - len(lst))
        
        # Encode the word and the ngram
        indices = []
        for word_or_ngram in seq:  # first in the list is the word itself
            try:
                indices.append(self.encoder[word_or_ngram])
            except:
                indices.append(self.encoder['[UNK]'])

        return indices
    
    def decode(self, token_id_or_ids):
        if type(token_id_or_ids) == list:
            return self.decoder[token_id_or_ids[0]][1:-1]
        elif type(token_id_or_ids) == int:
            return self.decoder(token_id_or_ids)
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
