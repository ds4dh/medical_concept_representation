import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from nltk.util import ngrams


class Tokenizer():
    """
    Word-level tokenizer.
    """
    def __init__(self, special_tokens):
        self.encoder = None
        self.special_tokens = special_tokens

    def fit(self, words):
        assert isinstance(words, list)
        unique_words, count = np.unique(words, return_counts=True)
        inds = count.argsort()[::-1]
        unique_words = unique_words[inds]

        self.encoder = self.special_tokens
        self.encoder.update({i: (idx + len(self.special_tokens))
                             for idx, i in enumerate(unique_words)})
        self.decoder = {v: k for k, v in self.encoder.items()}

    def encode(self, word):
        try:
            return self.encoder[word]
        except:
            return self.encoder['[UNK]']


class SubWordTokenizer():
    """
    Subword-level tokenizer.
    """
    def __init__(self, ngram_len, special_tokens):
        self.encoder = None
        self.ngram_len = ngram_len
        self.special_tokens = special_tokens
        
    @staticmethod
    def _generate_ngram(text, ngram_len):
        return ["".join(i) for i in ngrams(text, ngram_len)]
    
    @staticmethod
    def _flatten(t):
        return [item for sublist in t for item in sublist]
    
    def fit(self, words):
        assert isinstance(words, list)
        # Compute and sort vocabulary.
        unique_words, count = np.unique(words, return_counts=True)
        inds = count.argsort()[::-1]  # most occurent first
        unique_words = unique_words[inds]

        # Add angular brackets to unique words.
        angular_words = ["<" + i + ">" for i in unique_words]

        # Compute and sort ngram vocabulary.
        unique_angular_ngrams = [self._generate_ngram(
            i, self.ngram_len) for i in angular_words]
        self.max_len_ngram = max([len(i) for i in unique_angular_ngrams])
        unique_angular_ngrams = self._flatten(unique_angular_ngrams)
        unique_angular_ngrams, count = np.unique(
            unique_angular_ngrams, return_counts=True)
        inds = count.argsort()[::-1]
        unique_angular_ngrams = unique_angular_ngrams[inds]

        # Generate word level encoder.
        self.encoder = self.special_tokens
        self.encoder.update({i: (idx + len(self.special_tokens))
                             for idx, i in enumerate(unique_words)})
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        # Generate ngram level encoder.
        self.ngram_encoder = self.special_tokens
        self.ngram_encoder.update(
            {i: (idx + len(self.special_tokens))
             for idx, i in enumerate(unique_angular_ngrams)})

    def encode(self, word):
        # Word level encoder.
        ind = []
        try:
            ind.append(self.encoder[word])
        except:
            ind.append(self.encoder['[UNK]'])

        # Ngram level encoder.
        lst = self._generate_ngram("<" + word + ">", self.ngram_len)
        seq = lst + ["[PAD]"] * (self.max_len_ngram - len(lst))
        
        for ng in seq:
            try:
                ind.append(self.ngram_encoder[ng])
            except:
                ind.append(self.encoder['[UNK]'])

        return ind


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
