from .fasttext import FastText
from .bert import BERT
from .bert_classifier import BERTClassifier
from .glove import Glove

AVAILABLE_MODELS = {'fasttext': FastText,
                    'glove': Glove,
                    'bert': BERT,
                    'bert_classifier': BERTClassifier}
