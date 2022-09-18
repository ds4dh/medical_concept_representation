from .fasttext import FastText
from .glove import Glove
from .transformer import Transformer
from .bert import BERT
from .bert_classifier import BERTClassifier

AVAILABLE_MODELS = {'fasttext': FastText,
                    'glove': Glove,
                    'transformer': Transformer,
                    'bert': BERT,
                    'bert_classifier': BERTClassifier}
