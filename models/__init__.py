from .fasttext import FastText
from .bert import BERT
from .glove import Glove
AVAILABLE_MODELS = {'fasttext': FastText,
                    'glove': Glove,
                    'bert': BERT}