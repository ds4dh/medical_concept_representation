from .word2vec import FastText, Word2Vec
from .glove import Glove
from .elmo import ELMO
from .transformer import Transformer
from .bert import BERT
from .fnet import FNet
from .bert_classifier import BERTClassifier
from .model_utils import (
    load_model_and_params_from_config,
    # load_checkpoint,
    # update_and_save_config,
    set_environment
)

AVAILABLE_MODELS = {'word2vec': Word2Vec,
                    'fasttext': FastText,
                    'glove': Glove,
                    'elmo': ELMO,
                    'transformer': Transformer,
                    'bert': BERT,
                    'fnet': FNet,
                    'bert_classifier': BERTClassifier}
