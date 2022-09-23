from .fasttext import FastText
from .glove import Glove
from .elmo import ELMO
from .transformer import Transformer
from .bert import BERT
from .bert_classifier import BERTClassifier
from .model_utils import (
    load_model_and_params_from_config,
    load_checkpoint,
    update_and_save_config,
    set_environment
)

AVAILABLE_MODELS = {'fasttext': FastText,
                    'glove': Glove,
                    'elmo': ELMO,
                    'transformer': Transformer,
                    'bert': BERT,
                    'bert_classifier': BERTClassifier}
