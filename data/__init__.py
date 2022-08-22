import os
from .dataset_fn import CBOWDataHolder
cwd = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = cwd  # os.path.join(cwd, 'datasets')
