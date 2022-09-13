import os
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from torchdata.datapipes.iter import IterDataPipe
from ..data_utils import load_dp, save_dp


class CoocMaker(IterDataPipe):
    """ Compute co-occurence matrix from the source pipeline, and load it to
        memory as a flat array, then get ready to yield the computed samples
    """
    def __init__(self, dp, tokenizer, data_dir, split, load_data=False):
        save_or_load_path = os.path.join(data_dir, f'cooc_{split}')
        if load_data:
            self.dp = load_dp(save_or_load_path)
        else:
            raw_cooc = self.create_cooc(list(dp))
            filtered_cooc = self.filter_cooc(raw_cooc)
            self.dp = self.format_cooc(filtered_cooc, tokenizer)
            if 0:
                save_dp(self.dp, save_or_load_path)
        
    def __iter__(self):
        """ Sample format: {'left': token_id (int) or ngram (list of ints),
                            'right': token_id (int) or ngram (list of ints),
                            'cooc': float}
        """
        for sample in self.dp:
            yield sample
    
    @staticmethod
    def create_cooc(document):
        """ Compute co-occurrences between tokens in document sentences """
        # Initialize cooccurrence matrix using document vocabulary
        print(' - Initializing co-occurrence matrix')
        if isinstance(document[0][0], list):  # for ngram encoding
            document = [[ids[0] for ids in sample] for sample in document]
        flattened_document = [id_ for sample in document for id_ in sample]
        token_ids = np.unique(flattened_document).tolist()
        cooc = OrderedDict((id_, OrderedDict((id_, 0) for id_ in token_ids))
                            for id_ in token_ids)
        
        # For each token pair, fill-in how many times they co-occur
        for sample in tqdm(document, desc=' - Filling co-occurrence matrix'):
            for i, center in enumerate(sample):
                for context in sample[:i] + sample[i+1:]:
                    cooc[center][context] += 1
        
        return cooc

    @staticmethod
    def filter_cooc(cooc_, min_cooc=1):
        """ Filters out token co-occurences that are below a threshold """
        assert min_cooc > 0, 'Tokens must co-occur at least once because \
                              GloVe is a log-regression model.'
        
        # Keep only co-occurrences that are above a given threshold
        cooc = cooc_.copy()
        for k, v in tqdm(cooc.items(),
                         desc=' - Filtering co-occurrence array'):
            cooc[k] = {x: y for x, y in v.items() if y >= min_cooc}
        
        return cooc
    
    @staticmethod
    def format_cooc(cooc, tokenizer):
        """ Format co-occurrence dict to left/right inputs and cooc target """
        # Take care of the case where token encoding uses ngrams
        ngram_map = {}
        for token in cooc.keys():
            ngram_map[token] = tokenizer.encode(tokenizer.decode(token))
        
        # Format the co-occurrence matrix as requested by the models
        samples = list()
        for l_token, r_dict in tqdm(cooc.items(),
                                    desc=' - Formatting co-occurrence array'):
            for r_token, cooc_i in r_dict.items():
                samples.append({'left': ngram_map[l_token],
                                'right': ngram_map[r_token],
                                'cooc': float(cooc_i)})
        
        return samples
