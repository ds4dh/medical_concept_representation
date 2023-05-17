import os
import numpy as np
import pickle
from tqdm import tqdm
from collections import OrderedDict
from torchdata.datapipes.iter import IterDataPipe


class CoocMaker(IterDataPipe):
    """ Compute co-occurence matrix from the source pipeline, and load it to
        memory as a flat array, then get ready to yield the computed samples
    """
    def __init__(self, dp, tokenizer, data_dir, split, model_params):
        super().__init__()
        self.tokenizer = tokenizer
        self.unk_token_id = tokenizer.encode('[UNK]')
        self.use_whole_sentence = model_params['use_whole_sentence']
        self.use_fixed_context = model_params['use_fixed_context']
        self.left_context_size = model_params['left_context_size']
        self.right_context_size = model_params['right_context_size']
        unique_cooc_name = '%s_w%s_f%s_l%s_r%s_%s' %(
            split,
            int(self.use_whole_sentence),
            int(self.use_fixed_context), 
            self.left_context_size, 
            self.right_context_size,
            tokenizer.unique_id
        )
        load_path = os.path.join(data_dir, 'cooc_data', unique_cooc_name)
        data_loaded = False
        if model_params['load_cooc_data']:
            print(' - Loading co-occurence matrix from %s' % load_path)
            try:
                self.dp = self.load_dp(load_path)
                data_loaded = True
            except:
                print(' - Co-occurrence data not found, rebuilding it')
        if not data_loaded:
            if self.is_ngram_encoding(dp):
                dp = ([ids[0] for ids in sample] for sample in dp)
            base_cooc = self.init_cooc(dp)
            raw_cooc = self.fill_cooc(dp, base_cooc)
            filtered_cooc = self.filter_cooc(raw_cooc)
            self.dp = self.format_cooc(filtered_cooc, tokenizer)
            self.save_dp(self.dp, load_path)
            print(' - Saved co-occurrence matrix at %s' % load_path)
            
    def __iter__(self):
        """ Sample format: {'left': token_id (int) or ngram (list of ints),
                            'right': token_id (int) or ngram (list of ints),
                            'cooc': float}
        """
        for sample in self.dp:
            yield sample
    
    def init_cooc(self, dp):
        """ Initialize co-occurrences using tokenizer vocabulary
        """
        print(' - Initializing co-occurrence matrix')
        flattened_document = (token_id for sample in dp for token_id in sample
                              if token_id != self.unk_token_id)
        token_ids = np.unique(np.fromiter(flattened_document, dtype=int))
        cooc = OrderedDict((id_, OrderedDict((id_, 0) for id_ in token_ids))
                           for id_ in token_ids)
        return cooc
    
    def is_ngram_encoding(self, dp):
        """ Check if the tokens in a data pipelines have ngram encoding
            - The operation should not modify the iterable style dp (I think)
        """
        for sample in dp:  # this will only go one step
            if isinstance(sample[0], list):  # i.e., ngram encoding
                return True
            return False
    
    def fill_cooc(self, dp, cooc):
        """ Fill co-occurrence matrix based on a document of token samples
        """
        # Compute token co-occurence using all dataset samples
        for sample in tqdm(dp, desc=' - Filling co-occurrence'):
            sample = [s for s in sample if s != self.unk_token_id]
            fixed_context, loc_context = self.compute_fixed_context(sample)
            for center_pos, center_token_id in enumerate(sample):

                # Intialize context around center word and add fixed context
                context = self.init_context(sample, center_pos, center_token_id,
                                            fixed_context, loc_context)
                for context_token_id in context:
                    cooc[center_token_id][context_token_id] += 1
                        
                # Update location context (after computing co-occurrence)
                loc_context = self.update_loc_context(
                    loc_context, center_token_id, center_pos)
        
        return cooc

    def init_context(self, sample, center_pos, center_token_id,
                     fixed_context, loc_context):
        """ ...
        """
        if self.use_whole_sentence:
            context = [s for s in sample if s != self.unk_token_id]
            context.pop(context.index(center_token_id))
            return context

        left_ward = max(0, center_pos - self.left_context_size)
        right_ward = center_pos + self.right_context_size + 1
        context = sample[left_ward:right_ward]
        context.pop(context.index(center_token_id))
        
        if self.use_fixed_context:
            context.extend([c['token_id'] for c in fixed_context + loc_context
                            if c['pos'] != center_pos])
            
        context = [c for c in context if c != self.unk_token_id]
        return context
    
    def compute_fixed_context(self, sample, n_diagnoses=3):
        """ ...
        """
        fixed_context, loc_context = [], []
        if self.use_fixed_context:
            
            dems = [{'pos': p, 'token_id': t} for p, t in enumerate(sample)
                    if 'DEM_' in self.tokenizer.decode(t)]
            lbls = [{'pos': p, 'token_id': t} for p, t in enumerate(sample)
                    if 'LBL_' in self.tokenizer.decode(t)]
            dias = [{'pos': p, 'token_id': t} for p, t in enumerate(sample)
                    if 'DIA_' in self.tokenizer.decode(t)]
            fixed_context += dems + lbls + dias[:n_diagnoses]
            fixed_context = [c for c in fixed_context
                             if c['token_id'] != self.unk_token_id]
        
        return fixed_context, loc_context
    
    def update_loc_context(self, loc_context, center_token_id, center_pos):
        """ ...
        """
        loc_context = []
        if self.use_fixed_context\
        and 'LOC_' in self.tokenizer.decode(center_token_id):
            
            new_loc_entry = {'pos': center_pos, 'token_id': center_token_id}
            if loc_context and center_pos - loc_context[-1]['pos'] > 1:
                loc_context = [new_loc_entry]
            else:
                loc_context.append(new_loc_entry)
                
        return loc_context
    
    @staticmethod
    def filter_cooc(cooc_, min_cooc=10):
        """ Filters out token co-occurences that are below a threshold
        """
        assert min_cooc > 0, 'Tokens must co-occur at least once because \
                              GloVe is a log-regression model.'
        
        # Keep only co-occurrences that are above a given threshold
        cooc = cooc_.copy()
        for k, v in tqdm(cooc.items(),
                         desc=' - Filtering co-occurrence'):
            cooc[k] = {x: y for x, y in v.items() if y >= min_cooc}
        
        return cooc
    
    @staticmethod
    def format_cooc(cooc, tokenizer):
        """ Format co-occurrence dict to left/right inputs and cooc target
        """
        # Take care of the case where token encoding uses ngrams (very edgy)
        ngram_map = {}
        for token in cooc.keys():
            ngram_map[token] = tokenizer.encode(tokenizer.decode(token))
        
        # Format the co-occurrence matrix as requested by the models
        samples = list()
        for l_token, r_dict in tqdm(cooc.items(),
                                    desc=' - Formatting co-occurrence'):
            for r_token, cooc_i in r_dict.items():
                samples.append({'left': ngram_map[l_token],
                                'right': ngram_map[r_token],
                                'cooc': float(cooc_i)})
        
        return samples
    
    @staticmethod
    def save_dp(dp, save_path):
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
        with open(save_path, 'wb') as handle:
            pickle.dump(dp, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    @staticmethod
    def load_dp(load_path):
        with open(load_path, 'rb') as file:
            saved_dp = pickle.load(file)
            return saved_dp
