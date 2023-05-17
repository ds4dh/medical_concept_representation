import random
from itertools import compress
from torchdata.datapipes.iter import IterDataPipe


# Adapted from https://github.com/Andras7/word2vec-pytorch
class SkipGramMaker(IterDataPipe):
    """ Compute all possible skipgram pairs from the source pipeline and load
        them to memory, then get ready to yield the computed sample pairs
    """
    def __init__(self, dp, tokenizer, model_params):
        super().__init__()
        self.tokenizer = tokenizer
        self.unk_token_id = tokenizer.encode('[UNK]')
        self.n_neg_samples = model_params['n_neg_samples']
        self.use_fixed_context = model_params['use_fixed_context']
        self.max_context_size = model_params['max_context_size']
        self.subsample_probs = self.compute_subsample_probs()
        if self.n_neg_samples > 0: self.init_neg_list(tokenizer)
        self.dp = dp
        
    def __iter__(self):
        """ Sample format: {'center': token_id (int) or ngrams (list of ints),
                            'context': token_id (int) or ngrams (list of ints)}
        """
        for sentence in self.dp:
            subsampled_sentence = self.subsample_sentence(sentence)
            skipgram_samples = self.create_skipgram_samples(subsampled_sentence)
            for sample in skipgram_samples:
                yield sample
    
    def compute_subsample_probs(self, thresh=1e-4):
        """ Compute the subsample probability for each token id
        """
        word_counts = self.tokenizer.word_counts
        sum_of_all_word_counts = sum(word_counts.values())
        special_tokens = self.tokenizer.special_tokens.values()
        subsample_probs = {v: 1.0 for v in special_tokens}
        
        for token_id, word_occurence in word_counts.items():
            word_fraction = word_occurence / sum_of_all_word_counts
            keep_ratio = (thresh / word_fraction)
            keep_score = keep_ratio ** 0.5  # + keep_ratio
            subsample_probs[token_id] = min(keep_score, 1.0)
            
        return subsample_probs
    
    def subsample_sentence(self, sentence):
        """ Subsample tokens in a sentence (to skip common tokens)
        """
        if isinstance(sentence[0], list):
            check_sentence = [token_id[0] for token_id in sentence]
        else:
            check_sentence = sentence
            
        probs = [self.subsample_probs[token_id] for token_id in check_sentence]
        selection = [p > random.random() for p in probs]
        subsampled_sentence = list(compress(sentence, selection))
        
        return subsampled_sentence
    
    def create_skipgram_samples(self, sample):
        """ Generate a set of (context, target) pairs from sentence list
        """
        # Initialize sample pairs and fixed context
        sample_pairs = list()
        sample = [s for s in sample if s != self.unk_token_id]
        fixed_context, loc_context = self.compute_fixed_context(sample)
        for center_pos, center_token_id in enumerate(sample):
            
            # Intialize context around center word and add fixed context
            context = self.init_context(sample, center_pos, center_token_id,
                                        fixed_context, loc_context)
            for context_token_id in context:
                
                # Softmax case
                if self.n_neg_samples == 0:
                    if isinstance(center_token_id, list):  # word index needed
                        context_token_id = context_token_id[0]
                    sample_pairs.append({'pos_center': center_token_id,
                                         'pos_context': context_token_id})
                
                # Negative sampling case
                else:
                    neg_context_ids = self.get_neg_samples()
                    sample_pairs.append({'pos_center': center_token_id,
                                         'pos_context': context_token_id,
                                         'neg_context': neg_context_ids})
                    
            # Update location context (after computing co-occurrence)
            loc_context = self.update_loc_context(
                loc_context, center_token_id, center_pos)
            
        return sample_pairs
    
    
    def init_context(self, sample, center_pos, center_token_id,
                     fixed_context, loc_context):
        """ ...
        """
        context_size = random.randint(1, self.max_context_size)
        left_ward = max(0, center_pos - context_size)
        right_ward = center_pos + context_size + 1
        context = sample[left_ward:right_ward]
        context.pop(context.index(center_token_id))
        
        if self.use_fixed_context:
            context.extend([c['token_id'] for c in fixed_context + loc_context
                            if c['pos'] != center_pos])
            
        return [c for c in context if c != self.unk_token_id]
    
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
    
    def init_neg_list(self, neg_table_size=1e6):
        """ ...
        """
        print(' - Initializing negative sample list')
        word_ids, word_counts = self.tokenizer.word_counts.items()
        sqrt_word_counts = [c ** 0.5 for c in word_counts]
        word_powers = sqrt_word_counts / sum(sqrt_word_counts) * neg_table_size
        self.negatives = []
        for word_id, power in zip(word_ids, word_powers):
            self.negatives += [word_id] * int(power)
        random.shuffle(self.negatives)
        self.neg_cursor = 0
    
    def get_neg_samples(self):
        """ ...
        """
        neg_samples = self.negatives[
            self.neg_cursor:self.neg_cursor + self.n_neg_samples]
        self.neg_cursor = (self.neg_cursor + self.n_neg_samples)
        self.neg_cursor = self.neg_cursor % len(self.negatives)  # back to start
        if len(neg_samples) != self.n_neg_samples:
            neg_samples += self.negatives[0:self.neg_cursor]
        return neg_samples
