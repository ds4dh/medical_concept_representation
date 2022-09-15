import os
import random
import json
from itertools import groupby
from torchdata.datapipes.iter import IterDataPipe


class DynamicMasker(IterDataPipe):
    """ Take samples and replace a given proportion of token_ids by mask_id """
    def __init__(self, dp, tokenizer):
        super().__init__()
        self.dp = dp
        self.mask_id = tokenizer.encode('[MASK]')
        self.bos_id = tokenizer.encode('[CLS]')
        self.eos_id = tokenizer.encode('[END]')
        
    def __iter__(self):
        """ Sample format: {'masked': list of token_ids (ints)
                                           or ngrams (lists of ints),
                            'target': list of token_ids (only whole words)}
        """
        for sample in self.dp:
            yield {k: v for k, v in self.mask_fn(sample).items()}

    def beos_fn(self, sample):
        """ Add start and end tokens to a sequence of tokens """
        sample.insert(0, self.bos_id); sample.append(self.eos_id)
        return sample

    def mask_fn(self, sample):
        """ Implement dynamic masking (i.e., called at yield-time) by replacing
            a percentage of the sample tokens by the id of '[MASK]'.
            - Note: '[BOS]' and '[EOS]' tokens are added to the sequences, but
                are not subject to masking! (added after)
            - Note: for the ngram case, tgt sequences only contain whole words
        """
        assert isinstance(sample, list), 'Bad sample type %s' %  type(sample)
        
        # Mask tokens and add bos and eos ids afterwards, to avoid masking them
        msk = list(map(self.replace_fn, sample))
        msk = self.beos_fn(msk)

        # Add bos and eos ids, then collapse to whole words only for ngram case
        tgt = self.beos_fn(sample)
        tgt = [s[0] for s in sample] if isinstance(sample[0], list) else sample
        
        return {'masked': msk, 'target': tgt}

    def replace_fn(self, token, mask_prob=0.15):
        """ Replace a token by mask_id given a probability to mask """
        if mask_prob > random.random():
            return self.mask_id
        else:
            return token
    
    
class ReagentPredMaker(IterDataPipe):
    def __init__(self, dp, data_dir, n_classes):
        super().__init__()
        self.dp = dp
        self.reagent_map = self.get_reagent_info(data_dir, n_classes)

    def __iter__(self):
        for sample in self.dp:
            yield self.parse_reaction_and_reagents_fn(sample)
    
    def parse_reaction_and_reagents_fn(self, sample):
        """ Parse a reaction sample into a list of tokens for reactant(s) and
            product, and a one-hot encoded reagent(s) label vector """
        splits = [list(g) for _, g in groupby(sample, key='>'.__ne__)][::2]
        sample = splits[0] + ['>', '>'] + splits[-1]  # reactants > > product
        reagents = ''.join(splits[1]).split('.')
        reagent_labels = [self.reagent_map[r] for r in reagents]
        return {'sample': sample, 'reagent_label': reagent_labels}
    
    def get_reagent_info(self, data_dir, n_classes):
        """ Map reagents to label id, and provide weights to counterbalance 
            class imabalance
        """
        with open(os.path.join(data_dir, 'reagent_popularity.json'), 'r') as f:
            dicts = [json.loads(line) for line in f.readlines()]
            if len(dicts) > n_classes: dicts = dicts[:n_classes]
            return {d['reagent'].replace(' ', ''): i for i, d in enumerate(dicts)}
        
        