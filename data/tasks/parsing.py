import os
import json
from itertools import groupby
from torchdata.datapipes.iter import IterDataPipe


class ReagentPredParser(IterDataPipe):
    def __init__(self, dp, task, **kwargs):
        super().__init__()
        self.dp = dp
        self.parse_fn = self.select_parse_fn(task)
        self.__dict__.update(kwargs)
        
    def __iter__(self):
        for sample in self.dp:
            yield self.parse_fn(sample)
    
    def select_parse_fn(self, task):
        """ How data is parsed
        """
        if task == 'reagent_pred_mt':
            return self.parse_reaction_for_reagent_pred_mt     
        elif task == 'reagent_pred_mlm':
            return self.parse_reaction_for_reagent_pred_mlm       
        elif task == 'reagent_pred_cls':
            return self.parse_reaction_for_reagent_pred_cls   
        else:
            raise ValueError('Invalid task given to the pipeline %s' % task)
    
    @staticmethod
    def parse_reaction_for_reagent_pred_mt(sample):
        """ Parse a dict of src (reactant(s) and product) and tgt (reagent(s))
            strings into the same dict but as a list of tokens
        """
        return {k: v.split(' ') for k, v in sample.items()}
    
    @staticmethod
    def parse_reaction_for_reagent_pred_mlm(sample):
        """ Parse a dict of src (reactant(s) and product) and tgt (reagent(s))
            strings into a list of tokens representing the smiles reaction
        """
        product = sample['src'].split('.')[-1]
        reactants = sample['src'].replace(product, '')[:-1]
        reagents = sample['tgt']
        reaction = ' > '.join([reactants, reagents, product])
        return reaction.replace('  ', ' ').split(' ')
    
    def parse_reaction_for_reagent_pred_cls(self, sample):
        """ Parse a reaction sample into a list of tokens for reactant(s) and
            product, and a one-hot encoded reagent(s) label vector
        """
        if not hasattr(self, 'reagent_map'):
            self.reagent_map = self.get_reagent_map()
        sample = self.parse_reaction_for_reagent_pred_mlm(sample)
        splits = [list(g) for _, g in groupby(sample, key='>'.__ne__)][::2]
        sample = splits[0] + ['>', '>'] + splits[-1]  # reactants > > product        
        reagents = ''.join(splits[1]).split('.')
        reagent_labels = [self.reagent_map[r] for r in reagents]
        return {'sample': sample, 'label': reagent_labels}
    
    def get_reagent_map(self):
        """ Generate a map from any reagent to its corresponding label id
        """
        reagent_path = os.path.join(self.data_dir, 'reagent_popularity.json')
        with open(reagent_path, 'r') as f:
            dicts = [json.loads(line) for line in f.readlines()]
            if len(dicts) > self.n_classes: dicts = dicts[:self.n_classes]
            return {d['reagent'].replace(' ', ''): i
                    for i, d in enumerate(dicts)}
            