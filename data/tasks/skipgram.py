import os
import random
from tqdm import tqdm
from itertools import compress
from torchdata.datapipes.iter import IterDataPipe
from ..data_utils import load_dp, save_dp


class SkipGramMaker(IterDataPipe):
    """ Compute all possible skipgram pairs from the source pipeline and load
        them to memory, then get ready to yield the computed sample pairs
    """
    def __init__(self, dp, tokenizer, data_dir, split, load_data=False):
        super().__init__()
        save_or_load_path = os.path.join(data_dir, f'skipgram_{split}')
        if load_data:
            self.dp = load_dp(save_or_load_path)
        else:
            subsample_probs = self.compute_subsample_probs(tokenizer)
            subsampled_dp = self.subsample_document(dp, subsample_probs)
            self.dp = self.create_skipgram(subsampled_dp)
            if 0:
                save_dp(self.dp, save_or_load_path)
            
    def __iter__(self):
        """ Sample format: {'center': token_id (int) or ngrams (list of ints),
                            'context': token_id (int) or ngrams (list of ints)}
        """
        for sample in self.dp:
            yield sample
    
    @staticmethod
    def compute_subsample_probs(tokenizer, thresh=1e-4):
        """ Compute the subsample probability for each token id """
        word_counts = tokenizer.word_counts
        sum_of_all_word_counts = sum(word_counts.values())
        subsample_probs = {}
        for token_id, word_occurence in word_counts.items():
            word_fraction = word_occurence / sum_of_all_word_counts
            keep_score = (thresh / word_fraction) ** 0.5
            subsample_probs[token_id] = min(keep_score, 1.0)
        
        return subsample_probs
    
    @staticmethod
    def subsample_document(document, subsample_probs):
        """ Subsample tokens in a document (to skip common tokens) """
        subsampled_document = []
        for sentence in tqdm(document, desc=' - Subsampling document'):
            subsampled_sentence = []
            if isinstance(sentence[0], list):
                check_sentence = [token_id[0] for token_id in sentence]
            else:
                check_sentence = sentence
            probs = [subsample_probs[token_id] for token_id in check_sentence]
            selection = [p > random.random() for p in probs]
            subsampled_sentence = list(compress(sentence, selection))
            if subsampled_sentence:
                subsampled_document.append(subsampled_sentence)

        return subsampled_document

    @staticmethod
    def create_skipgram(document, max_context_size=5):
        """ Generate a set of (context, target) pairs from sentence list """
        sample_pairs = list()
        for sentence in tqdm(document, desc=' - Creating skipgram pairs'):
            for center_pos, center_token_id in enumerate(sentence):
                context_size = random.randint(1, max_context_size)
                for i in range(-context_size, context_size + 1):
                    # Find context word position and skip if outside sentence
                    context_pos = center_pos + i
                    if not 0 < context_pos < len(sentence) or i == 0:
                        continue

                    # Retrieve context word
                    context_token_id = sentence[context_pos]
                    if isinstance(center_token_id, list):  # for ngram encoding
                        context_token_id = context_token_id[0]
                    
                    # Update the sample pair list
                    sample_pairs.append({'center': center_token_id,
                                         'context': context_token_id})
                    
        return sample_pairs
