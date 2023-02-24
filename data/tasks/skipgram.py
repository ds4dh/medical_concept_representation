import random
from itertools import compress
from torchdata.datapipes.iter import IterDataPipe


# Adapted from https://github.com/Andras7/word2vec-pytorch
class SkipGramMaker(IterDataPipe):
    """ Compute all possible skipgram pairs from the source pipeline and load
        them to memory, then get ready to yield the computed sample pairs
    """
    def __init__(self, dp, tokenizer, n_neg_samples):
        super().__init__()
        self.subsample_probs = self.compute_subsample_probs(tokenizer)
        self.n_neg_samples = n_neg_samples
        if self.n_neg_samples > 0:
            self.init_neg_list(tokenizer)
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
    
    @staticmethod
    def compute_subsample_probs(tokenizer, thresh=1e-4):
        """ Compute the subsample probability for each token id
        """
        word_counts = tokenizer.word_counts
        sum_of_all_word_counts = sum(word_counts.values())
        subsample_probs = {v: 1.0 for v in tokenizer.special_tokens.values()}

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

    def create_skipgram_samples(self, sentence, max_context_size=5):
        """ Generate a set of (context, target) pairs from sentence list
        """
        sample_pairs = list()
        for center_pos, center_token_id in enumerate(sentence):
            context_size = random.randint(1, max_context_size)
            for i in range(-context_size, context_size + 1):

                # Find context word position and skip if outside sentence
                context_pos = center_pos + i
                if not 0 < context_pos < len(sentence) or i == 0:
                    continue
                
                # Softmax case
                if self.n_neg_samples == 0:
                    context_token_id = sentence[context_pos]
                    if isinstance(center_token_id, list):  # word index needed
                        context_token_id = context_token_id[0]
                    sample_pairs.append({'pos_center': center_token_id,
                                         'pos_context': context_token_id})
                
                # Negative sampling case
                else:
                    context_token_id = sentence[context_pos]
                    neg_context_ids = self.get_neg_samples()
                    sample_pairs.append({'pos_center': center_token_id,
                                         'pos_context': context_token_id,
                                         'neg_context': neg_context_ids})

        return sample_pairs

    def init_neg_list(self, tokenizer, neg_table_size=1e6):
        print(' - Initializing negative sample list')
        word_ids = list(tokenizer.word_counts.keys())
        sqrt_word_counts = [c ** 0.5 for c in tokenizer.word_counts.values()]
        word_powers = sqrt_word_counts / sum(sqrt_word_counts) * neg_table_size
        self.negatives = []
        for word_id, power in zip(word_ids, word_powers):
            self.negatives += [word_id] * int(power)
        random.shuffle(self.negatives)
        self.neg_cursor = 0
    
    def get_neg_samples(self):
        neg_samples = self.negatives[
            self.neg_cursor:self.neg_cursor + self.n_neg_samples]
        self.neg_cursor = (self.neg_cursor + self.n_neg_samples)
        self.neg_cursor = self.neg_cursor % len(self.negatives)  # back to start
        if len(neg_samples) != self.n_neg_samples:
            neg_samples += self.negatives[0:self.neg_cursor]
        return neg_samples
