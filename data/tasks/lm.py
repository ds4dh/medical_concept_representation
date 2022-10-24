import random
import numpy as np
from torchdata.datapipes.iter import IterDataPipe


class LMSetter(IterDataPipe):
    """ Add tokens encoding for beginning and end of sentence
    """
    def __init__(self, dp, tokenizer):
        super().__init__()
        self.dp = dp
        self.bos_id = tokenizer.encode('[CLS]')
        self.eos_id = tokenizer.encode('[END]')
            
    def __iter__(self):
        """ Sample format: list of token ids """
        for sample in self.dp:
            yield {'sample': self.beos_fn(sample)}

    def beos_fn(self, sample):
        """ Add start and end tokens to a sequence of tokens """
        sample.insert(0, self.bos_id); sample.append(self.eos_id)
        return sample


class DynamicMasker(LMSetter):
    """ Take samples and replace a given proportion of token_ids by mask_id """
    def __init__(self,
                 dp, tokenizer, mask_prob=0.15, keep_prob=0.1, rand_prob=0.1):
        super().__init__(dp, tokenizer)
        self.dp = dp
        last_special_id = tokenizer.vocab_sizes['special']
        last_word_id = last_special_id + tokenizer.vocab_sizes['word']
        words = list(tokenizer.encoder.values())[last_special_id:last_word_id]
        self.word_ids = [tokenizer.encode(tokenizer.decode(v)) for v in words]
        self.mask_id = tokenizer.encode('[MASK]')
        self.bos_id = tokenizer.encode('[CLS]')
        self.eos_id = tokenizer.encode('[END]')
        self.probs = [1.0 - mask_prob,
                      mask_prob * (1.0 - keep_prob - rand_prob),
                      mask_prob * keep_prob,
                      mask_prob * rand_prob]
        
    def __iter__(self):
        """ Sample format: {'masked': list of token_ids (ints)
                                           or ngrams (lists of ints),
                            'target': list of token_ids (only whole words)}
        """
        for sample in self.dp:
            yield {k: v for k, v in self.mask_fn(sample).items()}

    def replace_fn(self, token_id, mask_value):
        """ Replace token by the value it must follow, depending on the mask """
        if mask_value in [0, 2]:
            return token_id
        elif mask_value == 1:
            return self.mask_id
        else:
            return random.choice(self.word_ids)

    def mask_fn(self, sample, ):
        """ Implement dynamic masking (i.e., called at yield-time) by replacing
            a percentage of the sample tokens by the id of '[MASK]'.
            - Note: '[BOS]' and '[EOS]' tokens are added to the sequences, but
                are not subject to masking, i.e., added after
            - Note: for the ngram case, tgt sequences only contain whole words
        """
        assert isinstance(sample, list), 'Bad sample type %s' %  type(sample)

        # Mask tokens and add bos and eos ids afterwards, to avoid masking them
        mask = np.random.choice([0, 1, 2, 3], p=self.probs, size=len(sample))
        masked_sample = list(map(self.replace_fn, sample, mask))
        masked_sample = self.beos_fn(masked_sample)

        # Retrieve target tokens (and keep whole words only for ngram case)
        masked_ids, = np.where(mask)  # comma is important
        if isinstance(sample[0], list):
            target_token_ids = [sample[id][0] for id in masked_ids]
        else:
            target_token_ids = [sample[id] for id in masked_ids]

        # Return the sample to the pipeline
        masked_ids = (masked_ids + 1).tolist()  # take care of bos token
        return {'masked': masked_sample,
                'masked_label_ids': masked_ids,
                'masked_label': target_token_ids}
