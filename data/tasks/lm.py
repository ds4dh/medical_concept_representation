import random
from torchdata.datapipes.iter import IterDataPipe


class BosEosAdder(IterDataPipe):
    """ Simple pipeline that adds eos and bos tokens around samples """
    def __init__(self, dp, tokenizer):
        super().__init__()
        self.dp = dp
        self.bos_id = tokenizer.encode('[CLS]')
        self.eos_id = tokenizer.encode('[END]')
        
    def __iter__(self):
        """ Sample format: {'src': list of token_ids (ints)
                            'tgt': list of token_ids (ints)}
        """
        for sample in self.dp:
            yield {k: self.beos_fn(v) for k, v in sample.items()}
            
    def beos_fn(self, sequence):
        """ Add start and end tokens to a sequence of tokens """
        sequence.insert(0, self.bos_id); sequence.append(self.eos_id)
        return sequence


class ElmoSetter(IterDataPipe):
    """ Parse a token sequence into a char sequence and the corresponding
        sequence of words (and add eos/bos tokens around both sequences)
    """
    def __init__(self, dp, tokenizer):
        super().__init__()
        self.dp = dp
        self.special_token_ids = list(tokenizer.special_tokens.values())
        self.n_words = tokenizer.vocab_sizes['word']
        self.elmo_set_fn = self.elmo_set_fn_for_chars
        try:
            self.bos_id_char = tokenizer.encode('[CLS]')
            self.eos_id_char = tokenizer.encode('[END]')
            self.bos_id_word = self.bos_id_char[0]
            self.eos_id_word = self.eos_id_char[0]
        except TypeError:
            self.elmo_set_fn = self.elmo_set_fn_for_words_only
            self.bos_id_word = tokenizer.encode('[CLS]')
            self.eos_id_word = tokenizer.encode('[END]')
            
    def __iter__(self):
        """ Sample format: list of token ids """
        for sample in self.dp:
            yield self.elmo_set_fn(sample)

    def beos_fn_char(self, chars):
        """ Add start and end tokens to a sequence of tokens """
        chars.insert(0, self.bos_id_char); chars.append(self.eos_id_char)
        return chars
    
    def beos_fn_word(self, words):
        """ Add start and end tokens to a sequence of tokens """
        words.insert(0, self.bos_id_word); words.append(self.eos_id_word)
        return words

    def elmo_set_fn_for_chars(self, sample):
        words = [s[0] for s in sample]
        chars = [[c - self.n_words for c in s[1:]] for s in sample]
        return {'chars': self.beos_fn_char(chars),
                'words': self.beos_fn_word(words)}

    def elmo_set_fn_for_words_only(self, sample):
        return {'chars': [0],
                'words': self.beos_fn_word(sample)}


class DynamicMasker(BosEosAdder):
    """ Take samples and replace a given proportion of token_ids by mask_id """
    def __init__(self, dp, tokenizer):
        super().__init__(dp, tokenizer)
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
        # TODO: NEED TO IMPLEMENT ACTUAL MASKING WITH PROBS TO NOT MASK OR REPLACE BY RANDOM!!!!
        # TIP: SEND UNTOUCHED SAMPLE + MASKED INDICES TO LOSS FUNCTION
        return {'masked': msk, 'target': tgt}

    def replace_fn(self, token, mask_prob=0.15):
        """ Replace a token by mask_id given a probability to mask """
        if mask_prob > random.random():
            return self.mask_id
        else:
            return token
