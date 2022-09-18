from torchdata.datapipes.iter import IterDataPipe


class EosBosAdder(IterDataPipe):
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
