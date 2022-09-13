import torch
import torch.nn as nn


class FastText(nn.Module): 
    def __init__(self, vocab_size, d_embed, *args, **kwargs):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_embed)
        self.fc = nn.Linear(d_embed, vocab_size)
        self.loss_fn = FastTextLoss()

    def forward(self, center):
        """ Forward pass of the FastText model

        Args:
            center (torch.Tensor): center word whose context is to be predicted
            - shape: (batch_size) if word-level encoding
                     (batch_size, N > 1) if subword-level encoding

        Returns:
            torch.Tensor of shape (batch_size, vocab_size): context word logits
            
        """
        y = self.embed(center)  # (batch_size [, n_subwords + 1], d_embed)
        if len(y.shape) > 2:
            y = torch.sum(y, dim=-2)  # sum over [word + subwords] dim
        return self.fc(y)  # (batch_size, vocab_size)

    def get_embeddings(self):
        ...

    def export_as_gensim(self, path, tokenizer):
        embeddings = self.get_embeddings()
        with open(path, 'w', encoding='utf-8') as f:
            for tok, emb in zip(tokenizer.encoder.keys(), embeddings.tolist()):
                f.write(str(tok) + ' ' + str(emb).replace('[', '') \
                                                 .replace(']', '') \
                                                 .replace(',', '') + '\n')

class FastTextLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.nlll_loss = nn.NLLLoss()

    def forward(self, model_output, context):
        logits = self.log_softmax(model_output)
        return self.nlll_loss(logits, context)
    