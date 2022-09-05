import torch
import torch.nn as nn


class FastText(nn.Module): 
    def __init__(self, vocab_size, d_embed, *args, **kwargs):
        super(FastText, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_embed)
        self.fc = nn.Linear(d_embed, vocab_size)
        self.loss_fn = FastTextLoss()

    def forward(self, center):
        """ Forward pass of the FastText model

        Args:
            center (torch.Tensor): center word whose context is to be predicted
            - shape: (batch_size, d_embed) if word-level encoding
                     (batch_size, N > 1, d_embed) if subword-level encoding

        Returns:
            torch.Tensor of shape (batch_size, vocab_size): context word logits
            
        """
        y = self.embed(center)  # (batch_size, seq_len, vec_dim)
        if len(y.shape) == 3:
            y = torch.mean(y, dim=1)  # average subwords embeddings
        return self.fc(y)  # (batch_size, label_size)

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
    