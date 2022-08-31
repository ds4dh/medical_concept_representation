import torch
import torch.nn as nn


class FastText(nn.Module): 
    def __init__(self, vocab_size, d_embed, *args, **kwargs):
        super(FastText, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_embed)
        self.fc = nn.Linear(d_embed, vocab_size)

    def forward(self, x):
        x = self.embed(x)  # (batch_size, seq_len, vec_dim)
        y = torch.mean(x, dim=1)  # average over the word embeddings
        return self.fc(y)  # (batch_size, label_size)
