import torch
import torch.nn as nn


class Glove(nn.Module):
    """ Glove model
    """

    def __init__(self, vocab_sizes, special_tokens, d_embed, *args, **kwargs):
        super().__init__()
        assert special_tokens['[PAD]'] == 0, 'For this model, pad_id must be 0'
        pad_id = special_tokens['[PAD]']
        vocab_size = vocab_sizes['total']
        self.l_emb = nn.Embedding(vocab_size, d_embed, padding_idx=pad_id)
        self.l_bias = nn.Embedding(vocab_size, 1, padding_idx=pad_id)
        self.r_emb = nn.Embedding(vocab_size, d_embed, padding_idx=pad_id)
        self.r_bias = nn.Embedding(vocab_size, 1, padding_idx=pad_id)
        self.loss_fn = GloveLoss()

    def forward(self, left, right):
        # Compute embeddings
        l_v = self.l_emb(left)
        l_b = self.l_bias(left)
        r_v = self.r_emb(right)
        r_b = self.r_bias(right)
        
        # Mean over ngram dimension if existing
        if len(l_v.shape) > 2:  # mean should be consistent with padding tokens
            l_v = self.combine_ngram_embeddings(l_v, dim=-2)
            l_b = self.combine_ngram_embeddings(l_b, dim=-2)
            r_v = self.combine_ngram_embeddings(r_v, dim=-2)
            r_b = self.combine_ngram_embeddings(r_b, dim=-2)
                    
        return (l_v * r_v).sum(dim=-1) + l_b.squeeze() + r_b.squeeze()
    
    def combine_ngram_embeddings(self, x, dim, reduce='mean'):
        if reduce == 'mean':
            norm_factor = (x != 0).sum(dim=dim).clip(min=1) / x.shape[dim]
            return x.mean(dim=dim) / norm_factor
        else:
            return x.sum(dim=dim)
    
    def get_token_embeddings(self, token_indices):
        """ Compute static embeddings for a list of tokens
        """
        all_embeddings = self.l_emb.weight + self.r_emb.weight
        token_embeddings = []
        for token_index in token_indices:
            embedded = all_embeddings[token_index]
            if len(embedded.shape) > 1:  # ngram case
                embedded = self.combine_ngram_embeddings(embedded, dim=-2)
            token_embeddings.append(embedded)
        return torch.stack(token_embeddings, dim=0).detach().cpu()
    
    def get_sequence_embeddings(self, sequence, weights=None):
        """ Compute static embedding for a sequence of tokens
        """
        embeddings = self.get_token_embeddings(sequence)
        return self.collapse_sequence_embeddings(embeddings, weights)
    
    def collapse_sequence_embeddings(self, embeddings, weights, dim=-2):
        """ Average sequence embedding over sequence dimension
        """
        if weights == None:  # classic average
            return embeddings.mean(dim=dim)
        else:  # weighted average
            weights = torch.tensor(weights, dtype=embeddings.dtype)
            return embeddings.T @ weights / weights.sum()
        

class GloveLoss(nn.Module):
    """ Loss for the GloVe Model
    """
    def __init__(self, x_max=100, alpha=3/4, reduce='sum'):
        """ Hyperparameters as in the original article.
        """
        super().__init__()
        self.m = x_max
        self.a = alpha
        self.reduce = reduce

    def normalize(self, t):
        """ Normalization as in the original article.
        """
        return torch.where(t < self.m,
                           (t / self.m) ** self.a,
                           torch.ones_like(t))
    
    def forward(self, model_output, cooc):
        """ Expects flattened model output and target coocurence matrix.
        """
        n_t = self.normalize(cooc)
        l_t = torch.log(cooc)
        if self.reduce == 'mean':
            return torch.mean(n_t * (model_output - l_t) ** 2)
        elif self.reduce == 'sum':
            return torch.sum(n_t * (model_output - l_t) ** 2)
        else:
            raise ValueError('Invalid reduce mode')
