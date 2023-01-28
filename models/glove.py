import torch
import torch.nn as nn


class Glove(nn.Module):
    """ Glove model
    """

    def __init__(self,
                 vocab_sizes: dict,
                 special_tokens: dict,
                 d_embed: int,
                 *args, **kwargs):
        super().__init__()
        pad_id = special_tokens['[PAD]']
        vocab_size = vocab_sizes['total']
        self.l_emb = nn.Embedding(vocab_size, d_embed, padding_idx=pad_id)
        self.l_bias = nn.Embedding(vocab_size, 1, padding_idx=pad_id)
        self.r_emb = nn.Embedding(vocab_size, d_embed, padding_idx=pad_id)
        self.r_bias = nn.Embedding(vocab_size, 1, padding_idx=pad_id)
        self.loss_fn = GloveLoss()

    def forward(self, input_dict: dict):
        # Compute embeddings
        l_v = self.l_emb(input_dict['left'])
        l_b = self.l_bias(input_dict['left'])
        r_v = self.r_emb(input_dict['right'])
        r_b = self.r_bias(input_dict['right'])
        
        # Mean over ngram dimension if existing
        if len(l_v.shape) > 2:
            l_v = self.combine_ngram_embeddings(l_v, dim=-2)
            l_b = self.combine_ngram_embeddings(l_b, dim=-2)
            r_v = self.combine_ngram_embeddings(r_v, dim=-2)
            r_b = self.combine_ngram_embeddings(r_b, dim=-2)
                    
        return (l_v * r_v).sum(dim=-1) + l_b.squeeze() + r_b.squeeze()
    
    def combine_ngram_embeddings(self,
                                 x: torch.Tensor,
                                 dim: int,
                                 reduce: str='mean'):
        """ Combine stacked ngram embedding vectors coming from subword tokens
            or from a sequence of tokens, into a single embedding vector
        """
        if reduce == 'mean':  # take the mean only over non-padding tokens
            norm_factor = (x != 0).sum(dim=dim).clip(min=1) / x.shape[dim]
            return x.mean(dim=dim) / norm_factor
        else:
            return x.sum(dim=dim)
    
    def get_token_embeddings(self, token_indices: list):
        """ Compute static embeddings for a list of tokens as a stacked tensor
        """
        all_embeddings = self.l_emb.weight + self.r_emb.weight
        token_embeddings = []
        for token_index in token_indices:
            embedded = all_embeddings[token_index]
            if len(embedded.shape) > 1:  # ngram case
                embedded = self.combine_ngram_embeddings(embedded, dim=-2)
            token_embeddings.append(embedded)
        return torch.stack(token_embeddings, dim=0).detach().cpu()
    
    def get_sequence_embeddings(self, sequence: list, weights: list=None):
        """ Compute a single static embedding vector for a sequence of tokens
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
    def __init__(self, cooc_max=500, alpha: float=0.75, reduce: str='mean'):
        """ Hyperparameters modified from the original article (100, 0.75)
        """
        super().__init__()
        self.cooc_max = cooc_max
        self.alpha = alpha
        assert reduce in ('mean', 'max'), 'Invalid reduce mode.'
        self.reduce = torch.mean if reduce == 'mean' else torch.max
        
    def normalize(self, cooc: torch.Tensor):
        """ Normalization as in the original article.
        """
        return torch.where(condition=(cooc < self.cooc_max),
                           input=(cooc / self.cooc_max) ** self.alpha,
                           other=1.0)
    
    def forward(self, model_output: torch.Tensor, cooc: torch.Tensor):
        """ Expects flattened model output and target coocurence matrix
        """
        norm_cooc = self.normalize(cooc)
        log_cooc = torch.log(cooc)
        return self.reduce(norm_cooc * (model_output - log_cooc) ** 2)
        