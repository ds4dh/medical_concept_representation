import torch
import torch.nn as nn
import torch.nn.functional as F


# Adapted from https://github.com/Andras7/word2vec-pytorch
class FastText(nn.Module):

    def __init__(self, vocab_sizes, d_embed, special_tokens, *args, **kwargs):
        super(FastText, self).__init__()
        vocab_size = vocab_sizes['total']
        pad_id = special_tokens['[PAD]']
        self.loss_fn = FastTextLoss()

        self.center_embeddings = \
            nn.Embedding(vocab_size, d_embed, padding_idx=pad_id)  # sparse=True
        self.context_embeddings = \
            nn.Embedding(vocab_size, d_embed, padding_idx=pad_id)  # sparse=True

        bounds = 1.0 / d_embed
        nn.init.uniform_(self.center_embeddings.weight.data, -bounds, bounds)
        nn.init.constant_(self.context_embeddings.weight.data, 0)

    def forward(self, pos_center, pos_context, neg_context):
        pos_center = self.center_embeddings(pos_center)
        pos_context = self.context_embeddings(pos_context)
        neg_context = self.context_embeddings(neg_context)

        if len(pos_center.shape) > 2:  # ngram case (mean over word+subword dim)
            pos_center = self.combine_ngram_embeddings(pos_center, dim=-2)

        return {'pos_center': pos_center,
                'pos_context': pos_context,
                'neg_context': neg_context}

    def combine_ngram_embeddings(self, x, dim, reduce='mean'):
        if reduce == 'mean':
            norm_factor = (x != 0).sum(dim=dim).clip(min=1) / x.shape[dim]
            return x.mean(dim=dim) / norm_factor
        else:
            return x.sum(dim=dim)
    
    def get_token_embeddings(self, token_indices):
        """ Compute static embeddings each token in a list
        """
        all_embeddings = self.center_embeddings.weight
        token_embeddings = []
        for token_index in token_indices:
            embedded = all_embeddings[token_index]
            if len(embedded.shape) > 1:  # ngram case
                embedded = self.combine_ngram_embeddings(embedded, dim=-2)
            token_embeddings.append(embedded)
        return torch.stack(token_embeddings, dim=0).detach().cpu()
    
    def get_sequence_embeddings(self, sequence, weights=None):
        """ Compute static embedding of a token sequence
        """
        embeddings = self.get_token_embeddings(sequence)
        return self.collapse_sequence_embeddings(embeddings, weights)
    
    def collapse_sequence_embeddings(self, embeddings, weights, dim=-2):
        """ Average sequence embedding over sequence dimension
            Each token in the sentence is weighted by inverse term frequency
        """
        if weights == None:  # classic average
            return embeddings.mean(dim=dim)
        else:  # weighted average
            weights = torch.tensor(weights, dtype=embeddings.dtype)
            return embeddings.T @ weights / weights.sum()
            

class FastTextLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, model_output, *args, **kwargs):
        pos_center = model_output['pos_center']
        pos_context = model_output['pos_context']
        neg_context = model_output['neg_context']
        
        score = torch.sum(torch.mul(pos_center, pos_context), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(neg_context, pos_center.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)
