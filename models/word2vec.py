import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):
    # Adapted from https://github.com/Andras7/word2vec-pytorch
    def __init__(self,
                 vocab_sizes: dict,
                 special_tokens: dict,
                 d_embed: int,
                 n_neg_samples: int,
                 *args, **kwargs):
        super(SkipGram, self).__init__()
        vocab_size = vocab_sizes['total']
        pad_id = special_tokens['[PAD]']

        self.center_embeddings = nn.Embedding(
            vocab_size, d_embed, padding_idx=pad_id)  # sparse=True
        self.context_embeddings = nn.Embedding(
            vocab_size, d_embed, padding_idx=pad_id)  # sparse=True
        
        self.n_neg_samples = n_neg_samples
        if self.n_neg_samples == 0:
            self.center_fc = nn.Linear(d_embed, vocab_size)
            self.loss_fn = SoftMaxLoss()
        else:
            self.loss_fn = NegSamplingLoss()

        bounds = 1.0 / d_embed
        nn.init.uniform_(self.center_embeddings.weight.data, -bounds, bounds)
        nn.init.constant_(self.context_embeddings.weight.data, 0)

    def forward(self, input_dict: dict):
        # Assign input variables
        pos_center = input_dict['pos_center']
        pos_context = input_dict['pos_context']

        # Softmax case
        if self.n_neg_samples == 0:
            pos_center = self.center_embeddings(pos_center)
            if len(pos_center.shape) > 2:  # ngram case
                pos_center = self.combine_ngram_embeddings(pos_center, dim=-2)
            return {'pos_center': self.center_fc(pos_center),
                    'pos_context': pos_context}
        
        # Negative sampling case
        else:
            neg_context = input_dict['neg_context']
            pos_center = self.center_embeddings(pos_center)
            pos_context = self.context_embeddings(pos_context)
            if len(pos_center.shape) > 2:  # ngram case
                pos_center = self.combine_ngram_embeddings(pos_center, dim=-2)
                pos_context = self.combine_ngram_embeddings(pos_context, dim=-2)
            neg_context = self.context_embeddings(neg_context)
            return {'pos_center': pos_center,
                    'pos_context': pos_context,
                    'neg_context': neg_context}
                    
    def combine_ngram_embeddings(self,
                                 x: torch.Tensor,
                                 dim: int,
                                 reduce: str='mean'):
        """ Combine stacked ngram embedding vectors coming from subword tokens
            or from a sequence of tokens, into a single embedding vector
        """
        if reduce == 'mean':
            norm_factor = (x != 0).sum(dim=dim).clip(min=1) / x.shape[dim]
            return x.mean(dim=dim) / norm_factor
        else:
            return x.sum(dim=dim)
    
    def get_token_embeddings(self, token_indices: list):
        """ Compute static embeddings for a list of tokens as a stacked tensor
        """
        all_embeddings = self.center_embeddings.weight
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
            Each token in the sentence is weighted by inverse term frequency
        """
        if weights == None:  # classic average
            return embeddings.mean(dim=dim)
        else:  # weighted average
            weights = torch.tensor(weights, dtype=embeddings.dtype)
            return embeddings.T @ weights / weights.sum()
            

class NegSamplingLoss(nn.Module):
    # Adapted from https://github.com/Andras7/word2vec-pytorch
    def __init__(self):
        super().__init__()

    def forward(self, model_output: torch.Tensor, *args, **kwargs):
        pos_center = model_output['pos_center']
        pos_context = model_output['pos_context']
        neg_context = model_output['neg_context']

        score = torch.sum(torch.mul(pos_center, pos_context), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)
        
        neg_score = torch.bmm(neg_context, pos_center.unsqueeze(2)).squeeze(2)
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)


class SoftMaxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.nlll_loss = nn.NLLLoss()

    def forward(self, model_output: torch.Tensor):
        pos_center = model_output['pos_center']
        pos_context = model_output['pos_context']
        logits = self.log_softmax(pos_center)
        return self.nlll_loss(logits, pos_context)


class Word2Vec(SkipGram):
    def __init__(self,
                 vocab_sizes: dict,
                 special_tokens: dict,
                 d_embed: int,
                 n_neg_samples: int,
                 *args, **kwargs):
        assert 'ngram' not in vocab_sizes.keys(),\
            'Word2Vec model cannot have subword tokenization. '+\
            'Choose FastText model or word tokenization.'
        super().__init__(vocab_sizes=vocab_sizes,
                         special_tokens=special_tokens,
                         d_embed=d_embed,
                         n_neg_samples=n_neg_samples,
                         *args, **kwargs)


class FastText(SkipGram):
    def __init__(self,
                 vocab_sizes: dict,
                 special_tokens: dict,
                 d_embed: int,
                 n_neg_samples: int,
                 *args, **kwargs):
        assert 'ngram' in vocab_sizes.keys(),\
            'FastText model must have subword tokenization. '+\
            'Choose Word2Vec model or any of [subword, icd] tokenization.'
        super(FastText, self).__init__(vocab_sizes=vocab_sizes,
                                       special_tokens=special_tokens,
                                       d_embed=d_embed,
                                       n_neg_samples=n_neg_samples,
                                       *args, **kwargs)
        