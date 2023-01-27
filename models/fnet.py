# Main structure taken from https://github.com/rishikksh20/FNet-pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from itertools import zip_longest


class FNet(nn.Module):
    def __init__(self,
                 vocab_sizes: dict,
                 special_tokens: dict,
                 max_seq_len: int,
                 d_embed: int,
                 d_ff: int,
                 n_layers: int,
                 n_heads: int=None,
                 dropout: float=0.1,
                 *args, **kwargs):
        super().__init__()
        self.pad_id = special_tokens['[PAD]']
        self.bos_id = special_tokens['[CLS]']
        self.eos_id = special_tokens['[END]']
        self.mask_id = special_tokens['[MASK]']
        self.loss_fn = BertLoss(max_seq_len=max_seq_len)

        # Embedding layer (sum of positional, segment and token embeddings)
        self.max_seq_len = max_seq_len
        self.embedding = BertEmbedding(vocab_size=vocab_sizes['total'],
                                       max_len=max_seq_len,
                                       d_embed=d_embed,
                                       pad_id=self.pad_id)
        
        # BERT layers
        self.layers = nn.ModuleList([])
        for l in range(n_layers):
            if l < n_layers - 2:
                attn_module = FourierLayer()
            else:
                attn_module = MultiHeadedAttention(n_heads, d_embed, dropout)
            self.layers.append(nn.ModuleList([
                Norm(d_embed, attn_module),
                Norm(d_embed, FeedForward(d_embed, d_ff, dropout))]))
        
        # Final projection to predict words for each masked token
        self.final_proj = nn.Linear(d_embed, vocab_sizes['total'])

    def forward(self,
                masked: torch.Tensor,
                segment_labels: torch.Tensor=None,
                get_embeddings: bool=False):
        # Embed token sequences to vector sequences
        pad_mask = self.create_pad_mask(masked)
        x = self.embedding(masked, segment_labels)

        # Run through all BERT layers (with intermediate residual connection)
        for attn, ff in self.layers:
            x = attn(x, mask=pad_mask) + x
            x = ff(x) + x
        
        # Return embeddings or word projections
        return x if get_embeddings else self.final_proj(x)

    def create_pad_mask(self, masked: torch.Tensor):
        # Adapt the size of the array used to build the masks
        input_for_masks = masked
        if len(input_for_masks.shape) > 2:  # ngram case
            input_for_masks = input_for_masks[:, :, 0]
        
        # Attention mask for padded token (batch_size, 1, seq_len, seq_len)
        pad_mask = (input_for_masks != self.mask_id).unsqueeze(1)
        return pad_mask.repeat(1, masked.size(1), 1).unsqueeze(1)
    
    def get_token_embeddings(self, token_indices: list):
        """ Compute static embeddings for a list of tokens
        """
        embedder = self.embedding
        all_embeddings = embedder.tok.weight
        token_embeddings = []
        for token_index in token_indices:
            embedded = all_embeddings[token_index]
            if len(embedded.shape) > 1:  # ngram case
                embedded = embedder.combine_ngram_embeddings(embedded, dim=-2)
            token_embeddings.append(embedded)
        return torch.stack(token_embeddings, dim=0).detach().cpu()
    
    def get_sequence_embeddings(self,
                                sequence: list,
                                weights: list=None,
                                mode: str='context_avg'):
        """ Compute embedding (static or contextualized) for a token sequence
        """
        if mode == 'static':
            embedded = self.get_token_embeddings(sequence)
            embedded = self.collapse_sequence_embeddings(embedded, weights)

        elif 'context' in mode:
            sequence = self.pre_process_for_sequence(sequence)
            embedded = self.forward(sequence, get_embeddings=True)
            if mode == 'context_cls':
                embedded = embedded[:, 0]  # '[CLS]' token embedding
            elif mode == 'context_avg':
                embedded = self.collapse_sequence_embeddings(embedded, weights)
            else:
                raise ValueError('Bad context mode for elmo')

        return embedded.squeeze().detach().cpu()

    def pre_process_for_sequence(self, sequence: list):
        """ Add [EOS]/[BOS] tokens, trim too lengthy sequences, tensorize
        """
        if isinstance(sequence[0], list):  # ngram case
            sequence.insert(0, [self.bos_id]); sequence.append([self.eos_id])
            sequence = list(zip(*zip_longest(*sequence, fillvalue=self.pad_id)))
        else:
            sequence.insert(0, self.bos_id); sequence.append(self.eos_id)
        if len(sequence) > self.max_seq_len:  # after adding [EOS], [BOS] tokens
            sequence = sequence[:self.max_seq_len]
        return torch.tensor(sequence)[None, ...]  # add batch dimension
    
    def collapse_sequence_embeddings(self,
                                     embeddings: torch.Tensor,
                                     weights: list,
                                     dim: int=-2):
        """ Average sequence embedding over sequence dimension
        """
        if len(embeddings.shape) > 2:  # context_avg case
            if len(weights) > self.max_seq_len - 2:
                embeddings = embeddings[0, 1:]
                weights = weights[:self.max_seq_len - 1]
            else:
                embeddings = embeddings[0, 1:-1]
        if weights == None:  # classic average
            return embeddings.mean(dim=dim)
        else:  # weighted average
            weights = torch.tensor(weights, dtype=embeddings.dtype)
            return embeddings.T @ weights / weights.sum()
    

class BertLoss(nn.Module):
    def __init__(self, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.nlll_loss = nn.NLLLoss()

    def forward(self,
                model_output: torch.Tensor,
                masked_label_ids: list,
                masked_label: list):
        # Retrieve indexes of masked tokens and their values
        seq_len = model_output.shape[1]
        msk_ids, msk_lbls = [], []
        for i, (ids, lbls) in enumerate(zip(masked_label_ids, masked_label)):
            for id, lbl in zip(ids, lbls):
                if id < self.max_seq_len:
                    msk_ids.append(id + i * seq_len); msk_lbls.append(lbl)
        
        # Select the corresponding model predictions and compute loss
        model_output = model_output.view(-1, model_output.shape[-1])
        prediction = self.log_softmax(model_output[msk_ids])
        target = torch.tensor(msk_lbls, device=prediction.device)
        return self.nlll_loss(prediction, target)


class FeedForward(nn.Module):
    def __init__(self, d_embed: int, d_ff: int, dropout: float=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_embed, d_ff),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(d_ff, d_embed),
                                 nn.Dropout(dropout))

    def forward(self, x: torch.Tensor):
        return self.net(x)


class Norm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module, mode: str='pre'):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self.mode = mode

    def forward(self, x: torch.Tensor, **kwargs):
        if self.mode == 'pre':
            return self.fn(self.norm(x), **kwargs)
        elif self.mode == 'post':
            return self.norm(self.fn(x, **kwargs))  # not tested yet
        else:
            raise ValueError('Invalid mode for normalization (pre, post)')


class FourierLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, dropout: float=0.1):
        super().__init__()
        assert d_embed % n_heads == 0
        self.d_embed = d_embed
        self.d_head = d_embed // n_heads
        self.n_heads = n_heads
        self.q_linear = nn.Linear(d_embed, d_embed)
        self.k_linear = nn.Linear(d_embed, d_embed)
        self.v_linear = nn.Linear(d_embed, d_embed)
        self.output_linear = nn.Linear(d_embed, d_embed)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def _split_by_head(self, x: torch.Tensor, batch_size: int):
        return x.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
    
    def _combine_heads(self, x: torch.Tensor, batch_size: int):
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_embed)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor=None,
                return_attention: bool=False):
        batch_size = x.size(0)
        q = self._split_by_head(self.q_linear(x), batch_size)
        k = self._split_by_head(self.k_linear(x), batch_size)
        v = self._split_by_head(self.v_linear(x), batch_size)
        x, attn = self.attention(q, k, v, mask=mask, dropout=self.dropout)
        x = self._combine_heads(x, batch_size)
        if return_attention:
            return self.output_linear(x), attn
        else:
            return self.output_linear(x)


class Attention(nn.Module):
    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: torch.Tensor=None,
                dropout: float=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, v), p_attn


class BertEmbedding(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 pad_id: int,
                 max_len: int,
                 d_embed: int,
                 dropout:float=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_embed, padding_idx=pad_id)
        self.pos = PositionalEmbedding(d_embed, max_len=max_len)
        self.seg = nn.Embedding(3, d_embed)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,
                sequence: torch.Tensor,
                segment_labels: torch.Tensor):
        # Create an idle segment mask if none is used
        if segment_labels == None:
            sequence_for_lbls = sequence
            if len(sequence_for_lbls.shape) > 2:  # ngram case
                sequence_for_lbls = sequence_for_lbls[:, :, 0]
            segment_labels = torch.ones_like(sequence_for_lbls, dtype=torch.int)

        # For token_embeddings, sum over ngram dimension if existing
        tok_emb = self.tok(sequence)
        if len(tok_emb.shape) > 3:
            # (batch, seq, ngram, d_embed) -> (batch, seq_len, d_embed)
            tok_emb = self.combine_ngram_embeddings(tok_emb, dim=-2)
        
        # Add position and segment embeddings
        x = tok_emb + self.pos(sequence) + self.seg(segment_labels)
        return self.dropout(x)

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


class PositionalEmbedding(nn.Module):
    def __init__(self, d_embed: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_embed).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_embed, 2).float() \
                            * -(math.log(10000.0) / d_embed)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        return self.pe[:, :x.size(1)]
