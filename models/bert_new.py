# Taken from https://github.com/codertimo/BERT-pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BERT(nn.Module):
    """ BERT: Bidirectional Encoder Representations from Transformers. """
    def __init__(self, vocab_sizes, special_tokens, max_seq_len, d_embed,
                 d_ff, n_layers, n_heads, dropout=0.1, *args, **kwargs):
        """
        :param voc_size: voc_size of total words
        :param pad_id: index of the padding token
        :param d_embed: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param n_heads: number of attention heads
        :param dropout: dropout rate
        """
        super().__init__()
        self.mask_id = special_tokens['[MASK]']
        self.loss_fn = BertLoss(mask_id=self.mask_id, max_seq_len=max_seq_len)

        # Sum of positional, segment and token embeddings
        vocab_size = vocab_sizes['total']
        self.embedding = BERTEmbedding(vocab_size=vocab_size,
                                       max_len=max_seq_len,
                                       d_embed=d_embed,
                                       pad_id=special_tokens['[PAD]'])

        # Multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_embed, n_heads, d_ff, dropout) \
                for _ in range(n_layers)])

        # Final projection to predict words for each masked token
        self.final_proj = nn.Linear(d_embed, vocab_size)

    def forward(self, masked, segment_labels=None, get_embeddings_only=False):
        # Adapt the size of what is used to build the masks
        input_for_masks = masked
        if len(input_for_masks.shape) > 2:  # ngram case
            input_for_masks = input_for_masks[:, :, 0]
        
        # Attention mask for padded token (batch_size, 1, seq_len, seq_len)
        pad_mask = (input_for_masks != self.mask_id).unsqueeze(1) \
                    .repeat(1, masked.size(1), 1).unsqueeze(1)

        # Embed token_id sequences to vector sequences
        if segment_labels == None:
            segment_labels = torch.ones_like(input_for_masks, dtype=torch.int)
        x = self.embedding(masked, segment_labels)

        # Run through all transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, pad_mask)
        
        # Returns embeddings or word projections
        if get_embeddings_only:
            return x
        else:
            return self.final_proj(x)


class BERT(nn.Module):
    def __init__(self, special_tokens, vocab_sizes, max_seq_len, n_layers,
                 d_embed, d_ff, attn_type, dropout=0.1, n_heads=None):
        super().__init__()
        self.attn_type = attn_type
        self.mask_id = special_tokens['[MASK]']
        self.loss_fn = BertLoss(mask_id=self.mask_id, max_seq_len=max_seq_len)

        # Sum of positional, segment and token embeddings
        vocab_size = vocab_sizes['total']
        self.embedding = BERTEmbedding(vocab_size=vocab_size,
                                       max_len=max_seq_len,
                                       d_embed=d_embed,
                                       pad_id=special_tokens['[PAD]'])
        
        # BERT layers
        self.layers = nn.ModuleList([])
        for l in range(n_layers):
            if attn_type == 'fourier' and l < n_layers - 2:
                attn_module = FourierNet()
            else:
                attn_module = MultiHeadedAttention(n_heads, d_embed, dropout)
            self.layers.append(nn.ModuleList([
                PreNorm(d_embed, attn_module),
                PreNorm(d_embed, FeedForward(d_embed, d_ff, dropout))]))
        
        # Final projection to predict words for each masked token
        self.final_proj = nn.Linear(d_embed, vocab_size)

    def forward(self, masked):
        # Adapt the size of what is used to build the masks
        input_for_masks = masked
        if len(input_for_masks.shape) > 2:  # ngram case
            input_for_masks = input_for_masks[:, :, 0]
        
        # Attention mask for padded token (batch_size, 1, seq_len, seq_len)
        pad_mask = (input_for_masks != self.mask_id).unsqueeze(1) \
                    .repeat(1, masked.size(1), 1).unsqueeze(1)

        # Embed token_id sequences to vector sequences
        if segment_labels == None:
            segment_labels = torch.ones_like(input_for_masks, dtype=torch.int)
        x = self.embedding(masked, segment_labels)

        for attn, ff in self.layers:
            x = attn(x, pad_mask) + x
            x = ff(x) + x
        return x


class BertLoss(nn.Module):
    def __init__(self, mask_id, max_seq_len):
        super().__init__()
        self.mask_id = mask_id
        self.max_seq_len = max_seq_len
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.nlll_loss = nn.NLLLoss()

    def forward(self, model_output, masked_label_ids, masked_label):
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


class FeedForward():
    def __init__(self, d_embed, d_ff, dropout=0.1):
        self = nn.Sequential(nn.Linear(d_embed, d_ff),
                             nn.GELU(),
                             nn.Dropout(dropout),
                             nn.Linear(d_ff, d_embed),
                             nn.Dropout(dropout))


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FourierNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, d_embed, dropout=0.1):
        super().__init__()
        assert d_embed % n_heads == 0

        # We assume d_v always equals d_k
        self.d_k = d_embed // n_heads
        self.n_heads = n_heads
        self.q_linear = nn.Linear(d_embed, d_embed)
        self.k_linear = nn.Linear(d_embed, d_embed)
        self.v_linear = nn.Linear(d_embed, d_embed)
        self.output_linear = nn.Linear(d_embed, d_embed)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None, return_attention=False):
        b_size = q.size(0)
        q = self.q_linear(q).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.q_linear(k).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.q_linear(v).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        x, attn = self.attention(q, k, v, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_k)
        if return_attention:
            return self.output_linear(x), attn
        else:
            return self.output_linear(x)


class Attention(nn.Module):
    def forward(self, q, k, v, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, v), p_attn


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, pad_id, max_len, d_embed, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_embed, padding_idx=pad_id)
        self.pos = PositionalEmbedding(d_embed, max_len=max_len)
        self.seg = nn.Embedding(3, d_embed)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence, segment_labels):
        # For token_embeddings, sum over ngram dimension if existing
        tok_embeddings = self.tok(sequence)
        if len(tok_embeddings.shape) > 3:
            # (batch, seq, ngram, d_embed) -> (batch, seq_len, d_embed)
            tok_embeddings = self.combine_ngram_embeddings(tok_embeddings, dim=-2)
        
        # Add position and segment embeddings
        x = tok_embeddings + self.pos(sequence) + self.seg(segment_labels)
        return self.dropout(x)

    def combine_ngram_embeddings(self, x, dim, reduce='mean'):
        if reduce == 'mean':
            norm_factor = (x != 0).sum(dim=dim).clip(min=1) / x.shape[dim]
            return x.mean(dim=dim) / norm_factor
        else:
            return x.sum(dim=dim)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_embed, max_len):
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

    def forward(self, x):
        return self.pe[:, :x.size(1)]
