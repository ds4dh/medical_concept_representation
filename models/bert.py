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
        assert special_tokens['[PAD]'] == 0, 'For this model, pad_id must be 0'
        self.pad_id = special_tokens['[PAD]']
        self.mask_id = special_tokens['[MASK]']
        self.d_embed = d_embed
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.loss_fn = BertLoss(mask_id=self.mask_id)

        # Sum of positional, segment and token embeddings
        vocab_size = vocab_sizes['total']
        self.embedding = BERTEmbedding(vocab_size=vocab_size,
                                       max_len=max_seq_len,
                                       d_embed=d_embed,
                                       pad_id=self.pad_id)

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
        

class BertLoss(nn.Module):
    def __init__(self, mask_id):
        super().__init__()
        self.mask_id = mask_id
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.nlll_loss = nn.NLLLoss()

    def forward(self, model_output, masked_label_ids, masked_label):
        logits = self.log_softmax(model_output)
        return self.nlll_loss(logits[masked_label_ids], masked_label)


class TransformerBlock(nn.Module):
    """ Bidirectional Encoder = Transformer (self-attention)
        Transformer = MultiHead_Attn + Feed_Forward with sublayer connection
    :param d_embed: hidden size of transformer
    :param n_heads: number of heads in multi-head attention
    :param d_ff: hidden dimension in the position feedforward network
    :param dropout: dropout rate
    """
    def __init__(self, d_embed, n_heads, d_ff, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(n_heads=n_heads, d_embed=d_embed)
        self.feed_forward = PositionwiseFeedForward(d_embed=d_embed,
                                                    d_ff=d_ff,
                                                    dropout=dropout)
        self.input_sublayer = SublayerConnection(size=d_embed, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=d_embed, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: 
                                self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, d_embed, dropout=0.1):
        super().__init__()
        assert d_embed % n_heads == 0

        # We assume d_v always equals d_k
        self.d_k = d_embed // n_heads
        self.n_heads = n_heads
        self.linear_layers = nn.ModuleList([nn.Linear(d_embed, d_embed) \
                                            for _ in range(3)])
        self.output_linear = nn.Linear(d_embed, d_embed)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        # 1) Do all the linear projections in batch from d_embed => h x d_k
        batch_size = q.size(0)
        q, k, v = [
            l(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
                for l, x in zip(self.linear_layers, (q, k, v))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(q, k, v, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(
                                    batch_size, -1, self.n_heads * self.d_k)

        return self.output_linear(x)


class Attention(nn.Module):
    """ Compute 'Scaled Dot Product Attention """
    def forward(self, q, k, v, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, v), p_attn


class PositionwiseFeedForward(nn.Module):
    """ Implements FFN equation. """
    def __init__(self, d_embed, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_embed, d_ff)
        self.w_2 = nn.Linear(d_ff, d_embed)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class GELU(nn.Module):
    """ Paper notice that BERT used the GELU (not RELU) """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * \
                         (x + 0.044715 * torch.pow(x, 3))))


class SublayerConnection(nn.Module):
    """ A residual connection followed by a layer norm.
        Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    """ Construct a layernorm module (See citation for details). """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.parameter.Parameter(torch.ones(features))
        self.b_2 = nn.parameter.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    

class BERTEmbedding(nn.Module):
    """ BERT Embedding which is consisted with under features
            1. TokenEmbedding : normal embedding matrix
            2. PositionalEmbedding : adding positional information using sin, cos
            2. SegmentEmbedding : adding sentence segment info, (A:1, B:2)
                sum of all these features are output of BERTEmbedding
    :param voc_size: total vocab size
    :param d_embed: embedding size of token embedding
    :param dropout: dropout rate
    """
    def __init__(self, vocab_size, pad_id, max_len, d_embed, dropout=0.1):
        super().__init__()
        self.tok = TokenEmbedding(vocab_size=vocab_size,
                                  d_embed=d_embed,
                                  pad_id=pad_id)
        self.pos = PositionalEmbedding(d_embed=self.tok.embedding_dim,
                                       max_len=max_len)
        self.seg = SegmentEmbedding(d_embed=self.tok.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.d_embed = d_embed

    def forward(self, sequence, segment_labels):
        # For token_embeddings, sum over ngram dimension if existing
        token_embeddings = self.tok(sequence)
        if len(token_embeddings.shape) > 3:
            # (batch, seq, ngram, d_embed) -> (batch, seq_len, d_embed)
            token_embeddings = self.combine_ngram_embeddings(token_embeddings,
                                                             dim=-2)
        
        # Add the other embeddings
        x = token_embeddings + self.pos(sequence) + self.seg(segment_labels)
        return self.dropout(x)

    def combine_ngram_embeddings(self, x, dim, reduce='mean'):
        if reduce == 'mean':
            norm_factor = (x != 0).sum(dim=dim).clip(min=1) / x.shape[dim]
            return x.mean(dim=dim) / norm_factor
        else:
            return x.sum(dim=dim)
        
        
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_embed=512, pad_id=0):
        super().__init__(vocab_size, d_embed, padding_idx=pad_id)


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


class SegmentEmbedding(nn.Embedding):
    def __init__(self, d_embed=512):
        super().__init__(3, d_embed, padding_idx=0)
        