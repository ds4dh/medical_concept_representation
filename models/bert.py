# Taken from https://github.com/codertimo/BERT-pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BERT(nn.Module):
    """ BERT: Bidirectional Encoder Representations from Transformers. """
    def __init__(self, vocab_size, special_tokens, max_seq_len, d_embed, d_ff,
                 n_layers, n_heads, dropout=0.1, *args, **kwargs):
        """
        :param voc_size: voc_size of total words
        :param pad_id: index of the padding token
        :param d_embed: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param n_heads: number of attention heads
        :param dropout: dropout rate
        """
        super().__init__()
        self.pad_id = special_tokens['[PAD]']
        self.d_embed = d_embed
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.loss_fn = nn.NLLLoss()  # ??? check for mlm and log-softmax etc

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size,
                                       max_len=max_seq_len,
                                       d_embed=d_embed)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_embed, n_heads, d_ff, dropout) \
                for _ in range(n_layers)])

    def forward(self, x, segment_labels=None):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len])
        mask = (x != self.pad_id).unsqueeze(1) \
                                 .repeat(1, x.size(1), 1) \
                                 .unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        if segment_labels == None:
            segment_labels = torch.ones_like(x, dtype=torch.int)
        x = self.embedding(x, segment_labels)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x


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
    """ Take in number of heads and model hidden size. """
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
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
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
    def __init__(self, vocab_size, max_len, d_embed, dropout=0.1):
        super().__init__()
        self.tok = TokenEmbedding(vocab_size=vocab_size, d_embed=d_embed)
        self.pos = PositionalEmbedding(d_embed=self.tok.embedding_dim,
                                       max_len=max_len)
        self.seg = SegmentEmbedding(d_embed=self.tok.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.d_embed = d_embed

    def forward(self, sequence, segment_labels):
        x = self.tok(sequence) + self.pos(sequence) + self.seg(segment_labels)
        return self.dropout(x)


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_embed=512):
        super().__init__(vocab_size, d_embed, padding_idx=0)


class PositionalEmbedding(nn.Module):

    def __init__(self, d_embed, max_len):
        super().__init__()
        # Compute the positional encodings once in log space.
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
        