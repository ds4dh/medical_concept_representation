import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self,
                 n_enc_layers, n_dec_layers, d_embed, d_ff, n_heads, dropout,
                 max_seq_len, special_tokens, vocab_size, share_embeddings=True,
                 *args, **kwargs):
        ''' Initialize a Transformer model for seq to seq translation,
            using as much as possible the torch.nn transformer modules.
        
        Params:
        -------
        max_len: int
            Maximum number of tokens in a sequence.
        vocab_sizes: int
            Number of tokens in the source/target vocabularies.
        pad_id: int
            Index of the padding token in the any vocabulary.
        n_enc_layers: int
            Number of layers in the encoder.
        n_dec_layers: int
            Number of layers in the decoder.
        d_embed: int
            Dimension of the embedding vectors.
        d_ff: int
            Dimension of the feedforward network.
        n_heads: int
            Number of attention heads.
        dropout: float
            Dropout rate.
        device: torch.device
            Device on which the model and data are run.
        share_embeddings: bool
            Whether to share the embedding matrix between source and target.
            
        '''
        super().__init__()
        # Source and target vocab
        pad_id = special_tokens['[PAD]']
        self.pad_id = pad_id
        if share_embeddings:
            assert isinstance(vocab_size, int)
            src_vocab_size, tgt_vocab_size = vocab_size, vocab_size
        else:
            # Note: not implemented in the pipeline yet
            assert isinstance(vocab_size, dict)
            src_vocab_size = vocab_size['src']
            tgt_vocab_size = vocab_size['tgt']
        
        # Token and position embeddings
        self.src_embedding = TokenEmbedding(src_vocab_size, d_embed, pad_id)
        if share_embeddings:
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_embed, pad_id)
        self.pos_embedding = PositionalEmbedding(d_embed, max_seq_len)
        
        # PyTorch version of transformer (note: it has no token/pos embedding)
        self.pt_transformer = nn.Transformer(d_model=d_embed,
                                             nhead=n_heads,
                                             num_encoder_layers=n_enc_layers,
                                             num_decoder_layers=n_dec_layers,
                                             dim_feedforward=d_ff,
                                             dropout=dropout,
                                             activation='gelu',
                                             batch_first=True,
                                             norm_first=True)

        # Causal mask function
        self.causal_mask_fn = nn.Transformer.generate_square_subsequent_mask

        # Final projection layer (for logits)
        self.ff_final_proj = nn.Sequential(nn.Linear(d_embed, tgt_vocab_size),
                                           nn.LogSoftmax(dim=-1))
        
        # Loss function used in the pipeline
        self.loss_fn = TransformerLoss(pad_id)
        

    def forward(self, src, tgt):
        ''' Forward pass of the Transformer model.

        Params:
        -------
        src: torch.Tensor of shape (batch_size, src_seq_len)
            Input tokens of the source language.
        tgt: torch.Tensor of shape (batch_size, tgt_seq_len)
            Input tokens of the target language.

        Returns:
        --------
        logits: torch.Tensor of shape (batch_size, seq_length, d_embed)
            Output tokens of the target language (logits for any word).
        
        '''
        # Define attention masks for encoding, memory and decoding
        masks = self.make_masks(src, tgt)
        src_pad_mask, mem_pad_mask, tgt_pad_mask, src_mask, tgt_mask = masks
        
        # Vocab and positional embedding of source and target tokens
        src_embedded = self.src_embedding(src)
        src_embedded += self.pos_embedding(src_embedded)
        tgt_embedded = self.tgt_embedding(tgt)
        tgt_embedded += self.pos_embedding(tgt_embedded)
        
        # Run the model with the embedded tokens
        output = self.pt_transformer(src=src_embedded,  # (bs, seq, embed)
                                     tgt=tgt_embedded,  # (bs, seq, embed)
                                     src_mask=src_mask,
                                     tgt_mask=tgt_mask,
                                     memory_mask=None,
                                     src_key_padding_mask=src_pad_mask,
                                     tgt_key_padding_mask=tgt_pad_mask,
                                     memory_key_padding_mask=None)

        # Generate logits from the output of the transformer
        logits = self.ff_final_proj(output)
        return logits

    def make_masks(self, src_tokens, tgt_tokens):
        ''' Create a mask for the encoder, decoder, and for their interaction.
        
        Params:
        -------
        src_tokens: torch.Tensor of shape (batch_size, src_seq_len)
            Input tokens of the source sentence.
        tgt_tokens: torch.Tensor of shape (batch_size, tgt_seq_len)
            Input tokens of the target sentence.

        Returns:
        --------
        enc_mask: torch.Tensor of shape (batch_size, src_seq_len, src_seq_len)
            Mask for src to src sequence queries (take care of padding).
        mem_mask: torch.Tensor of shape (batch_size, tgt_seq_len, src_seq_len)
            Mask for tgt to src sequence queries (take care of padding).
        dec_mask: torch.Tensor of shape (batch_size, tgt_seq_len, tgt_seq_len)
            Mask for tgt to tgt sequence queries (take care of padding).
        cau_mask: torch.Tensor of shape (batch_size, tgt_seq_len, tgt_seq_len)
            Mask for tgt to tgt sequence queries (ensure causality).

        '''
        # Get parameters
        device = src_tokens.device
        s_len = src_tokens.size(1)
        t_len = tgt_tokens.size(1)

        # Create padding masks
        src_pad_mask = (src_tokens == self.pad_id)
        mem_pad_mask = (src_tokens == self.pad_id)
        tgt_pad_mask = (tgt_tokens == self.pad_id)
        
        # Create "subsequent" masks
        src_mask = torch.zeros((s_len, s_len), device=device).type(torch.bool)
        tgt_mask = self.causal_mask_fn(t_len).to(device)

        # Return all masks for efficient and causal processing
        return src_pad_mask, mem_pad_mask, tgt_pad_mask, src_mask, tgt_mask

    def encode(self, src_tokens):
        ''' Compute the encoder representation of the source sentence.

        Params:
        -------
        src_tokens: torch.Tensor of shape (batch_size, src_seq_len)
            Input tokens of the source language.
        
        Returns:
        --------
        torch.Tensor of shape (batch_size, seq_length, d_embed)
            Encoder memory (encoded source sentence).

        '''
        # Define attention masks for decoding and memory
        src_pad_mask, _, _, _, _ = self.make_masks(src_tokens, src_tokens)

        # Vocab and positional embedding of source tokens
        src_embedded = self.src_embedding(src_tokens)
        src_embedded = self.pos_embedding(src_embedded)
        
        # Encode source sentence, using the pytorch-transformer module
        encoded = self.pt_transformer.encoder(
                                        src_embedded,
                                        mask=None,
                                        src_key_padding_mask=src_pad_mask)
        return encoded

    def decode_probs(self, src, gen, mem):
        ''' Compute probabilities for the next token, given encoded memory
            and tokens already generated

        Params:
        -------
        src_tokens: torch.Tensor of shape (batch_size, src_seq_len)
            Input tokens of the source language (used for the padding mask).
        gen_tokens: torch.Tensor of shape (batch_size, tgt_seq_len)
            Tokens generated by the model so far (in the target vocabulary).
        memory: torch.Tensor of shape (batch_size, src_seq_len, d_embed)
            Encoder memory.
        
        Returns:
        --------
        probs: torch.Tensor of shape (batch_size, seq_length, d_embed)
            Probabilities for the next token (in the target vocabulary).

        '''
        # Define attention masks for decoding and memory
        masks = self.make_masks(src, gen)
        _, mem_pad_mask, tgt_pad_mask, _, tgt_mask = masks
        
        # Vocab and positional embedding of target tokens
        gen_embedded = self.tgt_embedding(gen)
        gen_embedded = self.pos_embedding(gen_embedded)

        # Get the decoded output, using the pytorch-transformer module
        output = self.pt_transformer.decoder(
                                        tgt=gen_embedded,
                                        memory=mem,
                                        tgt_mask=tgt_mask,
                                        memory_mask=None,
                                        tgt_key_padding_mask=tgt_pad_mask,
                                        memory_key_padding_mask=mem_pad_mask)

        # Return the logits
        logits = self.ff_final_proj(output)
        probs = F.softmax(logits, dim=-1)
        return probs


class TransformerLoss(nn.Module):
    def __init__(self, pad_id):
        super().__init__()
        self.loss_fn = nn.NLLLoss(ignore_index=pad_id)
        
    def forward(self, model_output, tgt):
        pred = model_output[:, :-1].transpose(-2, -1)  # not sure here...
        gold = tgt[:, 1:]  # remove '[BOS]' token
        return self.loss_fn(pred, gold)


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_embed=512, pad_id=0):
        super().__init__(vocab_size, d_embed, padding_idx=pad_id)


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
