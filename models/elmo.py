import torch
import torch.nn as nn
from torch import relu, sigmoid


class ELMO(nn.Module):
    def __init__(self, n_lstm_layers, d_lstm, d_embed_char, d_embed_word,
                 d_conv, k_size, vocab_sizes, special_tokens, add_words,
                 dropout=0.5, *args, **kwargs):
        super().__init__()
        # Retrieve vocabulary information
        self.pad_id = special_tokens['[PAD]']
        self.special_token_ids = list(special_tokens.values())
        self.n_words = vocab_sizes['word']
        
        # Check types of token embeddings used in the model
        self.token_type = 'word' if 'ngram' not in vocab_sizes else 'char'
        if add_words:
            assert self.token_type == 'char', 'The model requires ngram ' + \
                'lengths > 0 to add whole word embeddings to char embeddings.'
            self.token_type = 'both'
        
        # Word embeddings
        word_vocab_size = vocab_sizes['word'] + vocab_sizes['special']
        if self.token_type in ['word', 'both']:
            self.word_embedding = nn.Embedding(word_vocab_size,
                                               d_embed_word,
                                               padding_idx=self.pad_id)
        
        # Layers from character to word embeddings
        if self.token_type in ['char', 'both']:
            char_vocab_size = vocab_sizes['ngram'] + vocab_sizes['special']
            self.char_embedding = nn.Embedding(char_vocab_size,
                                               d_embed_char,
                                               padding_idx=self.pad_id)
            self.char_cnn = CharCNN(d_conv, k_size, d_embed_char, d_embed_word)
        
        # Network layers
        self.word_lstm = WordLSTM(d_embed_word, d_lstm, n_lstm_layers, dropout)
        self.final_proj = nn.Linear(2 * d_embed_word, word_vocab_size)

        # Loss function used by the model
        self.loss_fn = ELMOLoss()
        
    def forward(self, chars, words, return_all_states=False):
        """ Forward pass of the ELMO model
            - Inputs:
                chars (batch_size, seq_len, ngram_len)
                words (batch_size, seq_len)
            - Output (batch_size, seq_len, d_emb_word)
        """
        # char_emb = self.char_embedding(chars)
        # static_emb = self.char_cnn(char_emb)
        static_emb = self.compute_static_embeddings(chars, words)
        context_emb = self.word_lstm(static_emb, return_all_states)
        if return_all_states:
            return context_emb  # for downstream task
        else:
            return self.final_proj(context_emb)  # for pre-training

    def compute_static_embeddings(self, chars, words):       
        """ Create word static embeddings from characters and/or words
            - Inputs:
                chars (batch_size, seq_len, ngram_len)
                words (batch_size, seq_len)
            - Output (batch_size, seq_len, d_emb_word)
        """ 
        if self.token_type == 'char':
            char_emb = self.char_embedding(chars)
            return self.char_cnn(char_emb)
        
        elif self.token_type == 'word':
            return self.word_embedding(words)
        
        elif self.token_type == 'both':
            char_emb = self.char_embedding(chars)
            char_emb = self.char_cnn(char_emb)
            word_emb = self.word_embedding(words)
            return self.combine_word_char_embeddings(char_emb, word_emb)
    
    def combine_word_char_embeddings(self, char_emb, word_emb, reduce='mean'):
        if reduce == 'mean':
            return (char_emb + word_emb) / 2
        else:
            return char_emb + word_emb

    def get_embeddings(self):
        """ Compute embeddings for ELMO model
        """
        ...
        

class ELMOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.nlll_loss = nn.NLLLoss()

    def forward(self, model_output, words):
        logits = self.log_softmax(model_output).transpose(2, 1)
        return self.nlll_loss(logits, words)
    

class CharCNN(nn.Module):
    def __init__(self, d_convs, k_sizes, d_emb_char, d_emb_word, activ_fn=relu):
        super().__init__()
        self.cnn_layers = nn.ModuleList([
            nn.Conv1d(in_channels=d_emb_char,
                      out_channels=d_conv,
                      kernel_size=k_size,  # kernel_size is like n-gram
                      bias=True)
            for (d_conv, k_size) in zip(d_convs, k_sizes)])
        self.activ_fn = activ_fn
        self.highway = Highway(sum(d_convs), n_layers=2)
        self.word_emb_proj = nn.Linear(sum(d_convs), d_emb_word)
    
    def forward(self, x):
        """ Forward pass of the character-cnn module
            - Input has shape (batch_size, seq_len, n_char_max, d_emb_char)
            - Output has shape (batch_size, seq_len, d_emb_word)
        """
        # import pdb; pdb.set_trace()
        x = self._reshape(x)
        cnn_out_list = []
        for layer in self.cnn_layers:
            cnn_out, _ = torch.max(layer(x), dim=-1)
            cnn_out_list.append(self.activ_fn(cnn_out))
        concat = self._deshape(torch.cat(cnn_out_list, dim=-1))
        return self.word_emb_proj(self.highway(concat))

    def _reshape(self, x):
        self.batch_size, L, C, D = x.shape
        new_shape = (self.batch_size * L, C, D)
        return torch.reshape(x, new_shape).transpose(1, 2)

    def _deshape(self, x):
        BL, D_SUM = x.shape
        new_shape = (self.batch_size, int(BL / self.batch_size), D_SUM)
        return x.reshape(new_shape)


class Highway(nn.Module):
    def __init__(self, d_highway, n_layers=2, proj_fn=relu, gate_fn=sigmoid):
        super().__init__()
        self.n_layers = n_layers
        self.proj_fn = proj_fn
        self.gate_fn = gate_fn
        self.proj = nn.ModuleList([
            nn.Linear(d_highway, d_highway) for _ in range(n_layers)])
        self.gate = nn.ModuleList([
            nn.Linear(d_highway, d_highway) for _ in range(n_layers)])

    def forward(self, x):
        """ Forward pass of the highway module
            - Input has shape (batch_size, seq_len, d_highway)
            - Output has shape (batch_size, seq_len, d_highway)
        """
        for p, g in zip(self.gate, self.proj):
            proj = self.proj_fn(p(x))
            gate = self.gate_fn(g(x))
            x = proj * gate + x * (1.0 - gate)

        return x
    
 
class WordLSTM(nn.Module):
    def __init__(self, d_emb_word, d_lstm, n_lstm_layers=2, dropout=0.5):
        super().__init__()
        self.n_lstm_layers = n_lstm_layers
        self.forw_lstm = nn.ModuleList([nn.LSTM(input_size=d_emb_word,
                                                hidden_size=d_lstm,
                                                proj_size=d_emb_word,
                                                batch_first=True)
                                        for _ in range(n_lstm_layers)])
        self.back_lstm = nn.ModuleList([nn.LSTM(input_size=d_emb_word,
                                                hidden_size=d_lstm,
                                                proj_size=d_emb_word,
                                                batch_first=True)
                                        for _ in range(n_lstm_layers)])
        self.forw_drop = nn.Dropout(dropout)
        self.back_drop = nn.Dropout(dropout)
        
    def forward(self, x, return_all_states=False):
        """ Forward pass of the multi-layered bilateral lstm module
            - Input has shape (batch_size, seq_len, d_emb_word)
            - If return_all_states == True:
                output has shape (1+2L, batch_size, seq_len, d_emb_word)
            - If return_all_states == False: (return only output of last layer)
                output has shape (batch_size, seq_len, 2 * d_emb_word)
        """
        context_embeddings = [x]
        forw_x, back_x = x, x.flip(dims=(1,))
        for l, (back, forw) in enumerate(zip(self.forw_lstm, self.back_lstm)):
            forw_out, _ = forw(x)  # (batch_size, seq_len, d_emb_word)
            back_out, _ = back(x)  # (batch_size, seq_len, d_emb_word)
            context_embeddings.extend(
                self._combine_forward_and_backward(forw_out, back_out))
            if l < self.n_lstm_layers - 1:  # residual and dropout
                forw_x = self.forw_drop(forw_out) + forw_x
                back_x = self.back_drop(back_out) + back_x
            else:
                forw_x, back_x = forw_out, back_out
        
        if return_all_states:  # (1+2L, batch_size, seq_len, d_emb_word)
            return torch.stack(context_embeddings, dim=0)
        else:  # (batch_size, seq_len, 2 * d_emb_word)
            return torch.cat(context_embeddings[-2:], dim=-1)
    
    def _combine_forward_and_backward(self, forw_out, back_out):
        """ Align lstm backward and forward hidden states for language modeling
            - Both inputs have shape (batch_size, seq_len, d_emb_word)
            - Both outputs have shape (batch_siez, seq_len, d_emb_word)
        """
        forw_out = forw_out.roll(shifts=(1,), dims=(1,))
        back_out = back_out.roll(shifts=(1,), dims=(1,))
        forw_out[:, 0, :] = 0; back_out[:, 0, :] = 0
        return forw_out, back_out.flip(dims=(1,))