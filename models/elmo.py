import torch
import torch.nn as nn
from torch import relu, sigmoid


class ELMO(nn.Module):
    def __init__(self, d_convs, k_sizes, d_emb_char, d_emb_word,
                 n_lstm_layers, token_type, vocab_size, char_vocab_size,
                 word_vocab_size, dropout=0.5):
        super().__init__()
        self.token_type = token_type
        if token_type in ['words', 'words_and_chars']:
            self.word_embedding = nn.Embedding(word_vocab_size, d_emb_word)
        if token_type in ['chars', 'words_and_chars']:
            self.char_embedding = nn.Embedding(char_vocab_size, d_emb_char)
            self.char_cnn = CharCNN(d_convs, k_sizes, d_emb_char, d_emb_word)
        self.word_lstm = WordLSTM(n_lstm_layers, d_emb_word, dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.loss_fn = ELMOLoss()
        
    def forward(self, sample, return_all_states=False):
        """ Forward pass of the ELMO model
            - Input (batch_size, seq_len, ngram_len) or (batch_size, seq_len)
            - Output (batch_size, seq_len, d_emb_word)
        """
        static_emb = self.compute_static_embeddings(sample)
        context_emb = self.word_lstm(static_emb)
        return self.dropout(context_emb)

    def compute_static_embeddings(self, sample):
        if self.token_type == 'words':
            return self.word_embedding(sample)
        
        elif self.token_type == 'chars':
            char_emb = self.char_embedding(sample[:, :, 1:])
            return self.char_cnn(char_emb)
        
        elif self.token_type == 'words_and_chars':
            word_emb = self.word_embedding(sample[:, :, 0])
            char_emb = self.char_embedding(sample[:, :, 1:])
            char_emb = self.char_cnn(char_emb)
            return self.combine_word_char_embeddings(word_emb, char_emb)
        
        else:
            raise ValueError(
                'Invalid use_word parameter (words, chars, words_and_chars)')
    
    def combine_word_char_embeddings(self, word_emb, char_emb):
        pass

    def combine_ngram_embeddings(self, x, dim, reduce='mean'):
        if reduce == 'mean':
            norm_factor = (x != 0).sum(dim=dim).clip(min=1) / x.shape[dim]
            return x.mean(dim=dim) / norm_factor
        else:
            return x.sum(dim=dim)

    def get_embeddings(self, sample):
        """ Compute embeddings for ELMO model
            - Input (batch_size, seq_len, ngram_len) or (batch_size, seq_len)
            - Output (batch_size, seq_len, d_emb_word, 2 * n_lstm_layers + 1)
        """
        pass  # not implemented yet
        

class ELMOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.nlll_loss = nn.NLLLoss()

    def forward(self, model_output, target):
        logits = self.log_softmax(model_output)
        return self.nlll_loss(logits, target)
    

class CharCNN(nn.Module):
    def __init__(self, d_convs, k_sizes, d_emb_char, d_emb_word, activ_fn=relu):
        super().__init__()
        self.cnn_layers = nn.ModuleList([
            nn.Conv1d(in_channels=d_emb_char,
                      out_channels=d_conv,
                      kernel_size=k_size,  # kernel_size is like n-gram
                      bias=True)
            for (d_conv, k_size) in zip(d_convs, k_sizes)])
        d_conv_sum = sum([d_convs])
        self.activ_fn = activ_fn
        self.highway = Highway(d_conv_sum, n_layers=2)
        self.final_proj = nn.Linear(d_conv_sum, d_emb_word)
    
    def forward(self, x):
        """ Forward pass of the character-cnn module
            - Input has shape (batch_size, seq_len, d_emb_char)
            - Output has shape (batch_size, seq_len, d_emb_word)
        """
        cnn_out_list = []
        for layer in self.cnn_layers:
            cnn_out = self.activ_fn(layer(x).max(dim=-2))
            cnn_out_list.append(cnn_out)
        concat = torch.cat(cnn_out_list, dim=-1)
        return self.final_proj(self.highway(concat))


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
    def __init__(self, d_emb_word, d_hidden, n_lstm_layers=2, dropout=0.5):
        super().__init__()
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size=d_emb_word,
                    hidden_size=d_hidden,
                    bidirectional=True,
                    dropout=dropout,
                    proj_size=d_emb_word)
            for _ in range(n_lstm_layers)])
        
    def forward(self, x, return_all_states=False):  
        """ Forward pass of the multi-layered bilateral lstm module
            - Input has shape (batch_size, seq_len, d_emb_word)
            - Output has shape (batch_size, seq_len, d_emb_word)
        """
        context_embeddings = [x]
        for i, layer in enumerate(self.lstm_layers):
            out, _ = layer(x)  # (h, c) default to zero at init if not provided
            if i < len(self.lstm_layers) - 1:
                x = out + x  # residual connection (?)
            else:
                x = out
            context_embeddings.append(x)
            
        return context_embeddings if return_all_states else x
    