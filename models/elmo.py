import torch
import torch.nn as nn
from torch import relu, sigmoid


class ELMO(nn.Module):
    def __init__(self, char_vocab_size, word_vocab_size, cnn_params,
                 d_embed_char, d_embed_word, n_lstm_layers, dropout=0.5):
        super().__init__()
        # self.word_embedding = nn.Embedding(word_vocab_size, d_embed_word)
        self.char_embedding = nn.Embedding(char_vocab_size, d_embed_char)
        self.char_cnn = CharCNN(cnn_params, d_embed_char, d_embed_word)
        self.word_lstm = WordLSTM(n_lstm_layers, d_embed_word, dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.loss_fn = ELMOLoss()
        
    def forward(self, chars, words=None):
        """ Forward pass of the ELMO model
            - Input chars have shape (batch_size, seq_len, word_max_len)
            - Input words have shape (batch_size, seq_len)
            - Output has shape (batch_size, ???) -> need to see with the task
        """
        # if words != None:
        #     whole_word_emb = self.word_embedding(words)  # (B, L, d_embed_word)
        #     pass  # what to do here?
        char_emb = self.char_embedding(chars)  # (B, L, word_max_len, d_embed_char)
        static_emb = self.char_cnn(char_emb)  # (B, L, d_embed_word)
        context_emb = self.word_lstm(static_emb)  # (B, L, d_embed_word)
        return self.dropout(context_emb)

    def get_embeddings(self, chars, words=None):
        """ This should return the static + all context embeddings
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
    def __init__(self, conv_params, d_embed_char, d_embed_word, activ_fn=relu):
        super().__init__()
        self.cnn_layers = nn.ModuleList([
            nn.Conv1d(in_channels=d_embed_char,
                      out_channels=d_conv,
                      kernel_size=kernel_size,  # kernel_size is like n-gram
                      bias=True)
            for (d_conv, kernel_size) in conv_params])
        d_conv_sum = sum([p[0] for p in conv_params])
        self.activ_fn = activ_fn
        self.highway = Highway(d_conv_sum, n_layers=2)
        self.final_proj = nn.Linear(d_conv_sum, d_embed_word)
    
    def forward(self, x):
        """ Forward pass of the character-cnn module
            - Input has shape (batch_size, seq_len, d_embed_char)
            - Output has shape (batch_size, seq_len, d_embed_word)
        """
        cnn_out_list = []
        for layer in self.cnn_layers:
            cnn_out = self.activ_fn(layer(x).max(dim=-2))  # (B, L, d_conv[i])
            cnn_out_list.append(cnn_out)
        concat = torch.cat(cnn_out_list, dim=-1)  # (B, L, sum([d_conv]))
        return self.final_proj(self.highway(concat))  # (B, L, d_embed_word)


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
            x = gate * proj + (1.0 - gate) * x

        return x


class WordLSTM(nn.Module):
    def __init__(self, d_embed_word, d_hidden, n_lstm_layers=2, dropout=0.5):
        super().__init__()
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size=d_embed_word,
                    hidden_size=d_hidden,
                    bidirectional=True,
                    dropout=dropout,
                    proj_size=d_embed_word)
            for _ in range(n_lstm_layers)])
        
    def forward(self, x):  
        """ Forward pass of the multi-layered bilateral lstm module
            - Input has shape (batch_size, seq_len, input_size)
            - Output has shape (batch_size, seq_len, d_embed_word)
        """
        for i, layer in enumerate(self.lstm_layers):
            out, (h, c) = layer(x)  # h and c default to zero at init if not provided
            if i < len(self.lstm_layers) - 1:
                x = out + x  # residual connection (?)
            else:
                x = out
        
        return x
    