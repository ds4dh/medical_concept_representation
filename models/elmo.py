import torch
import torch.nn as nn
from itertools import zip_longest
from torch import relu, sigmoid


class ELMO(nn.Module):
    def __init__(self, n_lstm_layers, d_lstm, d_embed_char, d_embed_word,
                 d_conv, k_size, vocab_sizes, special_tokens, token_type,
                 dropout=0.5, *args, **kwargs):
        super().__init__()
        # Retrieve vocabulary information (char_shift for char embeddings)
        self.pad_id = special_tokens['[PAD]']
        self.bos_id = special_tokens['[CLS]']
        self.eos_id = special_tokens['[END]']
        self.special_token_ids = list(special_tokens.values())
        self.char_shift = vocab_sizes['special'] + vocab_sizes['word'] - 1
        
        # Check types of token embeddings used in the model
        self.token_type = token_type
        assert (token_type == 'char' and 'ngram' in vocab_sizes) or \
               (token_type == 'both' and 'ngram' in vocab_sizes) or \
               (token_type == 'word' and 'ngram' not in vocab_sizes), \
               'Bad token_type / tokenizer combination. Good combinations ' \
               'are: (char or both / subword or icd, word / word)'

        # Word embeddings
        word_vocab_size = vocab_sizes['word'] + vocab_sizes['special']
        if self.token_type in ['word', 'both']:
            self.word_embedding = nn.Embedding(word_vocab_size,
                                               d_embed_word,
                                               padding_idx=self.pad_id)
        
        # Layers from character to word embeddings
        if self.token_type in ['char', 'both']:
            char_vocab_size = vocab_sizes['ngram'] + vocab_sizes['special']
            self.char_embedding = nn.Embedding(char_vocab_size + 1,  # for pad
                                               d_embed_char,
                                               padding_idx=self.pad_id)
            self.char_cnn = CharCNN(d_conv, k_size, d_embed_char, d_embed_word)
        
        # Network layers
        self.word_lstm = WordLSTM(d_embed_word, d_lstm, n_lstm_layers, dropout)
        self.final_proj = nn.Linear(2 * d_embed_word, word_vocab_size)

        # Loss function used by the model
        self.loss_fn = ELMOLoss()
        
    def forward(self, sample):
        """ Forward pass of the ELMO model
            - Inputs:
                sample (batch_size, seq_len, ngram_len) or (batch_size, seq_len)
            - Output (batch_size, seq_len, d_emb_word)
        """
        static_emb = self.compute_static_embeddings(sample)
        context_emb = self.word_lstm(static_emb, return_all_states=False)
        return self.final_proj(context_emb)  # for pre-training

    def parse_words_and_chars(self, sample):
        if self.token_type in ['both', 'char']:
            chars = sample[:, :, 1:] - self.char_shift  # for char embedding
            chars[chars < 0] = self.pad_id  # only char special token: pad_id
            if self.token_type == 'both':
                words = sample[:, :, 0]
                return words, chars
            else:
                return None, chars
        
        else:  # 'word'
            return sample, None
        
    def compute_static_embeddings(self, sample):
        """ Create word static embeddings from characters and/or words
            - Inputs:
                sample (batch_size, seq_len, ngram_len) or (batch_size, seq_len)
            - Output (batch_size, seq_len, d_emb_word)
        """
        words, chars = self.parse_words_and_chars(sample)

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

    def get_token_embeddings(self, token_indices):
        """ Compute static embeddings for a list of tokens
        """
        embeddings = []
        for token_index in token_indices:
            token_index = torch.tensor(token_index)[None, None, ...]
            embedded = self.compute_static_embeddings(token_index)
            embeddings.append(embedded.squeeze())
        return torch.stack(embeddings, dim=0).detach().cpu().numpy()
    
    def get_sequence_embeddings(self, sequence, mode='context'):
        """ Compute contextualized embeddings for a sequence of tokens
        """
        sequence = self.pre_process_for_embeddings(sequence)
        static_emb = self.compute_static_embeddings(sequence)
        context_emb = self.word_lstm(static_emb, return_all_states=True)
        embedded = context_emb[-2:].mean(dim=(0, 2))  # sequence embeddings
        # TODO: mean over dimension 2 should be weighted, dimension 0 should be conditional on full or last only mode
        return embedded.squeeze().detach().cpu().numpy()
    
    def pre_process_for_embeddings(self, sequence):
        if isinstance(sequence[0], list):  # ngram case
            sequence.insert(0, [self.bos_id]); sequence.append([self.eos_id])
            sequence = list(zip(*zip_longest(*sequence, fillvalue=self.pad_id)))
        else:
            sequence.insert(0, self.bos_id); sequence.append(self.eos_id)
        return torch.tensor(sequence)[None, ...]  # add batch dimension
        

class ELMOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.nlll_loss = nn.NLLLoss()

    def forward(self, model_output, sample):
        if len(sample.shape) > 2:  # ngram case
            sample = sample[:, :, 0]
        logits = self.log_softmax(model_output).transpose(2, 1)
        return self.nlll_loss(logits, sample)
    

class CharCNN(nn.Module):
    def __init__(self, d_convs, k_sizes, d_emb_char, d_emb_word, activ_fn=relu):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=d_emb_char,
                      out_channels=d_conv,
                      kernel_size=k_size)  # kernel_size is like n-gram
            for (d_conv, k_size) in zip(d_convs, k_sizes)])
        self.activ_fn = activ_fn
        self.highway = Highway(sum(d_convs), n_layers=2)
        self.word_emb_proj = nn.Linear(sum(d_convs), d_emb_word)
    
    def forward(self, x):
        """ Forward pass of the character-cnn module
            - Input has shape (batch_size, seq_len, n_char_max, d_emb_char)
            - Output has shape (batch_size, seq_len, d_emb_word)
        """
        x = self._reshape(x)
        cnn_out_list = []
        for layer in self.conv_layers:
            try:
                cnn_out, _ = torch.max(layer(x), dim=-1)
            except RuntimeError:  # if word too small for conv filter size  # WRONG IF SIZE > 1 -> BETTER SOLUTION: PADDING IN CONV
                x_adapted = x.repeat((1, 1, layer.kernel_size[0]))  # WRONG IF SIZE > 1 -> BETTER SOLUTION: PADDING IN CONV
                cnn_out, _ = torch.max(layer(x_adapted), dim=-1)  # WRONG IF SIZE > 1 -> BETTER SOLUTION: PADDING IN CONV
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
                output has shape (1 + 2L, batch_size, seq_len, d_emb_word)
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
        
        if return_all_states:  # (1 + 2L, batch_size, seq_len, d_emb_word)
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
