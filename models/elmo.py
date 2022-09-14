import torch
import torch.nn as nn
import torch.nn.functional as F


class Elmo(nn.Module):
    def __init__(self, d_embed, d_lstm, n_tags, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.word_lstm = nn.LSTM(d_embed, d_lstm, bidirectional=True)
        self.classifier = Classifier(layers=[2 * d_lstm, n_tags], drops=[0.5])
        
    def forward(self, input):
        # word_dim = (batch_size x seq_len)
        # char_dim = (batch_size x seq_len x word_len)
        
        word_emb = self.dropout(input.transpose(0, 1))
        output, (h, c) = self.word_lstm(word_emb)  # shape = S * B * d_lstm
        output = self.dropout(output)
        
        output = self.classifier(output)
        return output  # shape = S * B * n_tags
    
    
class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.drop = nn.Dropout(dropout)
        self.lin = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.lin(self.drop(self.bn(x)))


class Classifier(nn.Module):
    def __init__(self, d_lstm, n_tags, layers, drops):
        super().__init__()
        self.d_lstm = d_lstm
        self.n_tags = n_tags
        self.layers = nn.ModuleList([
            LinearBlock(layers[i], layers[i + 1], drops[i])
            for i in range(len(layers) - 1)])

    def forward(self, input):
        output = input
        sl,bs,_ = output.size()
        x = output.view(-1, self.d_lstm * 2)
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)
            
        return l_x.view(sl, bs, self.n_tags)


class ElmoLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, input_, target):
        pass


class Highway(nn.Module):
    """ Taken from https://github.com/kefirski/pytorch_Highway """
    def __init__(self, size, num_layers, non_linear_fn=F.relu):
        super().__init__()
        self.num_layers = num_layers
        self.non_linear_fn = non_linear_fn
        self.non_linear = nn.ModuleList([nn.Linear(size, size)
                                         for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size)
                                     for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size)
                                   for _ in range(num_layers)])

    def forward(self, x):
        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            non_linear = self.non_linear_fn(self.non_linear[layer](x))
            linear = self.linear[layer](x)
            x = gate * non_linear + (1 - gate) * linear

        return x
