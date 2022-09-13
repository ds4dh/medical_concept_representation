import torch
import torch.nn as nn


# IMPORTANT MAKE SURE THAT PADDING TOKENS DO NOT INFLUENCE SUM OVER NGRAMS IN NGRAMS-GLOVE!!!
class Glove(nn.Module):
    """ Glove model.
    """

    def __init__(self, vocab_size, d_embed, *args, **kwargs):
        super().__init__()
        
        self.l_emb = nn.Embedding(vocab_size, d_embed)  # check padding idx parameter for ngram-glove
        self.l_bias = nn.Embedding(vocab_size, 1)
        self.r_emb = nn.Embedding(vocab_size, d_embed)
        self.r_bias = nn.Embedding(vocab_size, 1)
        
        self.loss_fn = GloveLoss()

    def forward(self, left, right):
        # Compute embeddings
        l_v = self.l_emb(left)
        l_b = self.l_bias(left)
        r_v = self.r_emb(right)
        r_b = self.r_bias(right)
        
        # Sum over ngram dimension if existing
        if len(l_v.shape) > 2:  # sum should be consistent with padding tokens
            l_v = l_v.sum(dim=-2)
            l_b = l_b.sum(dim=-2)
            r_v = r_v.sum(dim=-2)
            r_b = r_b.sum(dim=-2)
                    
        return (l_v * r_v).sum(dim=-1) + l_b.squeeze() + r_b.squeeze()

    def get_embeddings(self):
        left, right = self.l_emb.weight.detach().cpu().numpy(), \
                      self.r_emb.weight.detach().cpu().numpy()

        return {"left": left, "right": right, "embeddings": left + right}

    def export_as_gensim(self, path, tokenizer):
        left, right, embeddings = self.get_embeddings().values()

        with open(path, 'w', encoding='utf-8') as f:
            for tok, emb in zip(tokenizer.encoder.keys(), embeddings.tolist()):
                f.write(str(tok) + ' ' + str(emb).replace('[', '') \
                                                 .replace(']', '') \
                                                 .replace(',', '') + '\n')


class GloveLoss(nn.Module):

    def __init__(self, x_max=100, alpha=3/4):
        """ Hyperparameters as in the original article.
        """
        super().__init__()
        self.m = x_max
        self.a = alpha

    def normalize(self, t):
        """ Normalization as in the original article.
        """
        return torch.where(t < self.m, (t / self.m) ** self.a, torch.ones_like(t))
    
    def forward(self, model_output, cooc):
        """ Expects flattened model output and target coocurence matrix.
        """
        n_t = self.normalize(cooc)
        l_t = torch.log(cooc)
        
        return torch.sum(n_t * (model_output - l_t) ** 2)
