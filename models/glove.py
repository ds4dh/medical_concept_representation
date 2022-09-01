import tqdm
import torch
import torch.nn as nn


class Glove(nn.Module):
    """
    Glove model.
    """

    def __init__(self, vocab_size, d_embed, *args, **kwargs):

        super().__init__()

        self.l_emb = nn.Embedding(vocab_size, d_embed)
        self.l_bias = nn.Embedding(vocab_size, 1)

        self.r_emb = nn.Embedding(vocab_size, d_embed)
        self.r_bias = nn.Embedding(vocab_size, 1)
        
        self.loss_fn = GloveLoss()

    def forward(self, left_id, right_id):

        l_v, l_b = self.l_emb(left_id), self.l_bias(left_id).squeeze()
        r_v, r_b = self.r_emb(right_id), self.r_bias(right_id).squeeze()

        return (l_v * r_v).sum(-1) + l_b + r_b

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
        """
        Hyperparameters as in the original article.
        """
        super().__init__()

        self.m = x_max
        self.a = alpha

    def normalize(self, t):
        """
        Normalization as in the original article.
        """
        return torch.where(condition=t < self.m,
                           x=(t / self.m) ** self.a,
                           y=torch.ones_like(t))

    def forward(self, prediction, target):
        """
        Expects flattened predictions and target coocurence matrix.
        """
        n_t = self.normalize(target)
        l_t = torch.log(target)

        return torch.sum(n_t * (prediction - l_t) ** 2)


def one_train(model,
              iterator,
              optimizer,
              criterion,
              device):

    # Needed to store losses.
    epoch_loss = 0

    # Model in training stage.
    model.train()

    for batch in tqdm(iterator, desc="Training", leave=False):

        # At each batch, we backpropagate the signal.
        optimizer.zero_grad()

        # Retrieve batch elements.
        l, r, c = batch["left"].to(device), batch["right"].to(
            device), batch["cooc"].to(device)

        # Compute predictions.
        predictions = model(l, r)

        # Compute loss.
        loss = criterion(predictions, c)

        # Compute gradients.
        loss.backward()

        # Backpropagate.
        optimizer.step()

        # Store loss.
        epoch_loss += loss.item()

    return epoch_loss/len(iterator)
