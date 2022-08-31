import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import CBOWDataHolder


class FastText(nn.Module): 
    def __init__(self, vocab_size, d_embed):
        super(FastText, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_embed)
        self.fc = nn.Linear(d_embed, vocab_size)

    def forward(self, x):
        x = self.embed(x)  # (batch_size, seq_len, vec_dim)
        y = torch.mean(x, dim=1)  # Average over the word embeddings
        return self.fc(y)  # (batch_size, label_size)


def train_fn(model, train_dl, epoch, optim, loss_fn, max_batches): 
    model.train()
    n_batches = min(len(train_dl), max_batches)
    total_loss = 0.0
    for batch_idx, batch in enumerate(train_dl):
        # For shorter runs
        if batch_idx >= max_batches:
            break

        # Compute model output
        data, label = batch['context'], batch['target']
        optim.zero_grad()
        logits = model(data)

        # Calculate the loss and backpropagate
        loss = loss_fn(logits, label)
        loss.backward()
        optim.step()

        # Compute model accuracy and update total loss
        predictions = logits.argmax(dim=1)
        local_acc = (predictions == label).sum().item() / label.size(0)
        total_loss += loss.item()

        # Print status information
        print('\rEpoch: %i | Batch: %i/%i | Loss: %.4f | Accuracy: %.4f' % \
            (epoch, batch_idx + 1, n_batches, loss.item(), local_acc), end='')
    
    # Return epoch info
    mean_loss = total_loss / (batch_idx + 1)
    return mean_loss


def valid_fn(model, valid_dl, epoch, max_batches): 
    model.eval()
    n_batches = min(len(valid_dl), max_batches)
    n_correct = 0 
    n_total = 0
    with torch.no_grad(): 
        for batch_idx, batch in enumerate(valid_dl): 
            # For shorter runs
            if batch_idx >= max_batches:
                break
            
            # Compute the output of the model
            data, label = batch['context'], batch['target']
            logits = model(data)
            predictions = logits.argmax(dim=1)

            # Compute model hits
            n_trials = label.size(0)
            n_hits = (predictions == label).sum().item()
            local_acc = n_hits / n_trials
            print('\rValidation | Batch: %i/%i | Accuracy: %.4f' % \
                (batch_idx + 1, n_batches, local_acc), end='')

            # Update the number of correct predictions
            n_correct += n_hits
            n_total += n_trials

        # Compute valid accuracy and print results
        mean_accuracy = (n_correct / n_total)
        return mean_accuracy


if __name__ == '__main__':
    # Training parameters
    num_workers = 0
    lr = 1e-3
    n_epochs = 10
    batch_size = 64
    max_batches = torch.inf  # 500

    # Datasets
    data_dir = './data/json'
    tokenizer_type = 'code'  # 'code' / 'subcode'
    cbow_size = 5
    dataset_holder = CBOWDataHolder(data_dir=data_dir,
                                    tokenizer_type=tokenizer_type,
                                    cbow_size=cbow_size)
    train_dataset = dataset_holder.get_dataset('train')
    valid_dataset = dataset_holder.get_dataset('val')
    collate_fn = dataset_holder.get_collate_fn()

    # Dataloaders
    train_dl = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          collate_fn=collate_fn)
    valid_dl = DataLoader(dataset=valid_dataset,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          collate_fn=collate_fn)
    
    # Model
    embed_dim = 512
    vocab_size = len(dataset_holder.tokenizer.get_vocab().keys())
    model = FastText(vocab_size=vocab_size, embed_dim=embed_dim)
    
    # Training and testing
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LinearLR(optim, 1.0, 0.0, n_epochs)
    loss_fn = nn.CrossEntropyLoss()
    print('\nTraining started...')
    for epoch in range(n_epochs):
        train_loss = train_fn(model, train_dl, epoch, optim, loss_fn, max_batches)
        sched.step()
        print('\n\t--> Train epoch mean loss: %.4f | Learning rate: %.4f' % \
            (train_loss, optim.param_groups[0]['lr']))
        valid_accuracy = valid_fn(model, valid_dl, epoch, max_batches)
        print('\n\t--> Valid epoch mean accuracy: %.4f' % (valid_accuracy))
    print('Training finished!\n')
    