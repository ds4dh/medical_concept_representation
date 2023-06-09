import torch
import torch.nn as nn
import torch.nn.functional as F
from .bert import BERT


class BERTClassifier(nn.Module):
    """ BERTClassifier: Finetune BERT and classifier weights. """
    def __init__(self, vocab_sizes, special_tokens, max_seq_len, d_embed,
                 d_ff, n_layers, n_heads, n_classes, bert_ckpt_path,
                 pos_weights, bert_grad_type, dropout=0.1, *args, **kwargs):
        """
        :param voc_size: vocabulary size of words given to the model
        :param pad_id: index of the padding token
        :param d_embed: BERT model hidden size
        :param n_layers: numbers of encoder blocks (layers) in BERT
        :param n_heads: number of attention heads in each attention layer
        :param n_classes: number of output classes predicted by the classifier
        :param dropout: dropout rate in all layers and sublayers of BERT
        """
        super().__init__()
        self.bert = BERT(vocab_sizes, special_tokens, max_seq_len, d_embed,
                         d_ff, n_layers, n_heads, dropout)
        
        if bert_ckpt_path is not None:
            print('BERT pre-trained weights loaded from %s.' % bert_ckpt_path)
            self.load_bert_weights(bert_ckpt_path)
            self.set_grad_for_bert(bert_grad_type)
        else:
            print('BERT weights trained from scratch with the classifier.')
        
        self.classifier = nn.Linear(d_embed, n_classes)
        self.loss_fn = BertClassifierLoss(n_classes, pos_weights)
        
    def forward(self, sample):
        """ Forward pass of the BERT Classifier """
        embeddings = self.bert(sample, get_embeddings_only=True)
        cls_embedding = embeddings[:, 0, :]
        rest_embedding = embeddings[:, 1:, :].mean(dim=1)
        return self.classifier(cls_embedding + rest_embedding)
    
    def load_bert_weights(self, bert_ckpt_path):
        """ Load pre-trained weights for BERT """
        state_dict = torch.load(bert_ckpt_path)['state_dict']
        state_dict = {key.replace('model.', ''): parameters
                      for key, parameters in state_dict.items()}
        try:
            self.bert.load_state_dict(state_dict)
        except RuntimeError as error:
            print('Tip: maybe the dataset is not the same as what BERT was',
                  'trained on (is debug mode enabled?)')
            raise error
        
    def set_grad_for_bert(self, bert_grad_type):
        """ Set 'requires_grad' for the parameters of BERT
            - 'none' will let no weights of BERT be fine-tuned
            - 'norm' will only let the norm weights of BERT be fine-tuned
            - 'all' will let all weights of BERT be fine-tuned on the task
        """
        # TODO: implement 'last' method
        # Set all bert parameters to no-learning-mode
        if bert_grad_type in ['none', 'norm']:
            for param in self.bert.parameters():
                param.requires_grad = False
                            
        # Set the layer-norm parameters back to learning-mode
        if bert_grad_type in ['norm']:
            for name, param in self.bert.named_parameters():
                if 'norm' in name:
                    param.requires_grad = True
    
    def val_metric(self, logger, batch_idx, step, outputs, label,
                   test_mode=False, thresh=0.5):
        """ Compute different metrics and scores for the model output
        """
        # Load data and generate logits
        logits = torch.sigmoid(outputs)
        gold = self.loss_fn.multi_label_one_hot(label)
        gold = gold.to(outputs.device).float()

        # Compute top1 accuracy and log it
        pred = (logits > thresh).float()
        correct = [p.tolist() == g.tolist() for p, g in zip(pred, gold)]
        top1 = torch.tensor(correct).float().mean()
        if batch_idx == 0:
            for s, (p, g) in enumerate(zip(pred, gold)):
                pred_list = [i for i, x in enumerate(p.tolist()) if x == 1]
                gold_list = [i for i, x in enumerate(g.tolist()) if x == 1]
                to_log = 'top1: %s, thresh = %s  \npred: %s  \ngold: %s'\
                        % (top1, thresh, str(pred_list), str(gold_list))
                logger.add_text('sample %s' % s, to_log, global_step=step)
                if not test_mode: break

        # Compute true / false positive / negative rates
        TP = (pred * gold).mean()                # true positives
        TN = ((1 - pred) * (1 - gold)).mean()    # true negatives
        FP = ((pred - gold) > 0).float().mean()  # false positives
        FN = ((gold - pred) > 0).float().mean()  # false negatives
        
        # Compute metrics
        acc = (TP + TN) / (FP + FN + TP + TN)    # accuracy
        prec = TP / (FP + TP)                    # precision
        rec = TP / (FN + TP)                     # recall
        f1 = 2 * (prec * rec) / (prec + rec)     # f1-score
        
        return {'top1': top1, 'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}

    def test_metric(self, **kwargs):
        """ Compute same metrics as in validation but with more samples
        """
        return self.val_metric(test_mode=True, **kwargs)
    

class BertClassifierLoss(nn.Module):
    def __init__(self, n_classes, pos_weights, mode='bce'):
        super().__init__()
        self.n_classes = n_classes
        pos_weights = torch.tensor(pos_weights)
        if mode == 'bce':
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        elif mode == 'focal':
            self.loss_fn = FocalLoss(weight=pos_weights)
        else:
            raise ValueError('Invalid loss mode given to bert classifier loss')
        
    def forward(self, model_output, label):
        one_hot_label = self.multi_label_one_hot(label)
        one_hot_label = one_hot_label.to(model_output.device).float()
        return self.loss_fn(model_output, one_hot_label)
    
    def multi_label_one_hot(self, batch):
        return torch.stack(tuple(map(self.one_hot_tensor, batch)), dim=0)
    
    def one_hot_tensor(self, sample):
        return F.one_hot(torch.tensor(sample), self.n_classes).sum(dim=0)


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        # weight parameter acts as the alpha parameter to balance class weights
        self.weight = weight
        self.gamma = gamma
        
    def forward(self, input, target):
        ce_loss = F.cross_entropy(input,
                                  target,
                                  reduction=self.reduction,
                                  weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
    
