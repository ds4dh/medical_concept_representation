import torch
import torch.nn as nn
import torch.nn.functional as F
from .bert import BERT


class BERTClassifier(nn.Module):
    """ BERTClassifier: Finetune BERT and classifier weights. """
    def __init__(self, vocab_size, special_tokens, max_seq_len, d_embed, d_ff,
                 n_layers, n_heads, n_classes, bert_ckpt_path, pos_weights,
                 dropout=0.1, grad_only_for_norm_layers=True, *args, **kwargs):
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
        self.bert = BERT(vocab_size, special_tokens, max_seq_len, d_embed,
                         d_ff, n_layers, n_heads, dropout)
        self.load_bert_weights(bert_ckpt_path)
        if grad_only_for_norm_layers:
            self.enable_grad_only_for_norm_layers()
        self.classifier = nn.Linear(d_embed, n_classes)
        self.loss_fn = BertClassifierLoss(n_classes, pos_weights)
        
    def forward(self, sample):
        """ Forward pass of the BERT Classifier """
        embeddings = self.bert(sample, get_embeddings_only=True)
        return self.classifier(embeddings[:, 0, :])  # good like this?
    
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
        print('Pre-trained weights of BERT loaded from %s' % bert_ckpt_path)
        
    def enable_grad_only_for_norm_layers(self):
        """ Set 'requires_grad' to true only for layer-norm layers of BERT.
            The goal is to avoid breaking embedings when fine-tuning on the
            downstream task. Only normalization weights are updated.
        """
        # Set all bert parameters to no-learning-mode
        for p in self.bert.parameters():
            p.requires_grad = False
            
        # Set the layer-norm parameters back to learning-mode
        for name, param in self.bert.named_parameters():
            if 'norm' in name:
                param.requires_grad = True
    
    def validation_metric(self, prediction, target):
        pass
                

class BertClassifierLoss(nn.Module):
    def __init__(self, n_classes, pos_weights):
        super().__init__()
        self.n_classes = n_classes
        pos_weights = torch.tensor(pos_weights)
        self.bce_logits_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        
    def forward(self, model_output, reagent_label):
        one_hot_label = self.multi_label_one_hot(reagent_label)
        one_hot_label = one_hot_label.to(model_output.device).float()
        return self.bce_logits_loss(model_output, one_hot_label)

    def multi_label_one_hot(self, batch):
        return torch.stack(tuple(map(self.one_hot_tensor, batch)), dim=0)
    
    def one_hot_tensor(self, sample):
        return F.one_hot(torch.tensor(sample), self.n_classes).sum(dim=0)
