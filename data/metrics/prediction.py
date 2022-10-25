import os
import json
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity as cossim

def prediction_task_ehr(model, test_data_dir, tokenizer, cat='DIA_', topk=100):
    model.eval()  # may not be needed when used from test_fn
    test_data_path = [p for p in os.listdir(test_data_dir) if 'test' in p][0]
    test_data_path = os.path.join(test_data_dir, test_data_path)
    labels, labels_emb = get_label_emb(model, tokenizer, cat)

    # TODO: operate by batch to compute metric faster
    # TODO: do frequency-weighted average (use tokenizer.word_count, maybe sequence_embeddings return all sequence?)
    hits, trials = 0, 0
    with open(test_data_path, 'r') as f:
        for line in f:
            sequence = json.loads(line)
            gold_list = get_gold_list(sequence, tokenizer, cat)
            if gold_list is not None:
                input_emb = get_input_emb(model, sequence, tokenizer, cat)
                hits += cosine_hit(input_emb, labels_emb, labels, gold_list, topk)
                trials += 1

            # TODO: log in tensorboard (outside the line loop)
            topk_accuracy_any = hits / trials
            print('\rTrial: %i; easiest measure of accuracy: %.2f' %\
                    (trials, topk_accuracy_any), end='')
        
def get_label_emb(model, tokenizer, cat):
    vocab = tokenizer.get_vocab()
    labels = [tokenizer.encode(t) for t in vocab if cat in t and cat != t]
    labels_emb = model.get_token_embeddings(labels)  # TODO: CHECK WITH ELMO WITH CAPS
    return labels, labels_emb

def get_gold_list(sequence, tokenizer, cat):
    # TODO: check for unknown tokens (but rare), or have test in vocab
    gold_list = [tokenizer.encode(t) for t in sequence if cat in t]
    if len(gold_list) > 0:
        if isinstance(gold_list[0], list):
            gold_list = [elem[0] for elem in gold_list]
        return gold_list

def get_input_emb(model, sequence, tokenizer, cat):
    input_sequence = [tokenizer.encode(t) for t in sequence if cat not in t]
    input_emb = model.get_sequence_embeddings(input_sequence)
    return np.expand_dims(input_emb, axis=0)

def cosine_hit(input_emb, labels_emb, labels, gold_list, topk):
    similarities = cossim(input_emb, labels_emb)
    topk_sim_indices = np.argsort(similarities)[0][-topk:][::-1]
    topk_indices = [labels[k] for k in topk_sim_indices]
    if isinstance(topk_indices[0], list):
        topk_indices = [t[0] for t in topk_indices]
    return int(any([g in topk_indices for g in gold_list]))
