import os
import json
import numpy as np
import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity as cossim

def prediction_task_ehr(model, test_data_dir, tokenizer, cat='DIA_', topk=100):
    model.eval()  # may not be needed when used from test_fn
    test_data_path = [p for p in os.listdir(test_data_dir) if 'test' in p][0]
    test_data_path = os.path.join(test_data_dir, test_data_path)
    labels_info = get_label_infos(model, tokenizer, cat)

    # TODO: operate by batch to compute metric faster
    hits, trials = 0, 0
    with open(test_data_path, 'r') as f:
        for line in f:
            sequence = json.loads(line)
            gold_list = get_gold_list(sequence, tokenizer, cat)
            if gold_list is not None:
                embedded = get_input_embeddings(model, sequence, tokenizer, cat)
                hits += cosine_hit(embedded, labels_info, gold_list, topk)
                trials += 1

            # TODO: log in tensorboard (outside the line loop)
            topk_accuracy_any = hits / trials
            print('\rTrial: %i; easiest measure of accuracy: %.2f' %\
                    (trials, topk_accuracy_any), end='')
        
def get_label_infos(model, tokenizer, cat):
    vocab = tokenizer.get_vocab()
    labels = [tokenizer.encode(t) for t in vocab if cat in t and cat != t]
    labels_embeddings = model.get_token_embeddings(labels)
    return {'labels': labels, 'embedded': labels_embeddings}

def get_gold_list(sequence, tokenizer, cat):
    # TODO: check for unknown tokens (but rare), or have test in vocab
    gold_list = [tokenizer.encode(t) for t in sequence if cat in t]
    if len(gold_list) > 0:
        if isinstance(gold_list[0], list):
            gold_list = [elem[0] for elem in gold_list]
        return gold_list

def get_input_embeddings(model, sequence, tokenizer, cat):
    encoded = [tokenizer.encode(t) for t in sequence if cat not in t]
    if isinstance(encoded[0], list):
        weights = [1 / tokenizer.word_counts[t[0]] for t in encoded]
    else:
        weights = [1 / tokenizer.word_counts[t] for t in encoded]
    embedded = model.get_sequence_embeddings(encoded, weights)
    return np.expand_dims(embedded, axis=0)

def cosine_hit(input_embeddings, labels_info, gold_list, topk):
    similarities = cossim(input_embeddings, labels_info['embedded'])
    topk_sim_indices = np.argsort(similarities)[0][-topk:][::-1]
    topk_indices = [labels_info['labels'][k] for k in topk_sim_indices]
    if isinstance(topk_indices[0], list):
        topk_indices = [t[0] for t in topk_indices]
    return int(any([g in topk_indices for g in gold_list]))