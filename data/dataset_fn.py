import os
import json
import torch
import torch.nn.functional as F
import datasets
from tokenizers import Tokenizer
from tokenizers.models import WordLevel, BPE
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


class CBOWDataHolder():
    def __init__(self, data_dir, tokenizer_type, cbow_size):
        ''' Initialize a dataset for continuous bag of words task.
        
        Params:
        -------
        data_dir: str
            Path to the directory containing the data sets.
        tokenizer_type: str
            Type of tokenizer to use ('opennmt', 'minimal', 'full')
        context_size: int
            Number of tokens to use as context of predicted token.
        
        '''
        # Initialize cbow parameters
        self.cbow_size = cbow_size
        self.cbow_indices = torch.tensor(
                    [i for i in range(2 * cbow_size + 1) if i != cbow_size])

        # Get the original datasets
        data_path = 'conceptofmind/wikitext-2-v1-clean'  # TODO: use real data
        self.datasets = datasets.load_dataset(path=data_path,
                                              split={'train': 'train[:80%]',
                                                     'val': 'train[80%:90%]',
                                                     'test': 'train[90%:]'})

        # Replace text sequences by their tokenized versions
        self.tokenizer = self.build_and_train_word_tokenizer()
        self.add_subwords_to_vocabulary()
        tokenize_fn = lambda x: {'input_ids':
                    [e.ids for e in self.tokenizer.encode_batch(x['text'])]}
        self.datasets = self.datasets.map(tokenize_fn, batched=True)
        self.datasets.set_format(type='torch', columns=['input_ids'])

        # Compute fraction of each token in the datasets
        self.token_fractions = self.compute_token_keep_probs(data_dir)

    @staticmethod
    def collate_fn(batch, cbow_size, cbow_indices, pad_id=0):
        ''' Generate a set of (context, target) pairs and build the batch
            as a batched tensor for the continuous bag of words task.
        
        Params:
        -------
        sequence: str
            Text to split into a set of (context, target) pairs.
        cbow_size: int
            One-sided size of the context window in the cbow task.
        cbow_indices: torch.Tensor of shape (2 * cbow_size + 1)
            Indices that correspond to the cbow context in each sequence.

        '''
        context, target = [], []
        for sample in batch:
            sequence = sample['input_ids']
            if len(sequence) < 2 * cbow_size + 1:
                pad_len = (2 * cbow_size + 1) - len(sequence)
                sequence = F.pad(sequence, (0, pad_len), value=pad_id)
            for i in range(cbow_size, len(sequence) - cbow_size):
                words = sequence[i - cbow_size:i + cbow_size + 1]
                context.append(words.index_select(0, cbow_indices))
                target.append(sequence[i])
        return {'context': torch.stack(context), 'target': torch.stack(target)}
        
    def subsample(self, sequence):
        for token_id in sequence:
            if torch.rand() > self.token_fractions:
                pass

    def add_subwords_to_vocabulary(self):
        vocab = self.tokenizer.get_vocab()
        ngram_list = []
        for k in vocab.keys():
            ngram_list.append(self.generate_ngram_list(k))
        self.tokenizer.add_tokens(ngram_list)
    
    def generate_ngram_list(word):
        pass

    def build_and_train_word_tokenizer(self):
        # Define iterator to train the tokenizer
        dataset = self.get_dataset('train')
        def get_training_corpus():
            for start_idx in range(0, len(dataset), 1000):
                samples = dataset[start_idx : start_idx + 1000]
                yield samples['text']

        # Build and train the tokenizer (simple word-level here)
        special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]"]
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=special_tokens)
        tokenizer.train_from_iterator(get_training_corpus(), trainer)
        return tokenizer

    def compute_token_keep_probs(self, data_dir, thresh=1e-4, recompute=False):
        if recompute:
            # Add new word to the vocab dictionnary and count occurences
            word_occurences, keep_probs = {}, {}
            dataset = self.get_dataset('train')
            for idx, sample in enumerate(dataset):
                print(f'\rWord freq. computed {idx} / {len(dataset)}', end='')
                sentence = sample['input_ids']
                for token_id in sentence:
                    try:
                        word_occurences[token_id] += 1
                    except KeyError:
                        word_occurences[token_id] = 1

            # Compute the probability of not discarding each vocabulary word
            sum_of_all_words = sum(word_occurences.values())
            for token_id, word_occurence in word_occurences.items():
                if word_occurence > 5:  # from FastText paper
                    word_fraction = word_occurence / sum_of_all_words
                    keep_score = (thresh / word_fraction) ** 0.5
                    keep_probs[token_id] = min(keep_score, 1.0)
            
            # Save the dictionary to avoid recomputing
            with open(os.path.join(data_dir, 'data_info.json'), 'w') as f:
                json.dump(keep_probs, f)
        
        # Load from file if already computed
        else:
            with open(os.path.join(data_dir, 'data_info.json'), 'r') as f:
                keep_probs = json.load(f)
        return keep_probs

    def get_dataset(self, split):
        ''' Get the source and target datasets for the given split.

        '''
        return self.datasets[split]
    
    def get_collate_fn(self):
        ''' Get the collate function for the dataset.

        '''
        return lambda x: self.collate_fn(x, self.cbow_size, self.cbow_indices)
    
    def decode_text(self, input_ids):
        ''' Decode text from the dataset.
               
        '''
        return self.tokenizer.decode(input_ids, skip_special_tokens=True)
