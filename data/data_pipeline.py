import os
import data
from torchdata.datapipes.iter import (
    FileLister,
    OnDiskCacheHolder,
    EndOnDiskCacheHolder,
    FileOpener,
    Filter,
    LineReader,
    UnBatcher,
    Shuffler,
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


class DataHolder():
    """ Pipeline for a dataset consisting in a collection of token sequences.
        Each sequence can be {'src': [...], 'tgt': [...]}, or be standalone.
    """
    def __init__(self, data_dir, data_subdir, data_keys, special_tokens,
                 max_tokens, max_len, load_tokenizer):
        self.data_dir = data_dir
        self.data_subdir = data_subdir
        self.data_keys = data_keys
        self.max_len = max_len
        self.max_tokens = max_tokens
        self.special_tokens = special_tokens
        self.tokenizer = self.get_tokenizer(load_tokenizer)
        self.vocab = self.tokenizer.get_vocab()
        self.special_ids = {'bos_id': self.vocab[special_tokens['bos']],
                            'eos_id': self.vocab[special_tokens['eos']],
                            'pad_id': self.vocab[special_tokens['pad']]}
    
    # OLD PIPELINE FUNCTION
    # def pipeline(self, split, shuffle=False, for_tokenizer=False):
    #     """ Data pipeline for a sequence dataset. The pipeline extracts data
    #         from a given file, batches it efficiently and tokenizes samples.
    #     Args:
    #         split (str): split of the data to take ('train', 'val', 'test')
    #         shuffle (bool): whether batches are shuffled
    #         for_tokenizer (bool): whether the pipeline is used to train a
    #             tokenizer or a whole model.
    #     Returns:
    #         iterable: the pipeline that goes from a file to batched tensors.
    #     """        
    #     # Extract samples from the selected file of the data directory
    #     dp = FileLister(os.path.join(self.data_dir, self.data_subdir))
    #     dp = Filter(dp, lambda t: split in t)
    #     dp = FileOpener(dp)
    #     dp = LineReader(dp)
    #     dp = data.JsonFileParser(dp)
    #     # Batching (grouped by similar lengths, dynamic batch size, shuffle)
    #     if not for_tokenizer:
    #         dp = BucketBatcher(dp, 1024, bucket_num=8, sort_key=data.sort_fn)
    #         dp = UnBatcher(dp)
    #         len_fn = partial(data.len_fn, unique=True, method='sum')
    #         dp = DynamicBatcher(dp, self.max_tokens, len_fn=len_fn)
    #         if shuffle: dp = Shuffler(dp)
    #     # Tokenize and pad sequences, transform to tensors
    #     if not for_tokenizer:
    #         if self.data_keys is not None:
    #             dp = data.DictUnzipper(dp, self.data_keys)
    #         encode_fn = partial(data.encode_fn, tokenizer=self.tokenizer)
    #         dp = data.Encoder(dp, encode_fn, self.data_keys)
    #         dp = data.Padder(dp, self.special_ids, self.data_keys)
    #         dp = data.Torcher(dp, self.data_keys)    
    #     # Return the final pipeline
    #     return dp

    # EXAMPLE 1 OF HOW TO USE A CACHE    
    # # - csv.tar
    # # | - 0.csv
    # # | - 1.csv
    # # | - 2.csv
    # archive_dp = IterableWrapper([archive_file_path])
    # def _gen_filepath_fn(archive_path): # Generator function
    #     for i in range(3):
    #         yield os.path.join(os.path.dirname(archive_path), "csv", "{}.csv".format(i))
    # file_cache_dp = OnDiskCacheHolder(archive_dp, filepath_fn=_gen_filepath_fn)
    # file_cache_dp = FileLoader(file_cache_dp, mode="rb")
    # file_cache_dp = TarArchiveReader(file_cache_dp)
    # file_cache_dp = file_cache_dp.map(fn=lambda x: x.read().decode(), input_col=1)
    # def _csv_filepath_fn(csv_path):
    #     return os.path.join(os.path.dirname(os.path.dirname(csv_path)), "csv", os.path.basename(csv_path))
    # # Text mode and skip_read as the data is read and decoded
    # file_cache_dp = EndOnDiskCacheHolder(file_cache_dp, mode="w", filepath_fn=_csv_filepath_fn, skip_read=True)

    # EXAMPLE 2 OF HOW TO USE A CACHE
    # temp_dir = tempfile.TemporaryDirectory()
    # tar_file_dp = IterableWrapper([tar_file_url])
    # def _filepath_fn(url):
    #     filename = os.path.basename(url)
    #     return os.path.join(temp_dir.name, filename)
    # tar_hash_dict = {"xxxx": "yyyy"}
    # tar_cache_dp = tar_file_dp.on_disk_cache(filepath_fn=_filepath_fn, hash_dict=tar_hash_dict, hash_type="md5")
    # # Option 1
    # # Add map function to transform url to file path
    # # tar_cache_dp = HttpReader(tar_cache_dp).map(fn=_filepath_fn, input_col=0)
    # # tar_cache_dp = tar_cache_dp.end_caching(mode="wb")
    # # Option2 use `same_filepath_fn`
    # tar_cache_dp = HttpReader(tar_cache_dp).end_caching(mode="wb", same_filepath_fn=True)

    def pipeline(self, split, shuffle=False, for_tokenizer=False):
        """ Data pipeline for a sequence dataset. The pipeline extracts data
            from a given file, batches it efficiently and tokenizes samples.
        Args:
            split (str): split of the data to take ('train', 'val', 'test')
            shuffle (bool): whether batches are shuffled
            for_tokenizer (bool): whether the pipeline is used to train a
                tokenizer or a whole model.
        Returns:
            iterable: the pipeline that goes from a file to batched tensors.
        """
        # Extract samples from the selected file of the data directory
        dp = FileLister(os.path.join(self.data_dir, self.data_subdir))
        dp = Filter(dp, lambda t: split in t)
        # dp = OnDiskCacheHolder(dp, filepath_fn=data.path_fn)
        dp = FileOpener(dp)
        dp = LineReader(dp)
        dp = data.JsonFileParser(dp)
        dp = data.DynamicBucketBatcher(dp, max_tokens=1e5)
        
        # Tokenization and batching (grouped by similar lengths, dynamic batch size)
        if not for_tokenizer:
            dp = data.Encoder(dp, self.tokenizer, self.data_keys)
            # dp = EndOnDiskCacheHolder(dp, same_filepath_fn=True)
            dp = UnBatcher(dp)
            dp = data.DynamicBucketBatcher(dp, max_tokens=self.max_tokens)
            
        # Padding, shuffling and sending to tensors
        if not for_tokenizer:
            if len(self.data_keys ) > 0:
                dp = data.DictUnzipper(dp, self.data_keys)
            dp = data.Padder(dp, self.special_ids, self.max_len, self.data_keys)
            if shuffle: dp = Shuffler(dp)
            dp = data.Torcher(dp, self.data_keys)
        
        # Return the final pipeline
        return dp

    def get_tokenizer(self, load_tokenizer=False):
        """ Load a tokenizer from file, or train it from scratch and save it.
        Args:
            load_tokenizer (bool): whether to use a trained tokenizer.
        Returns:
            tokenizers.Tokenizer: trained or loaded tokenizer
        """
        tokenizer_path = os.path.join(self.data_dir, 'tokenizer.json')
        if load_tokenizer:
            print(f'Loading tokenizer from {tokenizer_path}')
            try:
                return Tokenizer.from_file(tokenizer_path)
            except Exception:
                print(f'Tokenizer not found at {tokenizer_path}. \
                        Set load_tokenizer to false in config.toml.')
        else:
            data_full_path = os.path.join(self.data_dir, self.data_subdir)
            print(f'Training tokenizer using training set in {data_full_path}')
            pipeline = self.pipeline('train', for_tokenizer=True)
            tokenizer = self.train_tokenizer(pipeline)
            tokenizer.save(tokenizer_path)
            print(f'Saved trained tokenizer at {tokenizer_path}')
            return tokenizer
    
    def train_tokenizer(self, dataset):
        """ Create a trained tokenizer with a given dataset
        Args:
            dataset (iter): batched collection of sequences (str dicts or strs)
        Returns:
            tokenizers.Tokenizer: trained tokenizer
        """
        def training_corpus():
            for batch in dataset:
                if type(batch[0]) is dict:
                    yield [' . '.join(s.values()) for s in batch]
                else:
                    yield batch
        tokenizer = Tokenizer(
            WordLevel(unk_token=self.special_tokens['unk']))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=list(self.special_tokens.values()))
        tokenizer.train_from_iterator(training_corpus(), trainer)
        return tokenizer
