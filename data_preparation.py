import os
import wget
from nltk.tokenize import WordPunctTokenizer
import torchtext
print(torchtext.__version__)

from torchtext.legacy.datasets import TranslationDataset, Multi30k
from torchtext.legacy.data import Field, BucketIterator


class DataPreparation:
    def __init__(self, path_to_data=None, split_ratio=None):
        self.path_to_data = path_to_data
        self.split_ratio = split_ratio if split_ratio is not None else [0.8, 0.15, 0.05]
        self.SRC = None
        self.TRG = None
        self.dataset = None

    def _download_data(self):
        self.path_to_data = self.path_to_data if self.path_to_data is not None else '.data/data.txt'
        if not os.path.exists(self.path_to_data):
            print("Dataset not found locally. Downloading from github.")
            url = 'https://raw.githubusercontent.com/neychev/made_nlp_course/master/datasets/Machine_translation_EN_RU/data.txt'
            wget.download(url, self.path_to_data)

    @staticmethod
    def _tokenize(x, tokenizer=WordPunctTokenizer()):
        return tokenizer.tokenize(x.lower())

    def _create_dataset(self):
        self.SRC = Field(tokenize=self._tokenize,
                         init_token='<sos>',
                         eos_token='<eos>',
                         lower=True)

        self.TRG = Field(tokenize=self._tokenize,
                         init_token='<sos>',
                         eos_token='<eos>',
                         lower=True)

        self.dataset = torchtext.legacy.data.TabularDataset(
            path=self.path_to_data,
            format='tsv',
            fields=[('trg', self.TRG), ('src', self.SRC)]
        )

    def _split_data(self):
        train_data, valid_data, test_data = self.dataset.split(split_ratio=self.split_ratio)
        print(f"Number of training examples: {len(train_data.examples)}")
        print(f"Number of validation examples: {len(valid_data.examples)}")
        print(f"Number of testing examples: {len(test_data.examples)}")
        return train_data, valid_data, test_data


    def _create_vocab(self, train_data):
        self.SRC.build_vocab(train_data, min_freq=3)
        self.TRG.build_vocab(train_data, min_freq=3)
        print(f"Unique tokens in source (ru) vocabulary: {len(self.SRC.vocab)}")
        print(f"Unique tokens in target (en) vocabulary: {len(self.TRG.vocab)}")

    def data_pipeline(self):
        print('download data')
        self._download_data()
        print('creating dataset')
        self._create_dataset()
        print('create train, valid and test data')
        train_data, valid_data, test_data = self._split_data()
        print('build vocab')
        self._create_vocab(train_data)
        return train_data, valid_data, test_data

