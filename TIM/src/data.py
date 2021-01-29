# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unicodedata

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

import io
import re

# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    #w = 'SOS' + w + ' EOS'
    return w

def check_data_params(params):
    """
    Check datasets parameters.
    """
    pass

def load_data(params):
    """
    Load data.
    The returned dictionary contains:
        - dico (dictionary)
        - vocab (FloatTensor)
        - train / valid / test (monolingual and/or parallel datasets)
    """
    data = {}
    # todo

    return data

class TranslationDataset(Dataset):
  
    def __init__(self, data, lang1_tokenizer, lang2_tokenizer):
        
        self.data = data
        self.lang1_tokenizer = lang1_tokenizer
        self.lang2_tokenizer = lang2_tokenizer

        self.PAD_IDX = 0
        self.BOS_IDX_1 = lang1_tokenizer.vocab_size
        self.EOS_IDX_1 = lang1_tokenizer.vocab_size + 1
        self.BOS_IDX_2 = lang2_tokenizer.vocab_size
        self.EOS_IDX_2 = lang2_tokenizer.vocab_size + 1


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        lang1, lang2 = self.data[idx]
        lang1, lang2 = preprocess_sentence(lang1), preprocess_sentence(lang2)  
        lang1, lang2 = self.lang1_tokenizer.encode(lang1), self.lang2_tokenizer.encode(lang2)  
        
        return torch.tensor(lang1), torch.tensor(lang2)

    def generate_batch(self, data_batch):
        lang1_batch, lang2_batch = [], []
        for (lang1_item, lang2_item) in data_batch:
            lang1_batch.append(torch.cat([torch.tensor([self.BOS_IDX_1]), lang1_item, torch.tensor([self.EOS_IDX_1])], dim=0))
            lang2_batch.append(torch.cat([torch.tensor([self.BOS_IDX_2]), lang2_item, torch.tensor([self.EOS_IDX_2])], dim=0))
        lang1_batch = pad_sequence(lang1_batch, padding_value=self.PAD_IDX)
        lang2_batch = pad_sequence(lang2_batch, padding_value=self.PAD_IDX)
        return lang1_batch, lang2_batch
