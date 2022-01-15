# Copyright (c) 2021-present, Pascal Tikeng, MILA.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from pandas.io.parsers import ParserError
import random
import os
import copy
import re
import collections

from .utils import to_bpe_py, to_bpe_cli, get_data_path, path_leaf, init_pretrained_word_embedding
# bias corpus
from .utils import special_tokens

from ..data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD, SPECIAL_WORD, SPECIAL_WORDS

def not_exclude(x, special_tokens, do_upper = False):
    return x and "%s\n"%x not in special_tokens and '<%s>\n'%(x.upper() if do_upper else x) not in special_tokens

def split_sentence(sentence, special_tokens, do_lower = False):
    wl = True
    if wl :
        t = re.split(r'\W+', sentence.strip())
    else :
        t = re.split(r'', sentence.strip())
    return (x.lower() if do_lower else x for x in t if not_exclude(x, special_tokens, do_lower))

def generate_vocabulary(train_captions, special_tokens, min_freq = 0, do_lower = False):
    """
    https://stackoverflow.com/questions/60681953/how-to-create-a-vocabulary-from-a-list-of-strings-in-a-fast-manner-in-python
    """
    word_count = collections.Counter()
    for current_sentence in train_captions:
        word_count.update(split_sentence(str(current_sentence), special_tokens, do_lower))
    # sort : https://stackoverflow.com/a/20950686/11814682
    word_count = dict(word_count.most_common()) # sort :
    
    # build the vocabulary
    vocab = [t.strip() for t in special_tokens] + [key for key, value in word_count.items() if value >= min_freq]
    
    return vocab, word_count

def get_dico(corpus, special_tokens=[], min_freq=0, max_vocab = None, logger = None) :
    log = logger.info if logger is not None else print
    st = [BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD]
    for t in special_tokens :
        if t not in st:
            st.append(t)    
    rest = len(st)
    word2id = {t : i for i, t in enumerate(st)}
    for i in range(SPECIAL_WORDS):
        word2id[SPECIAL_WORD % i] = rest + i
    counts = {k: 0 for k in word2id.keys()}
    vocab, word_count = generate_vocabulary(corpus, st, min_freq, do_lower = False)
    vocab = vocab[:max_vocab]
    skipped = 0
    for i, w in enumerate(vocab) :
        if '\u2028' in w:
            skipped += 1
            continue
        if w in word2id:
            skipped += 1
            log('%s already in vocab' % w)
            continue
        word2id[w] = rest + SPECIAL_WORDS + i - skipped # shift because of extra words
        counts[w] = word_count.get(w, 0)
    
    id2word = {v: k for k, v in word2id.items()}
    dico = Dictionary(id2word, word2id, counts, rest = rest)
    log("Read %i words from the vocabulary file." % len(dico))
    if skipped > 0:
        if logger is not None :
            logger.warning("Skipped %i words!" % skipped)
        else :
            print("Skipped %i words!" % skipped)
    return dico
    
def remove_extreme_label(label, v = 1) :
    if v == 1 :
        if label == 0 or label == 5:
            label = None
        else :
            label = label - 1
    if v == 2 :
        if label == 0 :
            label = 1
        elif label == 5 :
            label = 4
        label = label - 1

    return label

def filter(b, c, v=None) :
    if v is None :
        return b, c
    assert v in [1, 2]
    b = [remove_extreme_label(l, v) for l in b]
    if v==1 :
        c = [x for i, x in enumerate(c) if b[i] is not None]
        b = [x for x in b if x is not None]
    return b, c

class BiasClassificationDataset(Dataset):
    """ Dataset class for Bias Classification"""
    def __init__(self, file, split, params, model, logger, n_samples = None, min_len=1):
        super().__init__()
        assert params.version in [1, 2, 3, 4, 5, 6, 7]
        assert split in ["train", "valid", "test"]
        
        # For large data, it is necessary to process them only once
        data_path = get_data_path(params, file, n_samples, split)
        if os.path.isfile(data_path) :
            logger.info("Loading data from %s ..."%data_path)
            loaded_self = torch.load(data_path)
            for attr_name in dir(loaded_self) :
                try :
                    setattr(self, attr_name, getattr(loaded_self, attr_name))
                except AttributeError :
                    pass
            return
        
        logger.info("Loading data from %s ..."%file)
        
        if split == "train" :
            if os.path.isfile(params.data_info_path) :
                data_info = torch.load(params.data_info_path)
            else :
                data_info = {}
        else :
            data_info = torch.load(params.data_info_path)
            
        self.params = params
        self.to_tensor = model.to_tensor
        self.n_samples = n_samples
        self.shuffle = params.shuffle if split == "train" else False
        self.group_by_size = params.group_by_size
        self.version = params.version
        self.in_memory = params.in_memory
        self.threshold = params.threshold
        self.do_augment = params.do_augment
        self.do_downsampling = params.do_downsampling
        self.do_upsampling = params.do_upsampling
        assert not (self.do_downsampling and self.do_upsampling)
        
        if params.data_columns == "" :
            # assume is bias classification
            num_workers = 3
            self.text_column = "content"
            self.scores_columns = ['answerForQ1.Worker%d'%(k+1) for k in range(num_workers)]
            self.confidence_columns = ['answerForQ2.Worker%d'%(k+1) for k in range(num_workers)]
        else :
            # "content,scores_columns1-scores_columns2...,confidence_columns1-confidence_columns2..."
            """
            For text classification tasks other than bias classification. 
            Just make sure that the scores_columns matches the label, 
            and choose version 1 or 4 (4 is more stable and allows the model to converge quickly, we control the loss function)
            
            For example for stackoverflow tags classification task : data_columns='post,tags'
            """
            data_columns = params.data_columns.split(",")
            assert len(data_columns) >= 2
            self.text_column = data_columns[0]
            self.scores_columns = data_columns[1].split("-")
            if len(data_columns) > 2 :
                self.confidence_columns = data_columns[2].split("-")
                assert len(self.scores_columns) == len(self.confidence_columns)
            else :
                assert len(self.scores_columns) == 1
                self.confidence_columns = []
                
        logger.info("Get instances...")
        try :
            data = [inst for inst in self.get_instances(pd.read_csv(file))]
        except ParserError : # https://stackoverflow.com/questions/33998740/error-in-reading-a-csv-file-in-pandascparsererror-error-tokenizing-data-c-err
            data = [inst for inst in self.get_instances(pd.read_csv(file, lineterminator='\n'))]
        
        # remove short sentence
        l = len(data)
        data = [inst for inst in data if len(inst[0].split(" ")) >= min_len]
        logger.info('Remove %d sentences of length < %d' % (l - len(data), min_len))
        
        sentences = [inst[0] for inst in data]
        
        # lower
        logger.info("Do lower...")
        sentences = [s.lower() for s in sentences]
        
        if not params.google_bert :
            if not params.use_pretrained_word_embedding or params.train_embedding :
                if params.model_name == "XLM" or os.path.isfile(params.codes) :
                    # bpe-ize sentences
                    logger.info("bpe-ize sentences...")
                    #sentences = to_bpe_cli(sentences, codes=params.codes, logger = logger, vocab = params.vocab)
                    sentences = to_bpe_py(sentences, codes=params.codes, vocab = params.vocab)
        
        # check how many tokens are OOV
        if params.google_bert :
            corpus = model.tokenizer.tokenize(' '.join(sentences))
            n_w = len([w for w in corpus])
            n_oov = len([w for w in corpus if w not in model.tokenizer.vocab]) 
        else :
            if not params.use_pretrained_word_embedding :
                # XLM / RNN / LSTM / CNN
                corpus = ' '.join(sentences).split()
                n_w = len([w for w in corpus])
                if params.model_name == "XLM" or os.path.isfile(params.vocab) :
                    n_oov = len([w for w in corpus if w not in model.dico.word2id])
                else :
                    if split == "train" :
                        if "dico" in data_info :
                            dico = data_info["dico"]
                        else :
                            logger.info("Build word level vocabulary ...")
                            dico = get_dico(corpus = sentences, special_tokens=special_tokens, min_freq=0, logger = logger)
                            data_info["dico"] = dico
                        model.__init__(model.n_labels, params, logger, dico = dico)
                        model.self_to(params.device)
                    else :
                        dico = data_info["dico"]     
                    n_oov = len([w for w in corpus if w not in dico.word2id])
            else :
                # RNN / LSTM / CNN
                if not hasattr(model, "vocab") and split == "train":
                    if "corpus" in data_info :
                        corpus = data_info["corpus"]
                    else :
                        corpus = [s.split(" ") for s in sentences]
                        #self.corpus = corpus
                        data_info["corpus"] = corpus
                    
                    init_pretrained_word_embedding(model, sentences = corpus, params = params, logger = logger)

                if hasattr(model, "vocab") :
                    corpus = model.tokenize(' '.join(sentences))
                    n_w = len([w for w in corpus])
                    n_oov = len([w for w in corpus if w not in model.vocab.stoi]) 
                else :
                    # TODO
                    n_w = float('nan')
                    n_oov = float('nan')
                    
        p = n_oov/(n_w+1e-12)
        logger.info('Number of out-of-vocab words: %s/%s = %s %s' % (n_oov, n_w, p*100, "%"))
        
        for i in range(len(data)) :
            data[i][0] = sentences[i]
        
        if split == "train" :
            if self.do_augment  :
                # TODO : raise this hard code to the parameter level
                p = 0.3
                max_change = 5
                logger.info("EDA text augmentation : p = %s, max_change = %s..."%(p, max_change))
                data, self.weights = self.augment(data, p = p, max_change = max_change) 
            if self.do_downsampling :
                logger.info("Downsampling ...")
                data, self.weights = self.downsampling(data)
            if self.do_upsampling  :
                logger.info("Upsampling ...")
                data, self.weights = self.upsampling(data)
            
        logger.info("Weigths %s"%str(self.weights))
        if self.params.weighted_training :
            weights = [w + 1e-12 for w in self.weights]
            weights = torch.FloatTensor([1.0 / w for w in weights])
            weights = weights / weights.sum()
            self.weights = weights.to(self.params.device)
            logger.info("Normalized weigths %s"%str(self.weights.tolist()))
        else :
            self.weights = None

        self.n_samples = len(data)
        self.batch_size = self.n_samples if self.params.batch_size > self.n_samples else self.params.batch_size
        
        if self.in_memory :
            logger.info("In memory ...")
            i = 0
            tmp = []
            while self.n_samples > i :
                i += self.batch_size
                inst = list(zip(*data[i-self.batch_size:i]))
                langs_ids = self.langs_ids[i-self.batch_size:i]
                tmp.append(tuple([self.to_tensor(inst[0], lang_id = langs_ids)] + [torch.stack(y) for y in inst[1:]]))
            self.data = tmp
        else :
            self.test = True
            if self.test :
                self.data = data
            else :
                for i in range(len(data)) :
                    data[i][0] = self.to_tensor(sentences[i]) 
                self.data = data
        
        # For large data, it is necessary to process them only once
        logger.info("Saving data to %s ..."%data_path)
        torch.save(self, data_path)
        if split == "train" :
            torch.save(data_info, params.data_info_path)
    
    def reset(self, data) :
        self.data = data
        if self.batch_size == self.n_samples :
            self.n_samples = len(data)
        else :
            self.n_samples = len(data) * (self.batch_size if self.in_memory else 1)
        self.batch_size = self.n_samples if self.params.batch_size > self.n_samples else self.params.batch_size

    def __len__(self):
        if self.in_memory :
            return self.n_samples // self.batch_size
        else :
            return self.n_samples

    def __getitem__(self, index):
        if not self.in_memory :
            inst = self.data[index]
            if self.test :
                langs_id = self.langs_ids[index]
                return tuple([self.to_tensor(inst[0], lang_id = langs_id)] + [torch.stack(y) for y in inst[1:]])
            else :
                return tuple([inst[0]] + [torch.stack(y) for y in inst[1:]])
        else :
            return self.data[index]
        
    def __iter__(self): # iterator to load data
        if self.shuffle :
            random.shuffle(self.data)
        if not self.in_memory :
            i = 0
            while self.n_samples > i :
                i += self.batch_size
                inst = list(zip(*self.data[i-self.batch_size:i]))
                if self.test :
                    try :
                        langs_ids = self.langs_ids[i-self.batch_size:i]
                        yield tuple([self.to_tensor(inst[0], lang_id = langs_ids)] + [torch.stack(y) for y in inst[1:]])
                    except IndexError : # list index out of range
                        pass
                else :
                    inst1 = list(zip(*inst[0]))
                    yield tuple([(torch.stack(x) for x in inst1)] + [torch.stack(y) for y in inst[1:]])

        else :
            for batch in self.data :
                yield batch
                
    def get_instances(self, df):
        #columns = list(df.columns[1:]) # excapt "'Unnamed: 0'"
        rows = df.iterrows()
        if self.shuffle :
            if self.n_samples or (not self.group_by_size and not self.n_samples) :
                rows = list(rows)
                random.shuffle(rows)
        if self.n_samples :
            rows = list(rows)[:self.n_samples]
        if self.group_by_size :
            rows = sorted(rows, key = lambda x : len(x[1][self.text_column].split()), reverse=False)
        
        weights = [0] * self.params.n_labels
        self.langs_ids = [] #if params.n_langs > 1 else []
        
        for row in rows : 
            row = row[1]
            text = row[self.text_column]
            b = [row[col] for col in self.scores_columns]
            c = [row[col] for col in self.confidence_columns] if self.confidence_columns else [1] # Give a score of 1 to the label
            b, c = filter(b, c, v = None)
            s = sum(c)
            weight_out = torch.tensor(s / 30, dtype=torch.float)
            s = 1 if s == 0 else s
            #b = [i+1 for i in b]
            if self.version == 1 :
                #  label the average of the scores with the confidence scores as weighting
                label = int(round(sum([ score * conf for score, conf in  zip(b, c) ]) / s))
                weights[label] = weights[label] + 1 
                self.langs_ids.append(label)
                label = torch.tensor(label, dtype=torch.long)
                yield [text, label, label, weight_out]
            
            elif self.version in [2, 3, 4] : 
                p_c = [0]*self.params.n_labels
                for (b_i, c_i) in zip(b, c) :
                    p_c[b_i] += c_i/s
                    
                if self.version == 4 :
                    #  label the average of the scores with the confidence scores as weighting
                    label = int(round(sum([ score * conf for score, conf in  zip(b, c) ]) / s))
                else :
                    # the class with the maximum confidence score was used as the target
                    label = b[np.argmax(a = c)] 
                    
                weights[label] = weights[label] + 1
                self.langs_ids.append(label)
                yield [text, torch.tensor(p_c, dtype=torch.float), torch.tensor(label, dtype=torch.long), weight_out]
            
            elif self.version == 5 :
                # bias regression
                b = sum(b) / len(b)
                c = sum(c) / len(c)
                label = int(b >= self.threshold) # 1 if b >= self.threshold else 0
                weights[label] = weights[label] + 1
                self.langs_ids.append(label)
                yield [text, torch.tensor([b, c], dtype=torch.float), torch.tensor(label, dtype=torch.long), weight_out]
            
            elif self.version == 6:
                # TODO
                label = int(round(sum([ score * conf for score, conf in  zip(b, c) ]) / s))
                weights[label] = weights[label] + 1
                self.langs_ids.append(label)
                yield [text, torch.tensor([b, c], dtype=torch.long), torch.tensor(label, dtype=torch.long), weight_out]
            
            elif self.version == 7 :
                # TODO
                #assert 0 < self.threshold < 1
                #label = int(round(sum([ score * conf for score, conf in  zip(b, c) ]) / s))
                label = sum([ score * conf for score, conf in  zip(b, c) ]) / s
                label = int(label >= self.threshold) # 1 if label >= self.threshold else 0
                weights[label] = weights[label] + 1
                self.langs_ids.append(label)
                yield [text, torch.tensor(label, dtype=torch.float), torch.tensor(label, dtype=torch.long), weight_out]
            
        self.weights = weights

    def downsampling(self, data):
        tmp = []
        count = [0]*self.params.n_labels
        # mean or min
        #n_max = sum(self.weights) / len(self.weights)
        n_max = min(self.weights)
        random.shuffle(data)
        for x in data :
            a = x[-1].item() 
            if n_max > count[a] : # and a != 3 :
                count[a] += 1
                tmp.append(x)
            """
            elif a == 3 :
                if 100 > count[a] :
                    count[a] += 1
                    tmp.append(x)
            """
        return tmp, count
    
    def upsampling(self, data) :
        tmp = []
        count = copy.deepcopy(self.weights)
        # mean or max
        #n_min = sum(self.weights) / len(self.weights)
        n_min = max(self.weights)
        for c, _ in enumerate(self.weights) :
            data4c = [inst for inst in data if inst[-1].item() == c]
            i, m = 0, len(data4c)
            if m != 0 :
                while n_min > count[c] :
                    tmp.append(data4c[i % m])
                    i += 1
                    count[c] += 1 
        random.shuffle(tmp)
        return data + tmp, count
    
    def augment(self, data, p = 0.3, max_change = 5) :
        """https://github.com/dsfsi/textaugment
        Do this before :
        >>> ! pip install textaugment
        >>> import nltk
        >>> nltk.download('stopwords')
        >>> nltk.download('wordnet')
        """
        from textaugment import EDA
        t = EDA()
        tmp = []
        count = copy.deepcopy(self.weights)
        for inst in data :
            s = inst[0]
            label = inst[-1].item()
            l = len(s.split(" "))
            n = min(max_change, max(1, int(round(l*p))))

            aug = [
                t.synonym_replacement(s, n = n),
                t.random_deletion(s, p = p),
                t.random_swap(s, n = n),
                t.random_insertion(s, n = n)
            ]
            for s_aug in aug :
                if type(s_aug) == str and s_aug != "" and s_aug != s :
                    inst_aug = copy.deepcopy(inst)
                    inst_aug[0] = s_aug
                    tmp.append(inst_aug)
                    count[label] += 1
                    
        random.shuffle(tmp)
        return data + tmp, count