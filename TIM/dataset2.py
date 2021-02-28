from torch.utils.data import Dataset
import tqdm
import torch
import random


class BERTDataset(Dataset):
    # Thanks https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/dataset.py
    def __init__(self, corpus_path, vocab, seq_len, tokenize, params, encoding="utf-8", corpus_lines=None, 
                    on_memory=True, n_samples = None, shuffle=False):
        self.vocab = vocab
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.n_samples = n_samples

        self.tokenize = tokenize
        self.max_pred = params.max_pred
        self.mask_prob = params.mask_prob
        self.word_mask_keep_rand = params.word_mask_keep_rand
        self.word_mask_keep_rand = [float(i) for i in self.word_mask_keep_rand.split(",")]
        assert len(self.word_mask_keep_rand) == 3
        assert sum(self.word_mask_keep_rand) == 1.

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.lines = [line[:-1].split("\t")
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                if shuffle:
                    random.shuffle(self.lines)
                self.lines = self.lines[:self.n_samples]
                self.corpus_lines = len(self.lines)

        if not on_memory:
            # todo : shuffle and n_samples
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__1(self, item):
        t1, t2, is_next_label = self.random_sent(item)
        t1_random, t1_label = self.random_word1(self.tokenize(t1))
        t2_random, t2_label = self.random_word1(self.tokenize(t2))

        # [CLS] A [SEP] B [SEP]
        t1 = [self.vocab.cls_index] + t1_random + [self.vocab.sep_index]
        t2 = t2_random + [self.vocab.sep_index]

        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        t2_label = t2_label + [self.vocab.pad_index]

        #
        segment_label = ([0 for _ in range(len(t1))] + [1 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word1(self, tokens, mask_prob = 0.15):
        output_label = []

        for i, token in enumerate(tokens):
            if token != self.vocab.cls_index and token != self.vocab.sep_index :
                prob = random.random()
                if prob < mask_prob:
                    prob /= mask_prob # 0 < prob < mask_prob <==> 0 < prob/mask_prob < 1

                    # 80% randomly change token to mask token
                    if prob < self.word_mask_keep_rand[0]: # 0.8 for ex.
                        tokens[i] = self.vocab.mask_index

                    # 10% randomly change token to random token
                    elif prob < self.word_mask_keep_rand[0] + self.word_mask_keep_rand[1]: # 09 = 0.8 + 0.1 for ex.
                        tokens[i] = random.randrange(len(self.vocab))

                    # 10% randomly change token to current token
                    else:
                        tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                    output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                    output_label.append(0)

        return tokens, output_label

    def __getitem__(self, item):
        t1, t2, is_next_label = self.random_sent(item)

        # [CLS] A [SEP] B [SEP]
        t1 = [self.vocab.cls_index] + self.tokenize(t1) + [self.vocab.sep_index]
        t2 = self.tokenize(t2) + [self.vocab.sep_index]
        t = t1 + t2
        t = t[:self.seq_len]
        
        # the number of prediction is sometimes less than max_pred when sequence is short
        lt = len(t) - 3 # exclude [CLS], [SEP] and [SEP]
        n_pred = min(self.max_pred, max(1, int(round(lt*self.mask_prob))))
        mask_prob = n_pred/lt 

        input_ids, masked_ids, masked_pos, n_pred = self.random_word(t, mask_prob = mask_prob)
        segment_ids = ([0 for _ in range(len(t1))] + [1 for _ in range(len(t2))])[:self.seq_len]
        input_mask = [1]*len(input_ids)
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1]*len(masked_ids)

        # Zero Padding
        n_pad = self.seq_len - len(input_ids)
        padding = [self.vocab.pad_index for _ in range(n_pad)]
        input_ids.extend(padding)
        segment_ids.extend(padding)
        input_mask.extend(padding)

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            padding = [self.vocab.pad_index for _ in range(n_pad)]
            masked_ids.extend(padding)
            masked_pos.extend(padding)
            masked_weights.extend(padding)

        # input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next
        #return input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next_label
        return torch.tensor(input_ids), torch.tensor(segment_ids), torch.tensor(input_mask), \
               torch.tensor(masked_ids), torch.tensor(masked_pos), torch.tensor(masked_weights), \
               torch.tensor(is_next_label)

    def random_word(self, tokens, mask_prob = 0.15):
        masked_ids = []
        masked_pos  = []
        n_pred = 0

        for i, token in enumerate(tokens):
            if token != self.vocab.cls_index and token != self.vocab.sep_index :
                prob = random.random()
                if prob < mask_prob:
                    prob /= mask_prob # 0 < prob < mask_prob <==> 0 < prob/mask_prob < 1
                    
                    masked_ids.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                    masked_pos.append(i)
                    n_pred += 1

                    # 80% randomly change token to mask token
                    if prob < self.word_mask_keep_rand[0]: # 0.8 for ex.
                        tokens[i] = self.vocab.mask_index

                    # 10% randomly change token to random token
                    elif prob < self.word_mask_keep_rand[0] + self.word_mask_keep_rand[1]: # 09 = 0.8 + 0.1 for ex.
                        tokens[i] = random.randrange(len(self.vocab))

                    # 10% randomly change token to current token
                    else:
                        tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
     
        return tokens, masked_ids, masked_pos, n_pred

    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item][0], self.lines[item][1]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            t1, t2 = line[:-1].split("\t")
            return t1, t2

    def get_random_line(self):
        if self.on_memory:
            return self.lines[random.randrange(len(self.lines))][1]

        line = self.file.__next__()
        if line is None:
            self.file.close()
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()
        return line[:-1].split("\t")[1]

    def __iter__(self): # iterator to load data

        raise NotImplementedError("")
        #self.batch_size = 4
        #self.n_samples = 10

        assert self.batch_size
        self.batch_size = len(self) if self.batch_size > len(self) else self.batch_size
        n_samples = len(self) if self.n_samples > len(self) else self.n_samples
        
        i = 0
        while self.n_samples > i :
            i += self.batch_size
            yield [self[i] for i in range(i-self.batch_size, i)]
      