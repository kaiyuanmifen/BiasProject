# Copyright (c) 2021-present, Pascal Tikeng, MILA.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

import hashlib
import os
import ntpath
import subprocess

#import fastBPE

from ..data.dictionary import PAD_WORD, UNK_WORD
from ..model.transformer import Embedding

# bias corpus
special_tokens = ["<url>", "<email>", "<phone>", "<number>", "<digit>", "<cur>"]

def to_bpe_cli(sentences, codes : str, vocab : str = "", fastbpe = os.path.join(os.getcwd(), 'tools/fastBPE/fast'), logger = None):
    """
    Below is one way to bpe-ize sentences
    Sentences have to be in the BPE format, i.e. tokenized sentences on which you applied fastBPE.
    https://github.com/facebookresearch/XLM/blob/master/generate-embeddings.ipynb
    
    sentences : list of sentence to bpe-ize
    codes : path to the codes of the model
    vocab (optional) : path to the vocab of the model
    fastbpe : path to fastbpe
        installation : git clone https://github.com/glample/fastBPE tools/fastBPE && cd tools/fastBPE && g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
    logger (optional) : logger object
    """
    # write sentences to tmp file
    tmp_file1 = './tmp_sentences.txt'
    tmp_file2 = './tmp_sentences.bpe'
    with open(tmp_file1, 'w') as fwrite:
        for sent in sentences:
            fwrite.write(sent + '\n')
    
    # apply bpe to tmp file
    os.system('%s applybpe %s %s %s %s' % (fastbpe, tmp_file2, tmp_file1, codes, vocab))
    
    # load bpe-ized sentences
    sentences_bpe = []
    with open(tmp_file2) as f:
        for line in f:
            sentences_bpe.append(line.rstrip())
    
    if logger is not None :
        logger.info("Delete %s and %s"%(tmp_file1, tmp_file2))
    else :
        print("Delete %s and %s"%(tmp_file1, tmp_file2))
        
    os.remove(tmp_file1)
    os.remove(tmp_file2)
    
    return sentences_bpe

def to_bpe_py(sentences, codes : str,  vocab : str = ""):
    """
    Below is one way to bpe-ize sentences
    Sentences have to be in the BPE format, i.e. tokenized sentences on which you applied fastBPE.
    
    sentences : list of sentence to bpe-ize
    codes : path to the codes of the model
    vocab (optional) : path to the vocab of the model
    
    installation : pip install fastbpe
    """
    return sentences
    import fastBPE
    #if not os.path.isfile(vocab) :
    #    vocab = ""
    return fastBPE.fastBPE(codes, vocab).apply(sentences)

def path_leaf(path):
    # https://stackoverflow.com/a/8384788/11814682
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_hash_object(type_='sha-1'):
    """make a hash object"""
    assert type_ in ["sha-1", "sha-256", "md5"]
    if type_ == 'sha-1' :
        h = hashlib.sha1()
    elif type_ == "sha-256":
        h = hashlib.sha256()
    elif type_ == "md5" :
        h = hashlib.md5()
    return h

def hash_file(file_path, BLOCK_SIZE = 65536, type_='sha-1'):
    """This function returns the SHA-1/SHA-256/md5 hash of the file passed into it
    #  BLOCK_SIZE : the size of each read from the file

    https://www.programiz.com/python-programming/examples/hash-file
    https://nitratine.net/blog/post/how-to-hash-files-in-python/
    """
    assert os.path.isfile(file_path)
    # make a hash object
    h = get_hash_object(type_)
    # open file for reading in binary mode
    with open(file_path,'rb') as file:
        # loop till the end of the file
        chunk = 0
        while chunk != b'':
            # read only BLOCK_SIZE bytes at a time
            chunk = file.read(BLOCK_SIZE)
            h.update(chunk)
    # return the hex representation of digest #, hash value as a bytes object
    return h.hexdigest() #, h.digest()

def hash_var(var, type_='sha-1'):
    """This function returns the SHA-1/SHA-256/md5 hash of the variable passed into it
    https://nitratine.net/blog/post/how-to-hash-files-in-python/
    https://stackoverflow.com/questions/24905062/how-to-hash-a-variable-in-python"""
    # make a hash object
    h = get_hash_object(type_)
    h.update(var.encode('utf8'))
    # return the hex representation of digest #, hash value as a bytes object
    return h.hexdigest() #, h.digest()

def get_data_path(params, data_file, n_samples, split) :
    filename, _ = os.path.splitext(path_leaf(data_file))
    f = '%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s'%(
        params.version, params.n_labels, params.google_bert, params.weighted_training, params.batch_size, n_samples, 
        params.data_columns, params.in_memory, params.do_augment, params.do_downsampling, params.do_upsampling,
        split, params.threshold, params.model_name
    )
    if params.google_bert :
        f += "_%s"%params.bert_model_name
    else : 
        #if not params.use_pretrained_word_embedding :
        if os.path.isfile(params.codes) :
            f += "_%s"%hash_file(params.codes)
        if os.path.isfile(params.vocab) :
            f += "_%s"%hash_file(params.vocab)

    filename = "%s_%s"%(filename, hash_var(f))
    data_path = os.path.join(params.dump_path, '%s.pth'%filename)
    return data_path

def to_tensor(sentences, pad_index, dico = None, tokenize_and_cut = None, batch_first = False):
    if type(sentences) == str :
        sentences = [sentences]
    else :
        assert type(sentences) in [list, tuple]
    
    assert (dico is None) ^ (tokenize_and_cut is None)
    if dico is not None :
        # trucate ~ self.params.max_len ??
        bs = len(sentences)
        lengths = [len(sent) for sent in sentences]
        slen = max(lengths)
        lengths = torch.LongTensor(lengths)
        word_ids = torch.LongTensor(slen, bs).fill_(pad_index)
        for i in range(bs):
            sent = torch.LongTensor([dico.index(w) for w in sentences[i]])
            word_ids[:len(sent), i] = sent
        langs = None
        if batch_first :
            return word_ids.transpose(0,1), lengths, langs
        else :
            return word_ids, lengths, langs
            
    else :
        sentences = [tokenize_and_cut(s) for s in sentences]
        bs = len(sentences)
        lengths = [len(sent) for sent in sentences]
        slen = max(lengths)
        lengths = torch.LongTensor(lengths)
        word_ids = torch.LongTensor(bs, slen).fill_(pad_index)
        for i in range(bs):
            sent = torch.LongTensor(sentences[i])
            word_ids[i,:len(sent)] = sent
        langs = None
        if batch_first :
            return word_ids, lengths, langs
        else :
            return word_ids.transpose(0,1), lengths, langs
        
def init_embedding_vector(size) :
    return torch.zeros(size)
    
def init_pretrained_word_embedding(model, sentences, params, logger = None) :
    
    # import torchtext.vocab as vocab
    # glove = vocab.GloVe(name='6B', dim=100)
    # embedding_matrix = glove.vectors
    # # embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)
    # embedding_matrix = embedding_matrix.clone().detach().float()
    # model.embedding = nn.Embedding.from_pretrained(embedding_matrix)
    
    import torchtext
    
    vectors = params.use_pretrained_word_embedding 
    logger.info("Loading pretraining embedding : %s ..."%vectors)
    tokenize = params.tokenize
    max_vocab_size = params.max_vocab_size
    pad_index = params.pad_index
    
    # RNN
    TEXT = torchtext.legacy.data.Field(tokenize = tokenize, pad_token=PAD_WORD, unk_token=UNK_WORD)
    # LSTM
    #TEXT = torchtext.legacy.data.Field(tokenize = tokenize, include_lengths = True, pad_token=PAD_WORD, unk_token=UNK_WORD)
    # CNN
    #TEXT = torchtext.legacy.data.Field(tokenize = tokenize, batch_first = True, pad_token=PAD_WORD, unk_token=UNK_WORD)

    TEXT.build_vocab(sentences, max_size = max_vocab_size, vectors =  vectors, unk_init = torch.Tensor.normal_)
    logger.info(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")

    #LABEL = torchtext.legacy.data.LabelField(dtype = torch.float)
    #LABEL.build_vocab(labels)
    #print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

    vocab = TEXT.vocab#.stoi
    pretrained_embeddings = vocab.vectors
    vocab_size = len(vocab)
    embedding_dim = int(vectors.split(".")[-1].split("d")[0])
    
    pad_index = vocab.stoi[PAD_WORD]
    model.pad_index = pad_index
    #params.pad_index = pad_index

    i = max(vocab.stoi.values())
    st = []
    for t in special_tokens :
        if t not in vocab.stoi:
            st.append(t)
    vocab.stoi.update({t : j+i+1 for j, t in enumerate(st)})
    rest = len(st)
    vocab_size = vocab_size + rest     
    #model.tokenize = TEXT.tokenize if params.train_embedding else lambda x : x.split(" ")
    if params.train_embedding :
        st_vectors = init_embedding_vector((rest, embedding_dim)) # torch.zeros(rest, embedding_dim)
    else :
        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
        st_vectors = pretrained_embeddings[UNK_IDX].clone().expand(rest, -1) # rest x embedding_dim
    
    pretrained_embeddings = torch.cat((pretrained_embeddings, st_vectors), dim=0)    
    embedding = Embedding(num_embeddings = vocab_size, embedding_dim = embedding_dim, padding_idx=pad_index)
    embedding.weight.data.copy_(pretrained_embeddings)
    #UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    #embedding.weight.data[UNK_IDX] = init_embedding_vector(embedding_dim) # torch.zeros(embedding_dim)
    #embedding.weight.data[pad_index] = init_embedding_vector(embedding_dim) # torch.zeros(embedding_dim)
    embedding.eval()
    model.embedding = embedding
    model.vocab = vocab
    model.embedding_dim = embedding_dim
    model.init_net(params)
    model = model.to(params.device)
    
    if logger is not None :
        logger.info(model)