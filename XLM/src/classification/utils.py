import hashlib
import os
import ntpath

def to_bpe_cli(sentences, codes : str, vocab : str = "", fastbpe = os.path.join(os.getcwd(), 'tools/fastBPE/fast'), logger = None):
    """Below is one way to bpe-ize sentences
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
    """Below is one way to bpe-ize sentences
    Sentences have to be in the BPE format, i.e. tokenized sentences on which you applied fastBPE.
    
    sentences : list of sentence to bpe-ize
    codes : path to the codes of the model
    vocab (optional) : path to the vocab of the model
    
    installation : pip install fastbpe"""
    import fastBPE
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
    f = '%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s'%(
        params.version, params.n_labels, params.google_bert, params.weighted_training, params.batch_size, n_samples, 
        params.data_columns, params.in_memory, params.do_augment, params.do_downsampling, params.do_upsampling,
        split, params.threshold
    )
    if params.google_bert :
        f += "_%s"%params.bert_model_name
    else : 
        f += "_%s"%hash_file(params.codes)
        if os.path.isfile(params.vocab) :
            f += "_%s"%hash_file(params.vocab)

    filename = "%s_%s"%(filename, hash_var(f))
    data_path = os.path.join(params.dump_path, '%s.pth'%filename)
    return data_path