import tensorflow_datasets as tfds
import os
import itertools

from tokenization1 import convert_to_unicode

def build_tokenizer(tokenizer_path, corpus : tuple =None, vocab_size = None, special_tokens=[]):
    if os.path.exists(tokenizer_path) :
        tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(tokenizer_path)
    else :
        tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                      corpus, target_vocab_size=vocab_size,
                      max_subword_length=20, max_corpus_chars=None,
                      reserved_tokens=special_tokens)
        tokenizer.save_to_file(tokenizer_path)
    
    def tokenize(text):
        return [tokenizer.decode([ts]) for ts in tokenizer.encode(text)]
    tokenizer.tokenize = tokenize

    def convert_tokens_to_ids(tokens):
        return list(itertools.chain.from_iterable([tokenizer.encode(t) for t in tokens]))
    tokenizer.convert_tokens_to_ids = convert_tokens_to_ids
    
    tokenizer.convert_to_unicode = convert_to_unicode

    return tokenizer