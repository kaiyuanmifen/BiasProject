import tensorflow_datasets as tfds

def build_tokenizer(tokenizer_path, corpus : tuple =None, vocab_size = None):
    if os.path.exists(tokenizer_path) :
        tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(tokenizer_path)
    else :
        tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                      corpus, target_vocab_size=vocab_size)
        tokenizer.save_to_file(tokenizer_path)

    
    def tokenize(self, text):
        return [tokenizer.decode([ts]) for ts in self.tokenizer.encode(text)]
    tokenizer.tokenize = tokenize

    def convert_tokens_to_ids(self, tokens):
        return [self.tokenizer.encode(t) for t in tokens]
    tokenizer.convert_tokens_to_ids = convert_tokens_to_ids
    
    tokenizer.convert_to_unicode = lambda self, text : convert_to_unicode(text)

    return tokenizer