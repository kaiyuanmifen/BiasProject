from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer, CharBPETokenizer, SentencePieceBPETokenizer
from tokenizers.processors import BertProcessing

import itertools

from tokenization1 import convert_to_unicode

class BertTokenizer :
    def __init__(self, max_len = 512, path=".", name="corpus", train_path = "", vocab_size = 52000,
                 min_frequency=2, special_tokens=[]):
        # Initialize a tokenizer
        # Provided Tokenizers
        #self.tokenizer = ByteLevelBPETokenizer() # The byte level version of the BPE
        self.tokenizer = BertWordPieceTokenizer() # The famous Bert tokenizer, using WordPiece
        #self.tokenizer = CharBPETokenizer() # The original BPE
        #self.tokenizer = SentencePieceBPETokenizer() # A BPE implementation compatible with the one used by SentencePiece


        v = os.path.join(path, name + "-vocab.json")
        m = os.path.join(path, name + "-merges.txt")
        if os.path.isfile(v) and os.path.isfile(m) :
        #if False :
            self.tokenizer.from_file(v, m)
        else :    
            # Customize training
            self.tokenizer.train(train_path, vocab_size, min_frequency, show_progress = True,
                                 special_tokens=special_tokens)
            # Save files to disk
            self.tokenizer.save_model(path, name)

        #self.tokenizer._tokenizer.post_processor = BertProcessing(
        #    (EOS_token, self.tokenizer.token_to_id(EOS_token)),
        #    (SOS_token, self.tokenizer.token_to_id(SOS_token)),
        #)
        self.tokenizer.enable_truncation(max_length=max_len)
        self.vocab = self.tokenizer.get_vocab()

        #t = tokenizer.encode("Bias project. <pad>")
        #t.ids, t.type_ids, t.tokens, t.offsets, t.attention_mask, t.special_tokens_mask, t.overflowing
    
    def tokenize(self, text):
        return self.tokenizer.encode(text).tokens

    def convert_tokens_to_ids(self, tokens):
        #return convert_tokens_to_ids(self.vocab, tokens)
        return list(itertools.chain.from_iterable([self.tokenizer.encode(t).ids for t in tokens]))

    def convert_to_unicode(self, text):
        return convert_to_unicode(text)