from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer, CharBPETokenizer, SentencePieceBPETokenizer
from tokenizers.processors import BertProcessing

class BertTokenizer :
    def __init__(self, max_len = 512, path=".", name="corpus", train_path = "", vocab_size = 52000,
                 min_frequency=2):
        SOS_token = '<s>'
        EOS_token = '</s>'
        UNK_token = "<unk>"
        PAD_token = "<pad>"
        MASK_token =  "[MASK]" # "<mask>"
        CLS_token = "[CLS]"
        SEP_token = "[SEP]"
        self.special_tokens=[PAD_token, SOS_token, EOS_token, UNK_token, MASK_token, CLS_token, SEP_token]

        # Initialize a tokenizer
        # Provided Tokenizers
        #self.tokenizer = ByteLevelBPETokenizer() # The byte level version of the BPE
        self.tokenizer = BertWordPieceTokenizer() # The famous Bert tokenizer, using WordPiece
        #self.tokenizer = CharBPETokenizer() # The original BPE
        #self.tokenizer = SentencePieceBPETokenizer() # A BPE implementation compatible with the one used by SentencePiece


        v = os.path.join(path, name + "-vocab.json")
        m = os.path.join(path, name + "-merges.txt")
        #if os.path.isfile(v) and os.path.isfile(m) :
        if False :
            self.tokenizer.from_file(v, m)
        else :    
            # Customize training
            self.tokenizer.train(train_path, vocab_size, min_frequency, show_progress = True,
                                 special_tokens=self.special_tokens)
            # Save files to disk
            self.tokenizer.save_model(path, name)

        self.tokenizer._tokenizer.post_processor = BertProcessing(
            (EOS_token, self.tokenizer.token_to_id(EOS_token)),
            (SOS_token, self.tokenizer.token_to_id(SOS_token)),
        )
        self.tokenizer.enable_truncation(max_length=max_len)
        self.vocab = self.tokenizer.get_vocab()

        #t = tokenizer.encode("Bias project. <pad>")
        #t.ids, t.type_ids, t.tokens, t.offsets, t.attention_mask, t.special_tokens_mask, t.overflowing
    
    def tokenize(self, text):
        return self.tokenizer.encode(text).tokens

    def convert_tokens_to_ids(self, tokens):
        #return convert_tokens_to_ids(self.vocab, tokens)
        text = " ".join(tokens)
        return self.tokenizer.encode(text)

    def convert_to_unicode(self, text):
        return convert_to_unicode(text)