import torch
from torch import tensor
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math, copy
import numpy as np
import tqdm
from data import to_var, calc_bleu, id2text_sentence
from utils import bool_flag, add_log, add_output

def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention' """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class PositionalEncoding(nn.Module):
    """Implement the PE function."""
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

################ Encoder ################
class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask, return_intermediate=False):
        """Pass the input (and mask) through each layer in turn."""
        z = []
        for layer in self.layers:
            x = layer(x, mask)
            z.append(x)
        return (self.norm(x), z) if return_intermediate else self.norm(x) 


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


################ Decoder ################
class Decoder(nn.Module):
    """Generic N layer decoder with masking."""
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask, return_intermediate=False):
        z = []
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
            z.append(x)
        return (self.norm(x), z) if return_intermediate else self.norm(x) 

class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """Follow Figure 1 (right) for connections."""
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

################ Generator ################
class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, position_layer, model_size, latent_size):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.position_layer = position_layer
        self.model_size = model_size
        self.latent_size = latent_size
        self.sigmoid = nn.Sigmoid()

        # self.memory2latent = nn.Linear(self.model_size, self.latent_size)
        # self.latent2memory = nn.Linear(self.latent_size, self.model_size)
        self.ae_vs_ar = True # auto_encoder, auto_regressire
        if not self.ae_vs_ar :
            self.decoder = None
            self.tgt_embed = None
    def forward(self, src, tgt, src_mask, src_attn_mask, tgt_mask, return_intermediate=False):
        """
        Take in and process masked src and target sequences.
        """
        if self.ae_vs_ar :
            if return_intermediate :
                latent, z = self.encode(src, src_mask, return_intermediate)  # (batch_size, max_src_seq, d_model)
            else :
                latent = self.encode(src, src_mask)  # (batch_size, max_src_seq, d_model)
        
            latent = self.sigmoid(latent)
            # memory = self.position_layer(memory)

            latent = torch.sum(latent, dim=1)  # (batch_size, d_model)

            # latent = self.memory2latent(memory)  # (batch_size, max_src_seq, latent_size)

            # latent = self.memory2latent(memory)
            # memory = self.latent2memory(latent)  # (batch_size, max_src_seq, d_model)
        
            logit = self.decode(latent.unsqueeze(1), tgt, tgt_mask)  # (batch_size, max_tgt_seq, d_model)
            prob = self.generator(logit)  # (batch_size, max_seq, vocab_size)
            if return_intermediate :
                return latent, prob, z
            else :
                return latent, prob
        else :
            tensor = self.src_embed(tgt)
            tensor *= src_attn_mask.unsqueeze(-1).to(tensor.dtype)
            if return_intermediate :
                tensor, z = self.encoder(tensor, tgt_mask, return_intermediate)
            else :
                tensor, z = self.encoder(tensor, tgt_mask)
            tensor *= src_attn_mask.unsqueeze(-1).to(tensor.dtype)
            #pred_mask = None
            #masked_tensor = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)
            #scores, loss = self.pred_layer(masked_tensor, y, get_scores, reduction=reduction)
            #return scores, loss
            prob = self.generator(tensor)  
            latent = self.sigmoid(tensor)
            latent = torch.sum(latent, dim=1)
            if return_intermediate :
                return latent, prob, z
            else :
                return latent, prob

    def encode(self, src, src_mask, return_intermediate=False):
        return self.encoder(self.src_embed(src), src_mask, return_intermediate)

    def decode(self, memory, tgt, tgt_mask):
        # memory: (batch_size, 1, d_model)
        src_mask = torch.ones(memory.size(0), 1, 1).long().to(memory.device)
        # print("src_mask here", src_mask)
        # print("src_mask", src_mask.size())
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def greedy_decode(self, latent, max_len, start_id):
        '''
        latent: (batch_size, max_src_seq, d_model)
        src_mask: (batch_size, 1, max_src_len)
        '''
        if self.ae_vs_ar :
            batch_size = latent.size(0)

            # memory = self.latent2memory(latent)

            ys = torch.ones(batch_size, 1).fill_(start_id).long().to(latent.device)  # (batch_size, 1)
            for i in range(max_len - 1):
                # input("==========")
                # print("="*10, i)
                # print("ys", ys.size())  # (batch_size, i)
                # print("tgt_mask", subsequent_mask(ys.size(1)).size())  # (1, i, i)
                out = self.decode(latent.unsqueeze(1), to_var(ys), to_var(subsequent_mask(ys.size(1)).long()))
                prob = self.generator(out[:, -1])
                # print("prob", prob.size())  # (batch_size, vocab_size)
                _, next_word = torch.max(prob, dim=1)
                # print("next_word", next_word.size())  # (batch_size)

                # print("next_word.unsqueeze(1)", next_word.unsqueeze(1).size())

                ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
                # print("ys", ys.size())
            return ys[:, 1:]
        else :
            pad_id = 0
            batch_size = latent.size(0)
            ys = torch.ones(batch_size, 1).fill_(start_id).long().to(latent.device)  # (batch_size, 1)
            for i in range(max_len - 1):
                tensor = self.src_embed(to_var(ys))
                src_attn_mask = to_var((ys != pad_id).long())
                tgt_mask = to_var(subsequent_mask(ys.size(1)).long())
                tensor *= src_attn_mask.unsqueeze(-1).to(tensor.dtype)
                tensor = self.encoder(tensor, tgt_mask)
                tensor *= src_attn_mask.unsqueeze(-1).to(tensor.dtype)
                prob = self.generator(tensor[:, -1])
                _, next_word = torch.max(prob, dim=1)
                ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
            return ys[:, 1:]
                


def make_model(d_vocab, N, d_model, latent_size, d_ff=1024, h=4, dropout=0.1):
    """Helper: Construct a model from hyperparameters."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    share_embedding = Embeddings(d_model, d_vocab)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        # nn.Sequential(Embeddings(d_model, d_vocab), c(position)),
        # nn.Sequential(Embeddings(d_model, d_vocab), c(position)),
        nn.Sequential(share_embedding, c(position)),
        nn.Sequential(share_embedding, c(position)),
        Generator(d_model, d_vocab),
        c(position),
        d_model,
        latent_size,
    )
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
        
def make_deb(N, d_model, d_ff=1024, h=4, dropout=0.1) :
    """debiaser"""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    deb = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
    for p in deb.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return deb

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class Classifier(nn.Module):
    def __init__(self, latent_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(latent_size, 100)
        self.relu1 = nn.LeakyReLU(0.2, )
        self.fc2 = nn.Linear(100, 50)
        self.relu2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(50, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out = self.fc1(input)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)

        # out = F.log_softmax(out, dim=1)
        return out  # batch_size * label_size

def fgim_attack(model, origin_data, target, ae_model, max_sequence_length, id_bos,
                id2text_sentence, id_to_word, gold_ans = None):
    """Fast Gradient Iterative Methods"""

    dis_criterion = nn.BCELoss(size_average=True)
    t = 0.001 # Threshold
    lambda_ = 0.9 # Decay coefficient
    max_iter_per_epsilon=20
    
    if gold_ans is not None :
        gold_text = id2text_sentence(gold_ans, id_to_word)
        print("gold:", gold_text)
        
    flag = False
    for epsilon in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
        print("---------- epsilon:", epsilon)
        generator_id = ae_model.greedy_decode(origin_data, max_len=max_sequence_length, start_id=id_bos)
        generator_text = id2text_sentence(generator_id[0], id_to_word)
        print("z:", generator_text)
        
        data = to_var(origin_data.clone())  # (batch_size, seq_length, latent_size)
        b = True
        if b :
            data.requires_grad = True
            output = model.forward(data)
            loss = dis_criterion(output, target)
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            data = data - epsilon * data_grad
        else :
            data = origin_data
            
        it = 0 
        while True:
            if torch.cdist(output, target) < t :
                flag = True
                break
    
            data = to_var(data.clone())  # (batch_size, seq_length, latent_size)
            # Set requires_grad attribute of tensor. Important for Attack
            data.requires_grad = True
            output = model.forward(data)
            # Calculate gradients of model in backward pass
            loss = dis_criterion(output, target)
            model.zero_grad()
            # dis_optimizer.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            data = data - epsilon * data_grad
            it += 1
            # data = perturbed_data
            epsilon = epsilon * lambda_
            if False :
                generator_id = ae_model.greedy_decode(data,
                                                        max_len=max_sequence_length,
                                                        start_id=id_bos)
                generator_text = id2text_sentence(generator_id[0], id_to_word)
                print("| It {:2d} | dis model pred {:5.4f} |".format(it, output[0].item()))
                print(generator_text)
            if it > max_iter_per_epsilon:
                break
        
        generator_id = ae_model.greedy_decode(data, max_len=max_sequence_length, start_id=id_bos)
        generator_text = id2text_sentence(generator_id[0], id_to_word)
        print("|dis model pred {:5.4f} |".format(output[0].item()))
        print("z*", generator_text)
        print()
        if flag :
            return generator_text

def fgim(data_loader, args, ae_model, c_theta, gold_ans = None) :
    """
    Input: 
        Original latent representation z : (n_batch, batch_size, seq_length, latent_size)
        Well-trained attribute classifier C_θ
        Target attribute y
        A set of weights w = {w_i}
        Decay coefficient λ 
        Threshold t
        
    Output: An optimal modified latent representation z'
    """
    w = args.w
    lambda_ = args.lambda_
    t = args.threshold
    max_iter_per_epsilon = args.max_iter_per_epsilon
    max_sequence_length = args.max_sequence_length
    id_bos = args.id_bos
    id_to_word = args.id_to_word
    limit_batches = args.limit_batches
        
    text_z_prime = {}
    text_z_prime = {"source" : [], "origin_labels" : [], "before" : [], "after" : [], "change" : [], "pred_label" : []}
    if gold_ans is not None :
        text_z_prime["gold_ans"] = []
    z_prime = []
    
    dis_criterion = nn.BCELoss(size_average=True)
    n_batches = 0
    for it in tqdm.tqdm(list(range(data_loader.num_batch)), desc="FGIM"):
        if gold_ans is not None :
            text_z_prime["gold_ans"].append(gold_ans[it])
        
        _, tensor_labels, \
        tensor_src, tensor_src_mask, tensor_src_attn_mask, tensor_tgt, tensor_tgt_y, \
        tensor_tgt_mask, _ = data_loader.next_batch()
        # only on negative example
        negative_examples = ~(tensor_labels.squeeze()==args.positive_label)
        tensor_labels = tensor_labels[negative_examples].squeeze(0) # .view(1, -1)
        tensor_src = tensor_src[negative_examples].squeeze(0) 
        tensor_src_attn_mask = tensor_src_attn_mask[negative_examples].squeeze(0)
        tensor_src_mask = tensor_src_mask[negative_examples].squeeze(0)  
        tensor_tgt_y = tensor_tgt_y[negative_examples].squeeze(0) 
        tensor_tgt = tensor_tgt[negative_examples].squeeze(0) 
        tensor_tgt_mask = tensor_tgt_mask[negative_examples].squeeze(0) 
        #if gold_ans is not None :
        #    text_z_prime["gold_ans"][-1] = text_z_prime["gold_ans"][-1][negative_examples]

        #print("------------%d------------" % it)
        if negative_examples.any():
            text_z_prime["source"].append([id2text_sentence(t, args.id_to_word) for t in tensor_tgt_y])
            text_z_prime["origin_labels"].append(tensor_labels.cpu().numpy())

            origin_data, _ = ae_model.forward(tensor_src, tensor_tgt, tensor_src_mask, tensor_src_attn_mask, tensor_tgt_mask)

            # Define target label
            y_prime = 1.0 - (tensor_labels > 0.5).float()

            ############################### FGIM ######################################################
            generator_id = ae_model.greedy_decode(origin_data, max_len=max_sequence_length, start_id=id_bos)
            generator_text = [id2text_sentence(gid, id_to_word) for gid in generator_id]
            text_z_prime["before"].append(generator_text)
            
            flag = False
            for w_i in w:
                #print("---------- w_i:", w_i)
                data = to_var(origin_data.clone())  # (batch_size, seq_length, latent_size)
                b = True
                if b :
                    data.requires_grad = True
                    output = c_theta.forward(data)
                    loss = dis_criterion(output, y_prime)
                    c_theta.zero_grad()
                    loss.backward()
                    data = data - w_i * data.grad.data
                else :
                    data = origin_data
                    output = c_theta.forward(data)
                    
                it = 0 
                while True:
                    #if torch.cdist(output, y_prime) < t :
                    #if torch.sum((output - y_prime)**2, dim=1).sqrt().mean() < t :
                    if torch.sum((output - y_prime).abs(), dim=1).mean() < t :
                        flag = True
                        break
            
                    data = to_var(data.clone())  # (batch_size, seq_length, latent_size)
                    # Set requires_grad attribute of tensor. Important for Attack
                    data.requires_grad = True
                    output = c_theta.forward(data)
                    # Calculate gradients of model in backward pass
                    loss = dis_criterion(output, y_prime)
                    c_theta.zero_grad()
                    # dis_optimizer.zero_grad()
                    loss.backward()
                    data = data - w_i * data.grad.data
                    it += 1
                    # data = perturbed_data
                    w_i = lambda_ * w_i
                    if False :
                        if text_gen_params is not None :
                            generator_id = ae_model.greedy_decode(data,
                                                                    max_len=max_sequence_length,
                                                                    start_id=id_bos)
                            generator_text = id2text_sentence(generator_id[0], id_to_word)
                            print("| It {:2d} | dis model pred {:5.4f} |".format(it, output[0].item()))
                            print(generator_text)
                    if it > max_iter_per_epsilon:
                        break
                
                if flag :    
                    z_prime.append(data)
                    generator_id = ae_model.greedy_decode(data, max_len=max_sequence_length, start_id=id_bos)
                    generator_text = [id2text_sentence(gid, id_to_word) for gid in generator_id]
                    text_z_prime["after"].append(generator_text)
                    text_z_prime["change"].append([True]*len(output))
                    text_z_prime["pred_label"].append([o.item() for o in output])
                    break
            
            if not flag : # cannot debiaising
                z_prime.append(origin_data)
                text_z_prime["after"].append(text_z_prime["before"][-1])
                text_z_prime["change"].append([False]*len(y_prime))
                text_z_prime["pred_label"].append([o.item() for o in y_prime])
            
            n_batches += 1
            if n_batches > limit_batches:
                break        
    return z_prime, text_z_prime

class LossSedat:
    """"""
    def __init__(self,  penalty="lasso"):
        assert penalty in ["lasso", "ridge"]
        #self.penalty = penalty
        if penalty == "lasso" :
            self.criterion = F.l1_loss
        else :
            self.criterion = F.mse_loss

    def __call__(self, z, z_prime, is_list = True):
        """
        z, z_prime : (n_layers, batch_size, seq_length, latent_size) if is_list, else (batch_size, seq_length, latent_size)
        """
        if is_list:
            # TODO
            #return torch.sum([self.criterion(z_i, z_prime_i) for z_i, z_prime_i in zip(z, z_prime)])
            return sum([self.criterion(z_i, z_prime_i) for z_i, z_prime_i in zip(z, z_prime)])
        else :
            return self.criterion(z, z_prime)

if __name__ == '__main__':
    # plt.figure(figsize=(15, 5))
    # pe = PositionalEncoding(20, 0)
    # y = pe.forward(Variable(torch.zeros(1, 100, 20)))
    # plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
    # plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    # plt.show()

    # Small example model.
    # tmp_model = make_model(10, 10, 2)
    pass