import torch.nn as nn
from utils import clones
from sub_layer_connection import SublayerConnection


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)


    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."

        # x:      a tensor of size (batch, target_length - 1, d_model)
        # memory: a tensor of size (batch, source_length, d_model)

        """Follow Figure 1 (right) for connections
        @param x (tensor(float)):      a tensor of size (batch, target_length - 1, d_model)
        @param memory (tensor(float)): a tensor of size (batch, source_length, d_model)
        @param src_mask (tensor(int)): a tensor of size (batch_size, 1, source_length)
        @param tgt_mask (tensor(int)): a tensor of size (batch_size, target_length - 1, target_length - 1) 
        @returns a result tensor
        """
        m = memory
        # forward of MultiHeadAttention: [Query, Key, Value]
        # TBC: In the synthetic example, the tgt_mask is only a lower triangle matrix, but is it the real world data case? Do we also need to consider about word padding?
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # Yes, the length of query could be different with key and value.
        # We must have:
        #   key.length = value.length (sentence length)
        #   query.dim = key.dim (Or they can not do dot product)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
