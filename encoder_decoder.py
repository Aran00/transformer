import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
# import seaborn
# seaborn.set_context(context="talk")
# %matplotlib inline        # For jupyter notebook


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architechture. Base for this and many other models
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator


    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences.
        @param src (tensor(int)]): a tensor of size (batch_size, sentence_length)
        @param tgt (tensor(int)]): a tensor of size (batch_size, sentence_length - 1)
        @param src_mask (tensor(int)): a tensor of size (batch_size, 1, sentence_length)    (is the 2nd dim always 1?)
        @param tgt_mask (tensor(int)): a tensor of size (batch_size, sentence_length - 1, sentence_length - 1)   (Not sure about whether it's always that size)
        @returns a result tensor
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    # Memory: a tensor of size (batch, sent_length, d_model)
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
