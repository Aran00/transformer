import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):  # Create a upper triangle mask matrix
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


# input: # tensor of size (batch, h, sent_len, d_k)
def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention
    @param query (tensor(float)]): a tensor of size (batch_size, h, query_length, query_size)   here embed_size == d_model
    @param key (tensor(float)]): a tensor of size (batch_size, h, key_length, key_size)
    @param value (tensor(float)): a tensor of size (batch_size, h, value_length, value_size)
    @param mask (tensor(float)): a tensor of size (batch_size, 1, 1, key_length)  To be broadcastable with attention result scores
    @param dropout(nn.Module): A dropout layer
    We have key_length == value_length, query_size == key_size
    In our program we also have value_size == key_size, and key_size * h = d_model
    @returns a result tensor
    """

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)       # (batch, h, query_len, key_len)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)    # mask: (batch, 1, 1, key_len)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

