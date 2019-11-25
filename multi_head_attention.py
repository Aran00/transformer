import torch.nn as nn
from utils import clones, attention


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h        # The output dim of each head
        self.h = h                     # Number of heads
        # The former 3 linear modules are the combined {W_i}^Q, {W_i}^K, {W_i}^V
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    # For a transformer, the value could be of the different dim d_v.
    # What should be the input in this case? So boring to add a linear layer to get v?
    def forward(self, query, key, value, mask=None):
        """Implements Figure 2
        @param query (tensor(float)]): a tensor of size (batch_size, sentence_length, embed_size)   here embed_size == d_model
        @param key (tensor(float)]): a tensor of size (batch_size, sentence_length, embed_size)
        @param value (tensor(float)): a tensor of size (batch_size, sentence_length, embed_size)
        @param mask (tensor(float)): a tensor of size (batch_size, 1, 1, sentence_length)   (Not sure about whether it's always that size in dim 1,2)
        @returns a result tensor
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)    # (batch, h, sent_len, d_k)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
