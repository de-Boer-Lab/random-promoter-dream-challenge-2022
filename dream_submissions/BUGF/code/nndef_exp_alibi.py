import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from math import log2, floor

# Conditional positional encoding layer (Chu et al. arXiv:2102.10882v2)
class PEGLayer(nn.Module):
    def __init__(self, width, p_drop=0.0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=width, out_channels=width, kernel_size=7, padding=3, groups=width, bias=True)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        return x + self.dropout(self.conv(x.permute(0,2,1)).permute(0,2,1))

    
# Symmetric ALiBi relative positional bias adapted from https://github.com/lucidrains/x-transformers
class AlibiPositionalBias(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> () h () ()')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** floor(log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward(self, residx):
        relative_position = residx.unsqueeze(0) - residx.unsqueeze(1)
        bias = torch.abs(relative_position).clip(max=40).unsqueeze(0).unsqueeze(0).expand(1, self.heads, -1,-1)
        return bias * -self.slopes


# Implementation for tied multihead attention with ALiBi relative pos encoding
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, heads, k_dim=None, v_dim=None):
        super().__init__()
        if k_dim == None:
            k_dim = d_model
        if v_dim == None:
            v_dim = d_model

        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.scaling = self.d_k ** -0.25

        self.to_query = nn.Linear(d_model, d_model, bias=None)
        self.to_key = nn.Linear(k_dim, d_model, bias=None)
        self.to_value = nn.Linear(v_dim, d_model, bias=None)
        self.to_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, posbias=None, return_att=False):
        B, L = query.shape[:2]

        q = self.to_query(query).view(B, L, self.heads, self.d_k).permute(0,2,1,3) # (B, h, l, k)
        k = self.to_key(key).view(B, L, self.heads, self.d_k).permute(0,2,3,1) # (B, h, k, l)
        v = self.to_value(value).view(B, L, self.heads, self.d_k).permute(0,2,1,3) # (B, h, l, k)

        # Scale both Q & K to help avoid fp16 overflows
        q = q * self.scaling
        k = k * self.scaling
        attention = torch.einsum('bhik,bhkj->bhij', q, k)
        if posbias is not None:
            attention = attention + posbias
        attention = F.softmax(attention, dim=-1) # (B, h, L, L)
        #
        out = torch.matmul(attention, v) # (B, h, L, d_k)
        #print(out)
        out = out.permute(0,2,1,3).reshape(B, L, -1)
        return self.to_out(out)

# SwiGLU activation function
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.silu(gates)


# Sequence encoder block
class SeqEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, p_drop=0.1):
        super().__init__()

        # Multihead attention
        self.attn = MultiheadAttention(d_model, heads)

        self.gate = nn.Linear(d_model, d_model)
        nn.init.constant_(self.gate.weight, 0.)
        nn.init.constant_(self.gate.bias, 1.)

        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*8),
            SwiGLU(),
            nn.Dropout(p_drop),
            nn.Linear(d_model*4, d_model)
        )

        nn.init.zeros_(self.ff[3].weight)
        nn.init.zeros_(self.ff[3].bias)

        # Normalization and dropout modules
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x, bias):
        # Input shape for multihead attention: (BATCH, NRES, EMB)
        # Tied multihead attention w/ pre-LayerNorm
        B, L = x.shape[:2]
        x2 = x
        x = self.norm1(x)
        x = torch.sigmoid(self.gate(x)) * self.attn(x, x, x, bias) # Tied attention over L (requires 4D input)
        x = x2 + self.dropout(x)

        # feed-forward
        x2 = x
        x = self.norm2(x)
        x = self.ff(x)
        return x2 + x

# Main TranformerNet Module
class TransformerNet(nn.Module):
    def __init__(self,width,depth,heads):
        super(TransformerNet, self).__init__()

        self.width = width

        self.embed = nn.Embedding(6, width)

        self.peg = PEGLayer(width)

        self.alibi = AlibiPositionalBias(heads)

        layers = []
        layer = SeqEncoderLayer(width, heads, p_drop=0.1)
        for _ in range(depth):
            layers.append(layer)

        self.seqencoder = nn.ModuleList(layers)
        self.to_exp = nn.Linear(width, 18)
        self.to_mut = nn.Linear(width, 1)


    def forward(self, x):
        x = self.embed(x.long())
        L = x.size(1)
        x = self.peg(x)
        posbias = self.alibi(torch.arange(L, device=x.device))

        for m in self.seqencoder:
            x = m(x, posbias)

        exp_out = self.to_exp(x.mean(1))
        mut_out = self.to_mut(x).squeeze(-1)

        return exp_out, mut_out
