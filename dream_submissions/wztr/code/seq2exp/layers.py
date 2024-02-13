import math
import torch
from torch import einsum
import torch.nn as nn
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import numpy as np


# adopted some classes from ucidrains/enformer-pytorch
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# def ConvBlock(dim, dim_out = None, kernel_size = 1):
#     return nn.Sequential(
#         nn.BatchNorm1d(dim),
#         GELU(),
#         nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding = kernel_size // 2)
#     )
def HybridConvBlock(
    in_channels, out_channels, kernel_size, padding="same", stride=1, dilation_list=None
):
    if dilation_list is None:
        return ConvBlock(in_channels, out_channels, kernel_size, padding, stride)
    else:
        return NotImplementedError
        # TODO: implement


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class HybridConvBlockClass(nn.Module):
    def __init__(self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        padding="same",
        stride=1,
        dilation_list=None,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        if dilation_list is None:
            self.layers.append(ConvBlock(in_channels, out_channels, kernel_size, padding, stride))
        else:
            for dilation in dilation_list:
                print(dilation_list)
                print(out_channels/len(dilation_list))
                self.layers.append(ConvBlock(in_channels, int(out_channels/len(dilation_list)), kernel_size, padding=padding, dilation=dilation))

    def forward(self, x):
        result = torch.Tensor([]).to(device)
        for layer in self.layers:
            output = layer(x)
            result = torch.concat((result, output), 1)
        return result




def ConvBlock(
    in_channels,
    out_channels,
    kernel_size,
    padding=0,
    stride=1,
    dilation=1,
    activation="relu",
):
    if activation == "elu":
        act_func = nn.ELU()
    elif activation == "sigmoid":
        act_func = nn.Sigmoid()
    elif activation == "gelu":
        act_func = GELU()
    elif activation == "exp":
        act_func = EXP()
    else:
        act_func = nn.ReLU()

    return nn.Sequential(
        nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
        ),
        nn.BatchNorm1d(out_channels),
        act_func,
    )


def negative_pearson_loss(x, y):
    # print(torch.stack([x, y], dim=0).shape)
    return -torch.corrcoef(torch.stack([x, y], dim=0))[0, 1]


def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).sum() / weight.sum()


def mixup_augmentation(x, y, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    # if use_cuda:
    #     index = torch.randperm(batch_size).cuda()
    # else:
    #     index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index]

    return torch.cat([mixed_x, x]), torch.cat([mixed_y, y])


class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x


class EXP(nn.Module):
    def forward(self, x):
        return torch.exp(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size=2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange("b d (n p) -> b d n p", p=2)
        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value=0)
            mask = torch.zeros((b, 1, n), dtype=torch.bool, device=x.device)
            mask = F.pad(mask, (0, remainder), value=True)

        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)

        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = logits.softmax(dim=-1)

        return (x * attn).sum(dim=-1)


class SoftmaxPool(nn.Module):
    def __init__(self, dim, pool_size=2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange("b d (n p) -> b d n p", p=2)

    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value=0)
            mask = torch.zeros((b, 1, n), dtype=torch.bool, device=x.device)
            mask = F.pad(mask, (0, remainder), value=True)

        x = self.pool_fn(x)
        logits = x

        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = logits.softmax(dim=-1)

        return (x * attn).sum(dim=-1)


# RELU[0,17] is used as the last layer to make output in range [0,17]
class RELU0_17(nn.Module):
    def forward(self, x):
        return torch.clamp(x, 0, 17)


# relative positional encoding functions  (from enformer-pytorch)--------------------------


def get_positional_features_exponential(
    positions, features, seq_len, min_half_life=3.0
):
    max_range = math.log(seq_len) / math.log(2.0)
    half_life = 2 ** torch.linspace(
        min_half_life, max_range, features, device=positions.device
    )
    half_life = half_life[None, ...]
    positions = positions.abs()[..., None]
    return torch.exp(-math.log(2.0) / half_life * positions)


def get_positional_features_central_mask(positions, features, seq_len):
    center_widths = 2 ** torch.arange(1, features + 1, device=positions.device).float()
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).float()


def gamma_pdf(x, concentration, rate):
    log_unnormalized_prob = torch.xlogy(concentration - 1.0, x) - rate * x
    log_normalization = torch.lgamma(concentration) - concentration * torch.log(rate)
    return torch.exp(log_unnormalized_prob - log_normalization)


def get_positional_features_gamma(
    positions, features, seq_len, stddev=None, start_mean=None, eps=1e-8
):
    if not exists(stddev):
        stddev = seq_len / (2 * features)

    if not exists(start_mean):
        start_mean = seq_len / features

    mean = torch.linspace(start_mean, seq_len, features, device=positions.device)
    mean = mean[None, ...]
    concentration = (mean / stddev) ** 2
    rate = mean / stddev**2
    probabilities = gamma_pdf(positions.float().abs()[..., None], concentration, rate)
    probabilities = probabilities + eps
    outputs = probabilities / torch.amax(probabilities, dim=-1, keepdim=True)
    return outputs


def get_positional_embed(seq_len, feature_size, device):
    distances = torch.arange(-seq_len + 1, seq_len, device=device)

    feature_functions = [
        get_positional_features_exponential,
        get_positional_features_central_mask,
        get_positional_features_gamma,
    ]

    num_components = len(feature_functions) * 2

    if (feature_size % num_components) != 0:
        raise ValueError(
            f"feature size is not divisible by number of components ({num_components})"
        )

    num_basis_per_class = feature_size // num_components

    embeddings = []
    for fn in feature_functions:
        embeddings.append(fn(distances, num_basis_per_class, seq_len))

    embeddings = torch.cat(embeddings, dim=-1)
    embeddings = torch.cat(
        (embeddings, torch.sign(distances)[..., None] * embeddings), dim=-1
    )
    return embeddings


def relative_shift(x):
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim=-1)
    _, h, t1, t2 = x.shape
    x = x.reshape(-1, h, t2, t1)
    x = x[:, :, 1:, :]
    x = x.reshape(-1, h, t1, t2 - 1)
    return x[..., : ((t2 + 1) // 2)]


# Attention Layer--------------------------------------------------


class Attention(nn.Module):
    def __init__(
        self,
        dim,  # the input has shape (batch_size, len, dim) = (b, n, dim)
        *,
        heads=8,
        dim_key=64,
        dim_value=64,
        dropout=0.0,
        pos_dropout=0.0,
    ):
        super().__init__()
        self.scale = dim_key**-0.5
        self.heads = heads

        # Q, K, V

        self.to_q = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias=False)

        self.to_out = nn.Linear(dim_value * heads, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        # relative positional encoding

        self.num_rel_pos_features = 66  ###########

        self.to_rel_k = nn.Linear(
            self.num_rel_pos_features, dim_key * heads, bias=False
        )

        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))

        # dropouts

        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        n, h, device = x.shape[-2], self.heads, x.device

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        q = q * self.scale

        content_logits = einsum(
            "b h i d, b h j d -> b h i j", q + self.rel_content_bias, k
        )

        positions = get_positional_embed(n, self.num_rel_pos_features, device)
        positions = self.pos_dropout(positions)
        rel_k = self.to_rel_k(positions)

        rel_k = rearrange(rel_k, "n (h d) -> h n d", h=h)
        rel_logits = einsum("b h i d, h j d -> b h i j", q + self.rel_pos_bias, rel_k)
        rel_logits = relative_shift(rel_logits)

        logits = content_logits + rel_logits  # shape (b, h, n, n)
        attn = logits.softmax(dim=-1)  # softmax over the last dimension
        attn = self.attn_dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)  # (b, n, dim)
