import torch
from einops import rearrange
from torch import nn
from x_transformers import Encoder


class SimpleViT(nn.Module):
    def __init__(self, *, dim, depth, heads, emb_dropout=0):
        super().__init__()

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Encoder(dim=dim, depth=depth, heads=heads,
                                   rel_pos_bias=True,  # adds relative positional bias to all attention layers, a la T5
                                   sandwich_norm=True,  # set this to True
                                   )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.norm(x)
        x = x.mean(dim=1)

        return x


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=(3,),
                     stride=(stride,), padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.elu = nn.ELU()
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.residual = True
        if in_channels != out_channels:
            self.residual = False

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.residual:
            out = out + x
        out = self.elu(out)
        return out


def create_block_conv(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=(3,),
                  stride=(stride,), padding=1, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.ELU(),
        ResidualBlock(in_channels=out_channels, out_channels=out_channels),
        ResidualBlock(in_channels=out_channels, out_channels=out_channels),
    )


def create_block_aa(in_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, in_channels, groups=in_channels, kernel_size=(3,), padding=1),
        nn.BatchNorm1d(in_channels),
        nn.ELU(),
    )


def create_vit(dim=256, depth=2, heads=8, emb_dropout=0):
    return SimpleViT(
        dim=dim,
        depth=depth,
        heads=heads,
        emb_dropout=emb_dropout
    )


def create_reg(dim=256):
    return nn.Sequential(
        nn.Linear(dim, 1)
    )


class Model(nn.Module):
    def __init__(self, ):
        super(Model, self).__init__()

        self.c1 = create_block_conv(128, 128, 1)
        self.c2 = create_block_conv(128, 160, 1)
        self.c3 = create_block_conv(160, 192, 2)
        self.c4 = create_block_conv(192, 224, 2)
        self.c5 = create_block_conv(224, 256, 2)

        self.l1 = nn.Conv1d(256, 256, kernel_size=(1,))
        self.l2 = nn.Conv1d(224, 256, kernel_size=(1,))
        self.l3 = nn.Conv1d(192, 256, kernel_size=(1,))
        self.l4 = nn.Conv1d(160, 256, kernel_size=(1,))

        self.u2 = nn.ConvTranspose1d(256, 256, kernel_size=(3,), stride=(2,),
                                     padding=(1,), output_padding=(1,))
        self.u3 = nn.ConvTranspose1d(256, 256, kernel_size=(3,), stride=(2,),
                                     padding=(1,), output_padding=(1,))
        self.u4 = nn.ConvTranspose1d(256, 256, kernel_size=(3,), stride=(2,),
                                     padding=(1,), output_padding=(1,))

        self.d2 = create_block_aa(20)
        self.d3 = create_block_aa(40)
        self.d4 = create_block_aa(80)

        self.t1 = create_vit(depth=3)
        self.t2 = create_vit(depth=3)
        self.t3 = create_vit(depth=2)
        self.t4 = create_vit(depth=2)

        self.r1 = create_reg()
        self.r2 = create_reg()
        self.r3 = create_reg()
        self.r4 = create_reg()

    def forward(self, x, return_all=False):
        x = rearrange(x, 'b c d -> b d c')

        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)

        d5 = self.l1(c5)
        d4 = self.u2(d5) + self.l2(c4)
        d3 = self.u3(d4) + self.l3(c3)
        d2 = self.u4(d3) + self.l4(c2)

        d5, d4, d3, d2 = [rearrange(x, 'b d c -> b c d') for x in [d5, d4, d3, d2]]

        r1 = self.t1(d5)
        r2 = self.t2(self.d2(d4))
        r3 = self.t3(self.d3(d3))
        r4 = self.t4(self.d4(d2))

        r1 = self.r1(r1)
        r2 = self.r2(r2)
        r3 = self.r3(r3)
        r4 = self.r4(r4)

        r = torch.cat([r1, r2, r3, r4], dim=1)

        if return_all:
            return r
        else:
            return r.mean(dim=1)
