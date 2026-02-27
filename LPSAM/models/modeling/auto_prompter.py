import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Any, Optional, Tuple, Type


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class APBlock(nn.Module):
    """
    [cross attention, MLP]
    """

    def __init__(self, embedding_dim, num_heads=8):
        super(APBlock, self).__init__()
        self.attn = Attention(embedding_dim=embedding_dim, num_heads=num_heads)
        self.ln = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, q, k, v):
        return self.fc(self.ln(self.attn(q, k, v) + q))


class TwoWayAlignBlock(nn.Module):
    """
    align visual and query token
    """

    def __init__(self, embedding_dims, num_heads):
        super(TwoWayAlignBlock, self).__init__()

        self.cpa = Attention(embedding_dim=embedding_dims[0], num_heads=num_heads)
        self.single_conv = SingleConv(in_channels=embedding_dims[0],out_channels=embedding_dims[0])
        self.align_block = nn.ModuleList([APBlock(embedding_dim=embedding_dims[0], num_heads=num_heads),
                                          APBlock(embedding_dim=embedding_dims[1], num_heads=num_heads)])

    def forward(self, embeddings: list):
        # assert len(embeddings) == 2, "only support two way"
        vit_embeddings = embeddings[0]
        sparse_embeddings = embeddings[1]
        image_pe = embeddings[2]
        cnn_embeddings = self.single_conv(embeddings[3])
        cnn_embeddings_ = cnn_embeddings.flatten(2).permute(0, 2, 1)
        pe = torch.repeat_interleave(image_pe.flatten(2).permute(0, 2, 1), vit_embeddings.shape[0], dim=0)
        vit_embeddings = vit_embeddings + pe

        fusion_embeddings = self.cpa(vit_embeddings, cnn_embeddings_, cnn_embeddings_)

        return [self.align_block[0](fusion_embeddings, sparse_embeddings, sparse_embeddings),
                self.align_block[1](sparse_embeddings, fusion_embeddings, fusion_embeddings),
                image_pe, cnn_embeddings]


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SingleConv(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)


class SingleDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU()  # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class SingleCNNEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
            self,
            patchsize: int = 8,
            in_chans: int = 1,
            embed_dim: int = 768,
    ) -> None:
        """
        Args:
            patch_size (int): kernel size of the tokenization layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()
        downtimes = int(math.log2(patchsize))
        mid_channel = 64
        self.inc = SingleConv(in_chans, mid_channel)
        self.downs = nn.ModuleList()
        for i in range(downtimes):
            if i == downtimes - 1:
                down = SingleDown(mid_channel, embed_dim)
            else:
                down = SingleDown(mid_channel, mid_channel * 2)
            mid_channel = mid_channel * 2
            self.downs.append(down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inc(x)
        for down in self.downs:
            x = down(x)
        return x


class AutoPrompter(nn.Module):
    def __init__(self, num_tasks=2, patch_size=8, embedding_dim=256, num_heads=8, num_blocks=4):
        super(AutoPrompter, self).__init__()
        self.num_tasks = num_tasks
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.initial_sparse_embedding = nn.Embedding(num_embeddings=self.num_tasks, embedding_dim=self.embedding_dim)
        self.cnn_embed = SingleCNNEmbed(patchsize=self.patch_size, in_chans=3, embed_dim=self.embedding_dim)
        self.blocks = nn.Sequential(
            *[TwoWayAlignBlock(embedding_dims=[self.embedding_dim, self.embedding_dim], num_heads=self.num_heads) for _
              in range(self.num_blocks)])

    def forward(self, imgs, vit_embeddings, image_pe):
        vit_embeddings_ = vit_embeddings.flatten(2).permute(0, 2, 1)
        initial_sparse_embeddings = torch.repeat_interleave(self.initial_sparse_embedding.weight.unsqueeze(0),
                                                            vit_embeddings_.shape[0], dim=0)
        if imgs.size()[1] == 1:
            imgs_ = imgs.repeat(1, 3, 1, 1)

        cnn_embed = self.cnn_embed(imgs_)

        return self.blocks([vit_embeddings_, initial_sparse_embeddings, image_pe, cnn_embed])
