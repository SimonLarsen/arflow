from typing import Optional
import torch
from torch import nn
from einops import rearrange


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        head_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=embed_dim // head_dim,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.FloatTensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        x = self.attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            need_weights=False,
        )[0]
        return x


class MLP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        expand_ratio: int = 4,
        emb_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: nn.Module = nn.GELU,
    ):
        super().__init__()

        hidden_dim = embed_dim * expand_ratio
        self.expand = nn.Linear(embed_dim, hidden_dim)
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.contract = nn.Linear(hidden_dim, embed_dim)

        if emb_dim is not None:
            self.emb_proj = nn.Linear(emb_dim, 2 * hidden_dim)

    def forward(
        self,
        x: torch.FloatTensor,
        emb: Optional[torch.FloatTensor] = None,
    ):
        x = self.expand(x)
        x = self.act(x)

        if emb is not None:
            emb = rearrange(self.emb_proj(emb), "... c -> ... 1 c")
            scale, shift = emb.chunk(2, dim=-1)
            x = x * (1.0 + scale) + shift

        x = self.dropout(x)
        x = self.contract(x)

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        embed_dim: int,
        head_dim: int,
        num_layers: int,
        expand_ratio: int = 4,
        condition_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: nn.Module = nn.GELU,
        norm: nn.Module = nn.LayerNorm,
    ):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, embed_dim))

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = [
                norm(embed_dim),
                SelfAttention(
                    embed_dim=embed_dim,
                    head_dim=head_dim,
                    dropout=dropout,
                ),
                norm(embed_dim),
                MLP(
                    embed_dim=embed_dim,
                    expand_ratio=expand_ratio,
                    emb_dim=condition_dim,
                    dropout=dropout,
                    activation=activation,
                ),
            ]
            self.layers.append(nn.ModuleList(layer))

    def forward(
        self,
        x: torch.FloatTensor,
        attn_mask: Optional[torch.Tensor] = None,
        emb: Optional[torch.FloatTensor] = None,
    ):
        x = x + self.pos_embed[:, : x.size(1)]

        for attn_norm, attn, mlp_norm, mlp in self.layers:
            x = attn(attn_norm(x), attn_mask) + x
            x = mlp(mlp_norm(x), emb) + x
        return x
