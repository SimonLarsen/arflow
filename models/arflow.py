from typing import Optional, Union, Callable, Sequence
from functools import partial
import torch
from torch import nn
from torchdiffeq import odeint
from einops import rearrange, repeat
import tqdm
from .outputs import FlowModelOutput
from .activations import get_activation
from .norms import get_norm
from .embeddings import TimestepEmbedding
from .transformer import Transformer
from .cnn import ResBlock
from .schedules import sample_lognorm


class ARFlow(nn.Module):
    def __init__(
        self,
        input_size: int = 256,
        patch_size: int = 16,
        head_dim: int = 32,
        mlp_expand_ratio: int = 4,
        ar_embed_dim: int = 512,
        ar_num_layers: int = 12,
        flow_embed_dim: int = 256,
        flow_num_layers: int = 8,
        flow_condition_dim: int = 256,
        num_classes: Optional[int] = None,
        dropout: float = 0.0,
        activation: Union[str, nn.Module] = nn.GELU,
        norm: Union[str, nn.Module] = nn.LayerNorm,
        noise_schedule: Union[str, Callable] = sample_lognorm,
    ):
        super().__init__()

        if isinstance(activation, str):
            activation = get_activation(activation)

        if isinstance(norm, str):
            norm = get_norm(norm)

        if isinstance(noise_schedule, str):
            if noise_schedule == "lognorm":
                noise_schedule = sample_lognorm
            else:
                raise ValueError(
                    f"Unsupported schedule type '{noise_schedule}'."
                )
        self.noise_schedule = noise_schedule

        self.input_size = input_size
        self.patch_size = patch_size
        self.num_tokens = (input_size // patch_size) ** 2
        self.ar_embed_dim = ar_embed_dim
        self.flow_embed_dim = flow_embed_dim
        self.flow_condition_dim = flow_condition_dim

        # Embeddings
        self.time_embedding = TimestepEmbedding(
            embedding_dim=flow_condition_dim,
            max_period=1000,
        )

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=ar_embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        has_classes = num_classes is not None and num_classes > 0
        if has_classes:
            self.class_embed = nn.Embedding(
                num_embeddings=num_classes,
                embedding_dim=ar_embed_dim,
            )

        # Transformer model
        self.transformer = Transformer(
            num_tokens=self.num_tokens,
            embed_dim=ar_embed_dim,
            head_dim=head_dim,
            num_layers=ar_num_layers,
            expand_ratio=mlp_expand_ratio,
            condition_dim=ar_embed_dim if has_classes else None,
            dropout=dropout,
            activation=activation,
            norm=norm,
        )
        self.transformer_proj_out = nn.Linear(ar_embed_dim, flow_condition_dim)

        # Flow matching model
        self.flow_conv_in = nn.Conv2d(3, flow_embed_dim, 3, 1, 1)

        self.flow_blocks = nn.ModuleList()
        for _ in range(flow_num_layers):
            self.flow_blocks.append(
                ResBlock(
                    channels=flow_embed_dim,
                    emb_channels=flow_condition_dim,
                    activation=activation,
                    norm=norm,
                )
            )

        self.flow_conv_out = nn.Conv2d(flow_embed_dim, 3, 3, 1, 1)

        # Buffer causal mask
        attn_mask = nn.Transformer.generate_square_subsequent_mask(
            self.num_tokens
        )
        self.register_buffer("attn_mask", attn_mask)

    def forward_transformer(
        self,
        x: torch.FloatTensor,
        emb: Optional[torch.FloatTensor] = None,
    ):
        attn_mask = self.attn_mask
        if attn_mask is not None:
            seq_len = x.size(1)
            attn_mask = attn_mask[:seq_len, :seq_len]

        x = self.transformer(
            x=x,
            attn_mask=attn_mask,
            emb=emb,
        )
        x = self.transformer_proj_out(x)
        return x

    def forward_flow(self, x, z):
        x = self.flow_conv_in(x)
        for block in self.flow_blocks:
            x = block(x, z)
        x = self.flow_conv_out(x)
        return x

    def forward(
        self,
        inputs: Sequence[torch.Tensor],
    ) -> FlowModelOutput:
        x = inputs[0]
        classes = inputs[1] if len(inputs) > 1 else None

        # Class embedding
        ar_embed = None
        if classes is not None:
            ar_embed = self.class_embed(classes)

        # Compute image features
        z = self.patch_embed(x)
        z = rearrange(z, "... e h w -> ... (h w) e")
        z = torch.cat((torch.zeros_like(z[:, :1]), z[:, :-1]), dim=1)

        # Pass through transformer
        z = self.forward_transformer(z, ar_embed)

        # Noise patches for flow matching
        target = rearrange(
            x,
            "b c (py ph) (px pw) -> (b py px) c ph pw",
            ph=self.patch_size,
            pw=self.patch_size,
        )

        noise = torch.randn_like(target)
        flow = target - noise
        timesteps = self.noise_schedule((target.size(0),)).to(
            x.device, dtype=x.dtype
        )
        timesteps_expanded = timesteps.reshape(-1, 1, 1, 1)
        h = timesteps_expanded * target + (1.0 - timesteps_expanded) * noise

        # Predict flow
        z = rearrange(z, "b np e -> (b np) e")
        time_embed = self.time_embedding(timesteps * 1000.0)
        h = self.forward_flow(h.to(z.dtype), z + time_embed)

        # Reintroduce patch dimension
        h = rearrange(h, "(b np) ... -> b np ...", np=self.num_tokens)
        flow = rearrange(flow, "(b np) ... -> b np ...", np=self.num_tokens)

        return FlowModelOutput(
            flow_pred=h,
            flow_true=flow.to(h.dtype),
        )

    @torch.no_grad()
    def sample(
        self,
        num_images: int = 1,
        num_flow_iterations: int = 30,
        classes: Optional[Union[int, Sequence[int]]] = None,
        generator: Optional[torch.Generator] = None,
        guidance_scale: float = 2.0,
        odeint_method: str = "euler",
        progress: bool = True,
    ):
        bs = num_images
        device = self.transformer.pos_embed.device
        dtype = self.transformer.pos_embed.dtype
        cfg = guidance_scale > 1.0 and classes is not None

        # Embed classes labels
        ar_embed = None
        if classes is not None:
            classes = torch.as_tensor(classes, dtype=torch.long, device=device)
            classes = classes.reshape(-1).expand(bs)
            if cfg:
                classes = torch.cat((classes, torch.zeros_like(classes)))
            ar_embed = self.class_embed(classes)

        tokens = []
        patches = []
        flow_timesteps = torch.linspace(
            0, 1, num_flow_iterations, device=device, dtype=dtype
        )
        tokens.append(
            torch.zeros((bs, 1, self.ar_embed_dim), device=device, dtype=dtype)
        )

        iterations = range(self.num_tokens)
        if progress:
            iterations = tqdm.tqdm(iterations)

        for _ in iterations:
            # Compute patch features
            x = torch.cat(tokens, dim=1)
            if cfg:
                x = repeat(x, "b ... -> (2 b) ...")
            z = self.forward_transformer(x, ar_embed)

            # Predict flow based on last token
            z = z[:, -1]
            noise = torch.randn(
                bs, 3, 16, 16, device=device, dtype=dtype, generator=generator
            )

            def _pred_flow(t, x, z, model):
                temb = model.time_embedding(t.reshape(1) * 1000.0)
                if cfg:
                    x = repeat(x, "b ... -> (2 b) ...")
                pred = model.forward_flow(x, z + temb)
                if cfg:
                    pred, pred_uncond = pred.chunk(chunks=2, dim=0)
                    pred = pred_uncond + guidance_scale * (pred - pred_uncond)
                return pred

            traj = odeint(
                partial(_pred_flow, z=z, model=self),
                noise,
                flow_timesteps,
                method=odeint_method,
                atol=1e-5,
                rtol=1e-5,
            )

            # Add predicted patch to output
            patch = traj[-1].clamp(-1, 1)
            patches.append(patch)

            # Encode patch and add to sequence
            token = self.patch_embed(patch)
            tokens.append(rearrange(token, "b e 1 1 -> b 1 e"))

        output = rearrange(
            patches,
            "(py px) b c ph pw -> b c (py ph) (px pw)",
            py=self.input_size // self.patch_size,
        )
        output = output / 2.0 + 0.5

        return output
