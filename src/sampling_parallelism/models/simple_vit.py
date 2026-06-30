from typing import Any, Optional, Tuple, TypeVar

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torch_blue.vi import VIkwargs, VILinear, VIModule
from torch_blue.vi.distributions import Distribution, MeanFieldNormal

T = TypeVar("T")


def pair(t: T) -> Tuple[T, T]:
    """Ensure twin tuple."""
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(
    h: int,
    w: int,
    dim: int,
    temperature: int = 10000,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """2d cosine positional encoding."""
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class VIFeedForward(VIModule):
    """Vision Transformer feed forward block."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        variational_distribution: Distribution = MeanFieldNormal(),
        prior: Distribution = MeanFieldNormal(),
        rescale_prior: bool = True,
        kaiming_initialization: bool = True,
        prior_initialization: bool = False,
        return_log_probs: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        vikwargs: VIkwargs = dict(
            variational_distribution=variational_distribution,
            prior=prior,
            rescale_prior=rescale_prior,
            kaiming_initialization=kaiming_initialization,
            prior_initialization=prior_initialization,
            return_log_probs=return_log_probs,
            device=device,
            dtype=dtype,
        )
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            VILinear(dim, hidden_dim, **vikwargs),
            nn.GELU(),
            VILinear(hidden_dim, dim, **vikwargs),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.net(x)


class VIAttention(VIModule):
    """Vision Transformer Attention module."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        variational_distribution: Distribution = MeanFieldNormal(),
        prior: Distribution = MeanFieldNormal(),
        rescale_prior: bool = True,
        kaiming_initialization: bool = True,
        prior_initialization: bool = False,
        return_log_probs: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        vikwargs: VIkwargs = dict(
            variational_distribution=variational_distribution,
            prior=prior,
            rescale_prior=rescale_prior,
            kaiming_initialization=kaiming_initialization,
            prior_initialization=prior_initialization,
            return_log_probs=return_log_probs,
            device=device,
            dtype=dtype,
        )
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = VILinear(dim, inner_dim * 3, bias=False, **vikwargs)
        self.to_out = VILinear(inner_dim, dim, bias=False, **vikwargs)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.norm(x)

        qkv = self.to_qkv(x)
        qkv = qkv.chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class VITransformer(VIModule):
    """Vision Transformer Transformer core."""

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        variational_distribution: Distribution = MeanFieldNormal(),
        prior: Distribution = MeanFieldNormal(),
        rescale_prior: bool = True,
        kaiming_initialization: bool = True,
        prior_initialization: bool = False,
        return_log_probs: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        vikwargs: VIkwargs = dict(
            variational_distribution=variational_distribution,
            prior=prior,
            rescale_prior=rescale_prior,
            kaiming_initialization=kaiming_initialization,
            prior_initialization=prior_initialization,
            return_log_probs=return_log_probs,
            device=device,
            dtype=dtype,
        )
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        VIAttention(dim, heads=heads, dim_head=dim_head, **vikwargs),
                        VIFeedForward(dim, mlp_dim, **vikwargs),
                    ]
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class SimpleViT(VIModule):
    """Simple Vision Tranformer."""

    def __init__(
        self,
        *,
        image_size: int,
        in_channels: int = 3,
        out_features: int,
        patch_size: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        variational_distribution: Distribution = MeanFieldNormal(),
        prior: Distribution = MeanFieldNormal(),
        rescale_prior: bool = True,
        kaiming_initialization: bool = True,
        prior_initialization: bool = False,
        return_log_probs: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        vikwargs: VIkwargs = dict(
            variational_distribution=variational_distribution,
            prior=prior,
            rescale_prior=rescale_prior,
            kaiming_initialization=kaiming_initialization,
            prior_initialization=prior_initialization,
            return_log_probs=return_log_probs,
            device=device,
            dtype=dtype,
        )
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        patch_dim = in_channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            VILinear(patch_dim, dim, **vikwargs),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h=image_height // patch_height,
            w=image_width // patch_width,
            dim=dim,
        )

        self.transformer = VITransformer(
            dim, depth, heads, dim_head, mlp_dim, **vikwargs
        )

        self.pool = "mean"
        # self.to_latent = nn.Identity()

        self.linear_head = VILinear(dim, out_features, **vikwargs)

    def forward(self, img: Tensor) -> Tensor:
        """Forward pass."""
        device = img.device
        x = self.to_patch_embedding(img)
        x = x + self.pos_embedding.to(device, dtype=x.dtype)
        x = self.transformer(x)
        x = x.mean(dim=1)
        out = self.linear_head(x)
        return out


class TinySimpleViT(SimpleViT):
    """Tiny Variant of SimpleViT with default args."""

    def __init__(
        self,
        *,
        image_size: int,
        in_channels: int = 3,
        out_features: int,
        patch_size: int = 4,
        dim: int = 192,
        depth: int = 6,
        heads: int = 3,
        mlp_dim: int = 4 * 192,
        dim_head: int = 192 // 3,
        variational_distribution: Distribution = MeanFieldNormal(),
        prior: Distribution = MeanFieldNormal(),
        rescale_prior: bool = True,
        kaiming_initialization: bool = True,
        prior_initialization: bool = False,
        return_log_probs: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            image_size=image_size,
            in_channels=in_channels,
            out_features=out_features,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            variational_distribution=variational_distribution,
            prior=prior,
            rescale_prior=rescale_prior,
            kaiming_initialization=kaiming_initialization,
            prior_initialization=prior_initialization,
            return_log_probs=return_log_probs,
            device=device,
            dtype=dtype,
        )


class BaseSimpleViT(SimpleViT):
    """Base Variant of SimpleViT with default args."""

    def __init__(
        self,
        *,
        image_size: int,
        in_channels: int = 3,
        out_features: int,
        patch_size: int = 4,
        dim: int = 768,
        depth: int = 6,
        heads: int = 3,
        mlp_dim: int = 4 * 768,
        dim_head: int = 768 // 3,
        variational_distribution: Distribution = MeanFieldNormal(),
        prior: Distribution = MeanFieldNormal(),
        rescale_prior: bool = True,
        kaiming_initialization: bool = True,
        prior_initialization: bool = False,
        return_log_probs: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            image_size=image_size,
            in_channels=in_channels,
            out_features=out_features,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            variational_distribution=variational_distribution,
            prior=prior,
            rescale_prior=rescale_prior,
            kaiming_initialization=kaiming_initialization,
            prior_initialization=prior_initialization,
            return_log_probs=return_log_probs,
            device=device,
            dtype=dtype,
        )
