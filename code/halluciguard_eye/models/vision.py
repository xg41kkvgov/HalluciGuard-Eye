"""ViT-L/14-compatible image encoder. Ref: Sec. III-C.1, Eq. (3), and Fig. 2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from halluciguard_eye.models.layers import TransformerBlock


@dataclass(frozen=True)
class VisionOutput:
    tokens: Tensor
    pooled: Tensor
    attention_maps: tuple[Tensor, ...]


class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, width: int, channels: int = 3) -> None:
        super().__init__()
        if image_size <= 0 or patch_size <= 0 or image_size % patch_size:
            raise ValueError("image size must be a positive multiple of patch size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.width = width
        self.grid_size = image_size // patch_size
        self.patch_count = self.grid_size**2
        self.projection = nn.Conv2d(
            channels, width, kernel_size=patch_size, stride=patch_size, bias=False
        )

    def forward(self, images: Tensor) -> Tensor:
        expected = (self.image_size, self.image_size)
        if images.ndim != 4 or images.shape[1] != 3 or images.shape[-2:] != expected:
            raise ValueError(
                f"images must have shape batch x 3 x {self.image_size} x {self.image_size}"
            )
        return self.projection(images).flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 336,
        patch_size: int = 14,
        width: int = 1024,
        layers: int = 24,
        heads: int = 16,
        output_width: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if layers <= 0:
            raise ValueError("vision transformer must have at least one layer")
        self.width = width
        self.patch_embedding = PatchEmbedding(image_size, patch_size, width)
        tokens = self.patch_embedding.patch_count + 1
        self.class_token = nn.Parameter(torch.empty(1, 1, width))
        self.position_embedding = nn.Parameter(torch.empty(1, tokens, width))
        self.blocks = nn.ModuleList(
            TransformerBlock(width, heads, mlp_ratio=4.0, dropout=dropout) for _ in range(layers)
        )
        self.norm = nn.LayerNorm(width)
        self.output_projection = (
            nn.Linear(width, output_width, bias=False) if output_width else nn.Identity()
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.class_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, images: Tensor, return_attention: bool = False) -> VisionOutput:
        patches = self.patch_embedding(images)
        class_token = self.class_token.expand(images.shape[0], -1, -1)
        hidden = torch.cat((class_token, patches), dim=1)
        hidden = hidden + self.position_embedding[:, : hidden.shape[1]]
        attention_maps: list[Tensor] = []
        for block in self.blocks:
            hidden, attention = block(hidden)
            if return_attention:
                attention_maps.append(attention)
        hidden = self.output_projection(self.norm(hidden))
        return VisionOutput(hidden, hidden[:, 0], tuple(attention_maps))

    def patch_coordinates(self) -> Tensor:
        size = self.patch_embedding.grid_size
        axis = torch.linspace(0.0, 1.0, size)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        return torch.stack((x.flatten(), y.flatten()), dim=-1)


def interpolate_position_embedding(position_embedding: Tensor, target_grid: int) -> Tensor:
    if position_embedding.ndim != 3 or position_embedding.shape[0] != 1:
        raise ValueError("position embedding must have shape 1 x tokens x width")
    source_tokens = position_embedding.shape[1] - 1
    source_grid = int(source_tokens**0.5)
    if source_grid**2 != source_tokens:
        raise ValueError("source patch token count must be square")
    class_position = position_embedding[:, :1]
    patches = position_embedding[:, 1:].reshape(1, source_grid, source_grid, -1).permute(0, 3, 1, 2)
    resized = torch.nn.functional.interpolate(
        patches,
        size=(target_grid, target_grid),
        mode="bicubic",
        align_corners=False,
    )
    resized = resized.permute(0, 2, 3, 1).reshape(1, target_grid**2, -1)
    return torch.cat((class_position, resized), dim=1)


def load_visual_state(
    model: VisionTransformer, state: dict[str, Tensor], strict: bool = True
) -> None:
    position_key = "position_embedding"
    if position_key in state and state[position_key].shape != model.position_embedding.shape:
        target_grid = model.patch_embedding.grid_size
        state = dict(state)
        state[position_key] = interpolate_position_embedding(state[position_key], target_grid)
    model.load_state_dict(state, strict=strict)
