import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY


class LKConv(nn.Module):
    """Large kernel conv wrapper with name 'conv' (for ckpt compatibility)."""
    def __init__(self, dim: int, kernel_size: int = 13):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(dim, dim, kernel_size, padding=padding, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PLKBlockTiny(nn.Module):
    """
    PLKSR-tiny block (ckpt 구조에 맞춘 버전):
      - channe_mixer: Conv1x1 -> GELU -> Conv1x1
      - lk: LKConv(dim, ks)  (내부 conv 이름: 'conv')
      - refine: Conv1x1(dim -> dim)
      - residual add
    """
    def __init__(
        self,
        dim: int,
        kernel_size: int = 13,
        expansion: float = 2.0,
    ):
        super().__init__()
        hidden_dim = int(dim * expansion)

        # channe_mixer.0 / 1 / 2 로 저장되도록 Sequential 사용
        self.channe_mixer = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True),  # .0.weight/.bias
            nn.GELU(),                                             # .1 (no params)
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True),  # .2.weight/.bias
        )

        # lk.conv.weight / lk.conv.bias
        self.lk = LKConv(dim, kernel_size=kernel_size)

        # refine.weight / refine.bias
        self.refine = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.channe_mixer(x)
        x = self.lk(x)
        x = self.refine(x)
        return x + identity


@ARCH_REGISTRY.register()
class PLKSR(nn.Module):
    """
    PLKSR-tiny X4 (DF2K)와 ckpt 1:1 호환되는 구조.

    ckpt key 예:
      feats.0.weight / bias
      feats.1.channe_mixer.0/2.weight / bias
      feats.1.lk.conv.weight / bias
      feats.1.refine.weight / bias
      ...
      feats.12.channe_mixer...
      feats.13.weight / bias
    """
    def __init__(
        self,
        dim: int = 64,
        n_blocks: int = 12,
        kernel_size: int = 13,
        split_ratio: float = 0.25,   # ckpt에는 직접 쓰이진 않지만 인터페이스 유지용
        ccm_type: str = 'DCCM',
        lk_type: str = 'PLK',
        use_pa: bool = False,
        use_ea: bool = False,
        upscaling_factor: int = 4,
        ffn_expansion: float = 2.0,
        **kwargs,
    ):
        super().__init__()

        assert upscaling_factor == 4, "현재 ckpt는 x4 전용."
        self.scale = upscaling_factor
        self.dim = dim
        self.n_blocks = n_blocks

        # ckpt 구조에 맞게 feats 라는 리스트로 구성
        feats = []

        # feats[0]: 입력 conv (3 -> dim)
        feats.append(
            nn.Conv2d(3, dim, kernel_size=3, padding=1, bias=True)
        )

        # feats[1..n_blocks]: PLKBlockTiny
        for _ in range(n_blocks):
            feats.append(
                PLKBlockTiny(dim=dim, kernel_size=kernel_size, expansion=ffn_expansion)
            )

        # feats[n_blocks+1]: tail conv (dim -> 3 * r^2)
        out_channels = 3 * (self.scale ** 2)
        feats.append(
            nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=True)
        )

        # ModuleList 로 등록 → ckpt의 feats.* 구조와 일치
        self.feats = nn.ModuleList(feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: B x 3 x H x W
        return: B x 3 x (H * r) x (W * r)
        """
        b, c, h, w = x.shape
        r = self.scale

        # feats[0] : 입력 conv
        feat = self.feats[0](x)

        # feats[1..n_blocks] : 블록 반복
        for i in range(1, 1 + self.n_blocks):
            feat = self.feats[i](feat)

        # feats[n_blocks+1] : tail conv
        F_h = self.feats[1 + self.n_blocks](feat)  # B x (3*r^2) x H x W

        # LR repeat branch
        F_l = x.repeat(1, r * r, 1, 1)             # B x (3*r^2) x H x W

        F_sum = F_h + F_l

        # PixelShuffle
        out = F_sum.view(
            b, 3, r, r, h, w
        ).permute(
            0, 1, 4, 2, 5, 3
        ).reshape(
            b, 3, h * r, w * r
        )

        return out
