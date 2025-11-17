import torch
import torch.nn as nn
import torch.nn.functional as F


class LKConv(nn.Module):
    """
    Large-kernel conv wrapper.
    - ckpt 키: feats.N.lk.conv.weight / bias
    """

    def __init__(self, dim: int, kernel_size: int = 13):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(dim, dim, kernel_size, padding=padding, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PLKBlockTiny(nn.Module):
    """
    PLKSR-tiny block (PLKSR_tiny_X4_DF2K.pth 의 구조와 매칭):

      channe_mixer: Conv1x1 -> GELU -> Conv1x1
        - feats.N.channe_mixer.0.weight / bias
        - feats.N.channe_mixer.2.weight / bias

      lk: LKConv(dim, ks)
        - feats.N.lk.conv.weight / bias

      refine: Conv1x1(dim -> dim)
        - feats.N.refine.weight / bias

      + residual add
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
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True),  # .0
            nn.GELU(),                                             # .1 (no params)
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True),  # .2
        )

        # lk.conv.weight / bias
        self.lk = LKConv(dim, kernel_size=kernel_size)

        # refine.weight / bias
        self.refine = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.channe_mixer(x)
        x = self.lk(x)
        x = self.refine(x)
        return x + identity


class PLKSR_TINY_X4(nn.Module):
    """
    PLKSR_tiny_X4_DF2K.pth 와 1:1로 맞는 Tiny X4 모델.

    ckpt 키 형태 예:
      - feats.0.weight / bias              (input conv)
      - feats.1.channe_mixer.0/2.weight    (tiny block)
      - feats.1.lk.conv.weight / bias
      - feats.1.refine.weight / bias
      ...
      - feats.12.channe_mixer...
      - feats.13.weight / bias             (tail conv)

    입력:  B x 3 x H x W
    출력: B x 3 x (H*4) x (W*4)
    """

    def __init__(
        self,
        dim: int = 64,
        n_blocks: int = 12,
        kernel_size: int = 13,
        upscaling_factor: int = 4,
        ffn_expansion: float = 2.0,
    ):
        super().__init__()

        assert upscaling_factor == 4, "이 Tiny 가중치는 x4 전용입니다."
        self.scale = upscaling_factor
        self.dim = dim
        self.n_blocks = n_blocks

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

        # ckpt 키와 매칭되는 ModuleList
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

        # LR repeat branch (ckpt에서 F_l는 입력을 r^2만큼 반복한 형태로 가정)
        F_l = x.repeat(1, r * r, 1, 1)             # B x (3*r^2) x H x W

        F_sum = F_h + F_l

        # PixelShuffle
        out = F.pixel_shuffle(F_sum, r)

        return out


# 사용 편의를 위해 짧은 이름 alias 도 하나 만들어 둠
PLKSR_TINY = PLKSR_TINY_X4
