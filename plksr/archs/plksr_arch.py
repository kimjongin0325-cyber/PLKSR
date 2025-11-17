import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY


# -----------------------------
# 1) Channel Mixer (DCCM)
# -----------------------------
class DCCM(nn.Module):
    """Double Channel Mixing with 1x1 conv -> GELU -> 1x1 conv."""
    def __init__(self, dim, expansion=2.0):
        super().__init__()
        hidden_dim = int(dim * expansion)
        self.conv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        return out


# -----------------------------
# 2) Partial Large Kernel Conv (PLK)
# -----------------------------
class PLKConv(nn.Module):
    """
    Partial Large Kernel Conv:
      - split_ratio 비율만큼 채널을 large kernel conv로 처리
      - 나머지 채널은 3x3 conv로 처리
      - 이후 concat + 1x1 mixing
    """
    def __init__(self, dim, kernel_size=13, split_ratio=0.25):
        super().__init__()
        self.dim = dim
        self.split_ch = int(dim * split_ratio)
        self.remain_ch = dim - self.split_ch

        padding_lk = kernel_size // 2

        # Large kernel conv on a subset of channels
        if self.split_ch > 0:
            self.large_conv = nn.Conv2d(
                self.split_ch, self.split_ch,
                kernel_size=kernel_size,
                padding=padding_lk,
                bias=True
            )
        else:
            self.large_conv = None

        # Local 3x3 conv on remaining channels
        if self.remain_ch > 0:
            self.local_conv = nn.Conv2d(
                self.remain_ch, self.remain_ch,
                kernel_size=3,
                padding=1,
                bias=True
            )
        else:
            self.local_conv = None

        # Channel mixing after concatenation
        self.mix = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x):
        if self.split_ch == 0:
            out = self.local_conv(x)
        elif self.remain_ch == 0:
            out = self.large_conv(x)
        else:
            x_lk, x_local = torch.split(
                x, [self.split_ch, self.remain_ch], dim=1
            )
            if self.large_conv is not None:
                x_lk = self.large_conv(x_lk)
            if self.local_conv is not None:
                x_local = self.local_conv(x_local)
            out = torch.cat([x_lk, x_local], dim=1)

        out = self.mix(out)
        return out


# -----------------------------
# 3) (옵션) Element-wise / Pixel Attention
# -----------------------------
class ElementWiseAttention(nn.Module):
    """Per-element (per-pixel, per-channel) attention."""
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden = max(dim // reduction, 8)
        self.conv1 = nn.Conv2d(dim, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden, dim, kernel_size=1, bias=True)

    def forward(self, x):
        w = self.conv1(x)
        w = self.act(w)
        w = self.conv2(w)
        w = torch.sigmoid(w)
        return x * w


class PixelAttention(nn.Module):
    """Per-pixel scalar attention (PA)."""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, 1, kernel_size=1, bias=True)

    def forward(self, x):
        w = torch.sigmoid(self.conv(x))
        return x * w


# -----------------------------
# 4) PLK Block
# -----------------------------
class PLKBlock(nn.Module):
    """
    하나의 PLKSR block:
      x -> DCCM -> PLKConv -> (EA/PA) -> 1x1 Conv -> residual add
    """
    def __init__(
        self,
        dim,
        kernel_size=13,
        split_ratio=0.25,
        ffn_expansion=2.0,
        use_ea=False,
        use_pa=False,
    ):
        super().__init__()

        self.ccm = DCCM(dim, expansion=ffn_expansion)
        self.plk = PLKConv(dim, kernel_size=kernel_size, split_ratio=split_ratio)

        self.use_ea = use_ea
        self.use_pa = use_pa

        if use_ea:
            self.ea = ElementWiseAttention(dim)
        else:
            self.ea = nn.Identity()

        if use_pa:
            self.pa = PixelAttention(dim)
        else:
            self.pa = nn.Identity()

        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x):
        identity = x
        x = self.ccm(x)
        x = self.plk(x)
        x = self.ea(x)
        x = self.pa(x)
        x = self.proj(x)
        return x + identity


# -----------------------------
# 5) PLKSR Network (tiny X4 config 호환)
# -----------------------------
@ARCH_REGISTRY.register()
class PLKSR(nn.Module):
    """
    PLKSR network (BasicSR/PLKSR yaml과 호환되는 인터페이스)

    yaml 예시:
      network_g:
        type: PLKSR
        dim: 64
        n_blocks: 12
        kernel_size: 13
        split_ratio: 0.25
        ccm_type: DCCM
        lk_type: PLK
        use_pa: false
        use_ea: false
        upscaling_factor: 4
    """
    def __init__(
        self,
        dim=64,
        n_blocks=12,
        kernel_size=13,
        split_ratio=0.25,
        ccm_type='DCCM',
        lk_type='PLK',
        use_pa=False,
        use_ea=False,
        upscaling_factor=4,
        ffn_expansion=2.0,
        **kwargs
    ):
        super().__init__()

        # yaml 호환용 (현재 구현은 DCCM + PLK만 지원)
        assert ccm_type in ['DCCM'], f'Unsupported ccm_type: {ccm_type}'
        assert lk_type in ['PLK'], f'Unsupported lk_type: {lk_type}'
        assert upscaling_factor in [2, 3, 4], 'upscaling_factor must be 2, 3 or 4.'

        self.scale = upscaling_factor
        self.dim = dim

        # 1) 입력 컨볼루션 (3 -> dim)
        self.head = nn.Conv2d(3, dim, kernel_size=3, padding=1, bias=True)

        # 2) PLK blocks
        blocks = []
        for _ in range(n_blocks):
            blocks.append(
                PLKBlock(
                    dim=dim,
                    kernel_size=kernel_size,
                    split_ratio=split_ratio,
                    ffn_expansion=ffn_expansion,
                    use_ea=use_ea,
                    use_pa=use_pa,
                )
            )
        self.body = nn.Sequential(*blocks)

        # 3) high-frequency branch: feature -> 3 * r^2 채널
        out_channels = 3 * (self.scale ** 2)
        self.tail = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        """
        x: B x 3 x H x W
        return: B x 3 x (H * r) x (W * r)
        """
        b, c, h, w = x.shape
        r = self.scale

        # Feature path
        feat = self.head(x)
        feat = self.body(feat)
        F_h = self.tail(feat)                  # B x (3 * r^2) x H x W

        # LR repeat branch (F_l)
        F_l = x.repeat(1, r * r, 1, 1)         # B x (3 * r^2) x H x W

        # Merge & PixelShuffle
        F_sum = F_h + F_l

        # ONNX/TensorRT 호환을 위해 F.pixel_shuffle 대신 수동 구현 가능
        # 여기서는 nn.functional.pixel_shuffle 사용
        out = F.pixel_shuffle(F_sum, r)        # B x 3 x (H * r) x (W * r)

        return out
