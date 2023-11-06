import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
import math
from thop import profile

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
##---------- Adaptive frequency collaborative block-----------------------
class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 16

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features, 1, 1, self.patch_size, self.patch_size // 2 + 1)))

        self.conv1_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1,
                                 groups=hidden_features, bias=bias)

        self.conv3_3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1,
                                 groups=hidden_features, bias=bias)

        self.conv5_5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, padding=2,
                                 groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features * 3, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)

        x1 = self.conv1_1(x)
        x3 = self.conv3_3(x)
        x5 = self.conv5_5(x)
        gate1 = F.gelu(x1) * x3
        gate2 = F.gelu(x1) * x5
        x = self.project_out(torch.cat([gate1, gate2, x1], dim=1))

        return x

##########################################################################
##---------- Frequency domain guidance attention-----------------------
class FSAS(nn.Module):
    def __init__(self, dim, bias):
        super(FSAS, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.patch_size = 16

    def forward(self, x):
        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft

        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output


class DilatedSplitDwConvBlock2(nn.Module):
    def __init__(self, dim, scales=1):
        super().__init__()
        width = max(int(math.ceil(dim / scales)), int(math.floor(dim // scales)))
        self.width = width
        if scales == 1:
            self.nums = 1
        else:
            self.nums = scales - 1
        convs = []
        for i in range(self.nums):
            # print(i, self.nums)
            convs.append(nn.Sequential(
                nn.Conv2d(width, width * 2, 1),
                LayerNorm(width * 2, LayerNorm_type='WithBias'),
                nn.GELU(),
                nn.Conv2d(width * 2, width * 2, kernel_size=3, padding=1, groups=width * 2),
                LayerNorm(width * 2, LayerNorm_type='WithBias'),
                nn.GELU(),
                nn.Conv2d(width * 2, width, 1),
            ))
        self.convs = nn.ModuleList(convs)
        self.fusion = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        _, C, H, W = x.shape

        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        x = torch.cat((out, spx[self.nums]), 1)
        x = self.fusion(x)
        return x


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', att=False, dsdb1=True,
                 scale=1):
        super(TransformerBlock, self).__init__()
        self.norm0 = LayerNorm(dim, LayerNorm_type)
        self.dsdb = DilatedSplitDwConvBlock2(dim=dim, scales=scale)
        self.att = att

        if self.att:
            self.norm1 = LayerNorm(dim, LayerNorm_type)
            self.attn = FSAS(dim, bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)

        self.ffn = DFFN(dim, ffn_expansion_factor, bias)
        self.dsdb1 = dsdb1

    def forward(self, x):
        if self.dsdb1:
            x = x + self.dsdb(self.norm0(x))
        if self.att:
            x = x + self.attn(self.norm1(x))

        x = x + self.ffn(self.norm2(x))
        return x


##########################################################################
##---------- Scale feature enhancement block-----------------------
class Cross_Context_Fusion_Block(nn.Module):
    def __init__(self, in_channels):
        super(Cross_Context_Fusion_Block, self).__init__()
        bias = False
        self.feat_fusion = nn.Conv2d(in_channels * 2, in_channels, 3, stride=1, padding=1)
        self.feat_expand = nn.Conv2d(in_channels, in_channels * 2, 3, stride=1, padding=1)
        self.diff_fusion = nn.Conv2d(in_channels * 2, in_channels, 3, stride=1, padding=1)
        self.encoder1 = nn.Conv2d(in_channels * 2, in_channels * 2, 1)

    def forward(self, x, y):
        feat = self.encoder1(torch.cat([x, y], dim=1))
        fused_feat = self.feat_fusion(feat)
        exp_feat = self.feat_expand(fused_feat)
        residual = exp_feat - feat
        residual = self.diff_fusion(residual)
        fused_feat = fused_feat + residual
        return fused_feat


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x



class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
##---------- EFGformer -----------------------
class fftformer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[6, 6, 12, 8],
                 num_refinement_blocks=4,
                 ffn_expansion_factor=3,
                 scale=[2, 2, 3, 4],
                 bias=False,
                 ):
        super(fftformer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias, scale=scale[0], att=True) for i in
            range(num_blocks[0])])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             scale=scale[1], att=True) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             scale=scale[2], att=True) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))
        self.encoder_level4 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             scale=scale[3], att=True) for i in range(num_blocks[3])])
        self.up4_3 = Upsample(int(dim * 2 ** 3))

        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, scale=scale[2], att=True) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True, scale=scale[1]) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True, scale=scale[0]) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True, scale=scale[0]) for i in range(num_blocks[4])])

        self.fuse3 = Cross_Context_Fusion_Block(dim * 2 ** 2)
        self.fuse2 = Cross_Context_Fusion_Block(dim * 2)
        self.fuse1 = Cross_Context_Fusion_Block(dim)

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_dec_level2 = self.up3_2(out_enc_level3)

        inp_dec_level2 = self.fuse2(inp_dec_level2, out_enc_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = self.fuse1(inp_dec_level1, out_enc_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


model = fftformer(dim=32, num_blocks=[8, 8, 4, 4, 4], scale=[4, 4, 4, 4], ffn_expansion_factor=2.66, bias=False).cuda()
