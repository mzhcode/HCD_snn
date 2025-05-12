import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from functools import partial
from timm.models import create_model
from pa_lif import pa_lif_n

__all__ = ['QKFormer']


class Token_QK_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        # self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.q_lif = pa_lif_n(tau=0.5)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = pa_lif_n(tau=0.5)
        # self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        # self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')
        self.attn_lif = pa_lif_n(tau=0.5, threshold=0.5)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        # self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.proj_lif = pa_lif_n(tau=0.5)

        self.mem_alpha = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.proj_lif(x)
        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        mems = []

        for t in range(T):
            # mems[0] = q[0] ; mems[1] = q[0] ; mems[2] = 0.2*mems[1] + 0.8*q[1]
            if t == 0:
                mems.append(q[0].clone())
            elif t > 0:
                mems_t = self.mem_alpha * mems[t-1] + (1 - self.mem_alpha) * q[t-1].clone()
                mems.append(mems_t)
        
        mems = torch.stack(mems, dim=0)
        q_cat = torch.cat([mems, q], 3)
        # T B head 2C N
        q_cat = torch.sum(q_cat, dim = 3, keepdim = True)
        # T B head 1 N
        attn = self.attn_lif(q_cat)
        x = torch.mul(attn, k)
        # T B head C N

        x = x.flatten(2, 3)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)
        # x = self.proj_lif(x)

        return x


class GRToken_QK_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = pa_lif_n(tau=0.5)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = pa_lif_n(tau=0.5)

        self.attn_lif = pa_lif_n(tau=0.5, threshold=0.5)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = pa_lif_n(tau=0.5)

        self.mem_alpha = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.proj_lif(x)
        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        mems = []

        for t in range(T):
            # mems[0] = q[0] ; mems[1] = q[0] ; mems[2] = 0.2*mems[1] + 0.8*q[1]
            if t == 0:
                mems.append(q[0].clone())
            elif t > 0:
                mems_t = self.mem_alpha * mems[t - 1] + (1 - self.mem_alpha) * q[t - 1].clone()
                mems.append(mems_t)

        mems = torch.stack(mems, dim=0)
        # q_cat = torch.cat([mems, q], 3)
        q_cat = torch.cat([mems, q], 4)
        # T B head C 2N
        q_cat = torch.sum(q_cat, dim=4, keepdim=True)
        # T B head C 1
        attn = self.attn_lif(q_cat)
        x = torch.mul(attn, k)
        # T B head C N

        x = x.flatten(2, 3)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)
        return x


class C_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        # self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.q_lif = pa_lif_n(tau=0.5)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = pa_lif_n(tau=0.5)
        # self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        # self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')
        self.attn_lif = pa_lif_n(tau=0.5, threshold=0.5)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        # self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.proj_lif = pa_lif_n(tau=0.5)

        self.mem_alpha = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.proj_lif(x)
        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        mems = []

        for t in range(T):
            # mems[0] = q[0] ; mems[1] = q[0] ; mems[2] = 0.2*mems[1] + 0.8*q[1]
            if t == 0:
                mems.append(q[0])
            elif t > 0:
                mems_t = q[t] - q[t-1]
                # mems_t = self.mem_alpha * mems[t - 1] + (1 - self.mem_alpha) * q[t - 1].clone()
                mems.append(mems_t)

        mems = torch.stack(mems, dim=0)
        # q_cat = torch.cat([mems, q], 3)
        q_cat = torch.cat([mems, q], 4)
        # T B head C 2N
        q_cat = torch.sum(q_cat, dim=4, keepdim=True)
        # T B head C 1
        attn = self.attn_lif(q_cat)
        x = torch.mul(attn, k)
        # T B head C N

        x = x.flatten(2, 3)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)
        return x


class RSSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        # self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.q_lif = pa_lif_n(tau=0.5)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        # self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.k_lif = pa_lif_n(tau=0.5)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = pa_lif_n(tau=0.5)
        # self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        # self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')
        self.attn_lif = pa_lif_n(tau=0.5, threshold=0.5)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = pa_lif_n(tau=0.5)
        # self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.mems_k = None
        self.mems_v = None
        self.mem_beta = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.proj_lif(x)
        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T,B,C,N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T,B,C,N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T,B,C,N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x = x.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        for t in range(T):
            if t == 0:
                c_k = k[t]
                c_v = v[t]
                self.mems_k = k[t].clone()
                self.mems_v = v[t].clone()
            else:
                c_k = torch.cat([self.mems_k, k[t]], 2)
                c_v = torch.cat([self.mems_v, v[t]], 2)
            self.mems_k = self.mem_beta * self.mems_k + (1-self.mem_beta) * k[t].clone()
            self.mems_v = v[t].clone()
            # self.mems_v = self.mem_beta * self.mems_v + (1 - self.mem_beta) * v[t].clone()
            attn = (q[t] @ c_k.transpose(-2, -1)) * self.scale
            x[t] = attn @ c_v

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_bn(self.proj_conv(x)).reshape(T,B,C,H,W)
        return x


class CSSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = pa_lif_n(tau=0.5)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = pa_lif_n(tau=0.5)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = pa_lif_n(tau=0.5)
        self.attn_lif = pa_lif_n(tau=0.5, threshold=0.5)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = pa_lif_n(tau=0.5)

        self.mems_k = None
        self.mems_v = None
        self.mem_beta = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.proj_lif(x)
        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T,B,C,N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T,B,C,N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T,B,C,N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x = x.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        for t in range(T):
            if t == 0:
                c_k = torch.zeros_like(k[t])
                c_v = torch.zeros_like(v[t])
            else:
                self.mems_k = k[t] - k[t-1]
                self.mems_v = v[t] - v[t-1]
                c_k = torch.cat([self.mems_k, k[t]], 2)
                c_v = torch.cat([self.mems_v, v[t]], 2)
            attn = (q[t] @ c_k.transpose(-2, -1)) * self.scale
            x[t] = attn @ c_v

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_bn(self.proj_conv(x)).reshape(T,B,C,H,W)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)
        self.mlp1_lif = pa_lif_n(tau=0.5)

        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.mlp2_bn = nn.BatchNorm2d(out_features)
        self.mlp2_lif = pa_lif_n(tau=0.5)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.mlp1_lif(x)
        x = self.mlp1_conv(x.flatten(0, 1))
        x = self.mlp1_bn(x).reshape(T, B, self.c_hidden, H, W)

        x = self.mlp2_lif(x)
        x = self.mlp2_conv(x.flatten(0, 1))
        x = self.mlp2_bn(x).reshape(T, B, C, H, W)

        return x


class ChanelSpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.tssa = GRToken_QK_Attention(dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features= dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):

        x = x + self.tssa(x)
        x = x + self.mlp(x)

        return x


class CSpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.tssa = C_Attention(dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features= dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):

        x = x + self.tssa(x)
        x = x + self.mlp(x)

        return x


class TokenSpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.tssa = Token_QK_Attention(dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features= dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):

        x = x + self.tssa(x)
        x = x + self.mlp(x)

        return x


class SpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.ssa = RSSA(dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features= dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):

        x = x + self.ssa(x)
        x = x + self.mlp(x)

        return x


class CSSpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.ssa = CSSA(dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features= dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):

        x = x + self.ssa(x)
        x = x + self.mlp(x)

        return x


class PatchEmbedInit(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        # self.proj_lif0 = pa_lif_n(tau=0.5)
        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif = pa_lif_n(tau=0.5)

        self.proj1_conv = nn.Conv2d(embed_dims // 2, embed_dims // 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims // 1)
        self.proj1_lif = pa_lif_n(tau=0.5)

        self.proj_res_conv = nn.Conv2d(embed_dims//2, embed_dims //1, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        # self.proj_res_lif = pa_lif_n(tau=0.5)

    def forward(self, x):
        T, B, C, H, W = x.shape

        # x = self.proj_lif0(x)
        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W)
        x = self.proj_lif(x).flatten(0, 1)

        x_feat = x
        x = self.proj1_conv(x)
        x = self.proj1_bn(x).reshape(T, B, -1, H, W)
        # x = self.proj1_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H, W).contiguous()
        # x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat # shortcut

        return x


class PatchEmbeddingStage(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj3_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.proj3_lif = pa_lif_n(tau=0.5)

        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)
        self.proj4_lif = pa_lif_n(tau=0.5)

        self.proj_res_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = pa_lif_n(tau=0.5)

    def forward(self, x):
        T, B, C, H, W = x.shape
        # Downsampling + Res
        x_feat = x
        x = self.proj_res_lif(x)
        x = x.flatten(0, 1).contiguous()

        x = self.proj3_conv(x)
        x = self.proj3_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj3_lif(x).flatten(0, 1).contiguous()

        x = self.proj4_conv(x)
        x = self.proj4_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj4_lif(x)

        x_feat = x_feat.flatten(0, 1).contiguous()
        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H, W).contiguous()

        x = x + x_feat

        return x


class spiking_transformer(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[8, 8, 8], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T=4, pretrained_cfg=None,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        num_heads = [8, 8, 8]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed1 = PatchEmbedInit(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims//2)

        stage1 = nn.ModuleList([ChanelSpikingTransformer(
            dim=embed_dims//2, num_heads=num_heads[0], mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(1)])

        self.proj_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims)
        self.proj_lif = pa_lif_n(tau=0.5)

        stage3 = nn.ModuleList([TokenSpikingTransformer(
            dim=embed_dims, num_heads=num_heads[2], mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(1)])

        self.proj_conv2 = nn.Conv2d(embed_dims, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(16)
        self.proj_lif2 = pa_lif_n(tau=0.5)

        self.proj_conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(32)
        self.proj_lif3 = pa_lif_n(tau=0.5)


        setattr(self, f"patch_embed1", patch_embed1)
        setattr(self, f"stage1", stage1)
        setattr(self, f"stage3", stage3)

        self.head = nn.Linear(32, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        stage1 = getattr(self, f"stage1")
        patch_embed1 = getattr(self, f"patch_embed1")
        stage3 = getattr(self, f"stage3")
        T, B, C, H, W = x.shape

        x = patch_embed1(x)

        for blk in stage1:
            x = blk(x)

        x = self.proj_lif(x)
        x = x.flatten(0, 1).contiguous()
        x = self.proj_conv(x)
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()

        for blk in stage3:
            x = blk(x)

        # x = x - torch.roll(x, shifts=-1, dims=0)
        x_fuse = torch.empty_like(x)  # 新张量的形状为 (4, T, N)
        x_fuse[0] = x[0] - x[3]
        x_fuse[1] = x[1] - x[0]
        x_fuse[2] = x[2] - x[1]
        x_fuse[3] = x[3] - x[2]

        x_fuse = self.proj_lif2(x_fuse)
        x_fuse = x_fuse.flatten(0, 1).contiguous()
        x_fuse = self.proj_conv2(x_fuse)
        x_fuse = self.proj_bn2(x_fuse).reshape(T, B, -1, H, W).contiguous()

        x_fuse = self.proj_lif3(x_fuse)
        x_fuse = x_fuse.flatten(0, 1).contiguous()
        x_fuse = self.proj_conv3(x_fuse)
        x_fuse = self.proj_bn3(x_fuse).reshape(T, B, -1, H, W).contiguous()
        # x = self.proj_lif2(x)
        # x = x.flatten(0, 1).contiguous()
        # x = self.proj_conv2(x)
        # x = self.proj_bn2(x).reshape(T, B, -1, H, W).contiguous()
        #
        # x = self.proj_lif3(x)
        # x = x.flatten(0, 1).contiguous()
        # x = self.proj_conv3(x)
        # x = self.proj_bn3(x).reshape(T, B, -1, H, W).contiguous()

        return x_fuse.flatten(3).mean(3)

    def forward(self, x1, x2):
        x = torch.cat((x1.unsqueeze(0).repeat(2, 1, 1, 1), x2.unsqueeze(0).repeat(2, 1, 1, 1)), dim=0)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.reshape(4, x.shape[1], -1, 5, 5)
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x


@register_model
def QKFormer(pretrained=False, **kwargs):
    model = spiking_transformer(
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':
    input = torch.randn(2, 3, 32, 32).cuda()
    model = create_model(
        'QKFormer',
        pretrained=False,
        drop_rate=0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size_h=4, img_size_w=4,
        patch_size=4, embed_dims=384, num_heads=8, mlp_ratios=4,
        in_channels=103, num_classes=9, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=4, sr_ratios=1,
        T=4,
    ).cuda()

    from torchinfo import summary
    summary(model, input_size=(2, 3, 32, 32))