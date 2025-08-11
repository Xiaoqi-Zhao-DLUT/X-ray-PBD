# !/usr/bin/python3
# coding=utf-8
import os.path
import torch.nn as nn
import torch.nn.functional as F


from model.MS_module import ASPP
from model.dynamic_conv import DynamicConv2d

import math
import torch
import timm
from einops import rearrange, repeat
# from csrc import selective_scan_cuda
import selective_scan_cuda
from thop import profile


class Region_seg(nn.Module):
    def __init__(self):
        super(Region_seg, self).__init__()
        self.bkbone = timm.create_model('resnet50d', features_only=True, pretrained=True)
        ###############################Transition Layer########################################
        self.dem1 = ASPP(2048,64)
        # self.dem1 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem2 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem3 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem4 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        ################################FPN branch#######################################
        self.output1_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output2_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output3_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output4_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output5_1 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))

    def forward(self, x):
        input = x
        B, _, _, _ = input.size()
        E1, E2, E3, E4, E5 = self.bkbone(x)
        ################################Transition Layer#######################################
        T5 = self.dem1(E5)
        T4 = self.dem2(E4)
        T3 = self.dem3(E3)
        T2 = self.dem4(E2)
        T1 = self.dem5(E1)
        ################################Decoder Layer#######################################
        D4_1 = self.output2_1(F.upsample(T5, size=E4.size()[2:], mode='bilinear')+T4)
        D3_1 = self.output3_1(F.upsample(D4_1, size=E3.size()[2:], mode='bilinear')+T3)
        D2_1 = self.output4_1(F.upsample(D3_1, size=E2.size()[2:], mode='bilinear')+T2)
        D1_1 = self.output5_1(F.upsample(D2_1, size=E1.size()[2:], mode='bilinear')+T1)
        ################################Gated Parallel&Dual branch residual fuse#######################################
        output_fpn_p = F.upsample(D1_1, size=input.size()[2:], mode='bilinear')
        #######################################################################
        if self.training:
            return output_fpn_p
        # return F.sigmoid(output_fpn_p)
        return output_fpn_p



def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
    dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
    # Initialize special dt projection to preserve variance at initialization
    dt_init_std = dt_rank**-0.5 * dt_scale
    if dt_init == "constant":
        nn.init.constant_(dt_proj.weight, dt_init_std)
    elif dt_init == "random":
        nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
    else:
        raise NotImplementedError

    # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
    dt = torch.exp(
        torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    ).clamp(min=dt_init_floor)
    inv_dt = dt + torch.log(-torch.expm1(-dt))
    with torch.no_grad():
        dt_proj.bias.copy_(inv_dt)
    return dt_proj



class ShufflePatch(nn.Module):
    def __init__(self, patch_size, factor, threshold=0.4):
        super().__init__()
        self.factor = factor
        self.patch_size = patch_size * factor
        self.maxpool2d = nn.MaxPool2d(kernel_size=self.patch_size, stride=self.patch_size)
        self.threshold = threshold


    def forward(self, x):
        x = self.maxpool2d(x)
        B, _, H, W = x.shape
        mask = torch.ones((B, H, W))
        x[x > self.threshold] = 1
        x[x != 1] = 0
        mask[x[:,0,:,:] == 1] = 0
        mask[x[:,1,:,:] == 1] = 2
        mask = mask.reshape(B, -1)
        index_mapping = torch.argsort(mask, dim=-1, descending=False)
        indices = torch.arange(mask.size(-1)).unsqueeze(0).expand_as(mask)
        inverse_mapping = torch.zeros_like(indices)
        inverse_mapping = inverse_mapping.scatter_(-1, index_mapping, indices)
        batch_indices = torch.arange(B).view(-1, 1).expand_as(index_mapping)  # 批次索引
        return batch_indices, index_mapping, inverse_mapping


class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if not ctx.has_z:
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        # Here we just pass in None and dz will be allocated in the C++ code.
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
            False  # option to recompute out_z, not used here
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,
                None)


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                     return_last_state=False):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()  
        self.eps = eps  
        self.weight = nn.Parameter(torch.ones(d_model))  

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight  
        return output 



class MambaBlock(nn.Module):
    def __init__(self,
                patch_size,
                in_chans,
                expand: int = 2,
                d_state: int = 4,
                dt_rank: int = None,
                d_conv: int = 4,
                conv_bias: bool=True,
                bias: bool=False, 
                nums: int=1,
                dynamic: bool=True,
                ):
        super().__init__()
        
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.expand = expand
        self.d_model = in_chans * patch_size * patch_size
        self.d_inner = self.d_model * self.expand
        self.d_state = d_state
        self.dt_rank = int(dt_rank if(dt_rank is not None) else self.d_model / 16)
        self.d_conv = d_conv
        self.conv_bias = conv_bias
        self.bias = bias
        

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=self.bias)
        self.pin_proj = nn.Conv2d(self.d_model, self.d_inner, kernel_size=1, bias=self.bias)if dynamic else None
        
        self.shufflepatch = ShufflePatch(patch_size=patch_size, factor=1, threshold=0.5)
        
        self.conv2d = DynamicConv2d(
            in_planes=self.d_inner,
            out_planes=self.d_inner,
            groups=self.d_inner,
            bias=self.conv_bias,
            kernel_size=self.d_conv, 
            ratio=0.25, 
            padding=(self.d_conv - 1) // 2,
        ) if dynamic else nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=self.conv_bias,
            kernel_size=self.d_conv,
            padding=(self.d_conv - 1) // 2,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
    
        self.nums = nums
        dt_projs = [
            dt_init(self.dt_rank, self.d_inner)
            for _ in range(self.nums)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))  # (K, inner)
        del dt_projs
        # n->dxn
        A = repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=self.d_inner).contiguous()
        A_log = torch.log(A)

        A_log = A_log[None].repeat(self.nums, 1, 1).contiguous()
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        
        D = torch.ones(self.d_inner)
        D = D[None].repeat(self.nums, 1).contiguous()
        self.D = nn.Parameter(D)
        self.D._no_weight_decay = True

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=self.bias)
        

    def forward(self, hidden_states, course, px):
        
        # [B, h, w, embed_dim]]
        hidden_states = hidden_states.permute(0, 2, 3, 1)
        # x:(b, l, d)
        
        (b, h, w, d) = hidden_states.shape 
        seqlen = h * w

        x = self.in_proj(hidden_states)
        
        As = -torch.exp(self.A_log.float())
        dt_projs_bias = self.dt_projs_bias.float()
        Ds = self.D.float()
        
        xs = x.permute(0, 3, 1, 2)
        
        if(px == None):
            xs = self.conv2d(xs)
        else:
            px = self.pin_proj(px)
            xs = self.conv2d(xs, px)
        xs = F.silu(xs)
        
        
        xs = xs.permute(0, 2, 3, 1).view(b, seqlen, -1)

        if course is not None:
            batch_indices, index_mapping, inverse_mapping = self.shufflepatch(course)

            max_idx = xs.size(1) - 1
            index_mapping = torch.clamp(index_mapping, 0, max_idx)
            inverse_mapping = torch.clamp(inverse_mapping, 0, max_idx)
            xs = xs[batch_indices, index_mapping]
        
        xs = xs.unsqueeze(1)

        x_dbl = self.x_proj(xs)  # (b, k, l, d)
        dt, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = dt.permute(0, 1, 3, 2).contiguous()
        dt = torch.einsum("b k r l, k d r -> b k d l", dt, self.dt_projs_weight)
        xs = xs.permute(0, 1, 3, 2).contiguous()
        Bs = Bs.permute(0, 1, 3, 2).contiguous()
        Cs = Cs.permute(0, 1, 3, 2).contiguous()
        out_y = []
        
        
        for i in range(self.nums):
            yi = selective_scan_fn(
                    xs[:, i],
                    dt[:, i],
                    As[i],
                    Bs[:, i],
                    Cs[:, i],
                    Ds[i],
                    z=None,
                    delta_bias=dt_projs_bias[i],
                    delta_softplus=True,
            )
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        
        out_y = out_y.transpose(dim0=-2, dim1=-1).contiguous()
        y = out_y[:, 0]
        # (B, K, L, C)
        if(course!=None):
            y = y[batch_indices, inverse_mapping]
        
         # (B, L, C)
        y = self.out_norm(y).view(b, h, w, -1)
        out = self.out_proj(y)
        out = out.permute(0, 3, 1, 2)
        
        return out


class ResidualBlock(nn.Module):
    def __init__(self, 
                patch_size, 
                in_chans,
                expand: int = 2,
                d_state: int = 4,
                dt_rank: int = None,
                d_conv: int = 4,
                conv_bias: bool=True,
                bias: bool=False, 
                nums: int=1,
                dynamic: bool=True,
                ):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.expand = expand
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_bias = conv_bias
        self.bias = bias
        self.d_model = self.in_chans * self.patch_size * self.patch_size
        self.dt_rank = dt_rank if(dt_rank is not None) else self.d_model / 16
        self.nums = nums
        self.dynamic = dynamic
        self.mixer = MambaBlock(patch_size=self.patch_size,
                                in_chans=self.in_chans,
                                expand = self.expand,
                                d_state = self.d_state,
                                dt_rank = self.dt_rank,
                                d_conv = self.d_conv,
                                conv_bias = self.conv_bias,
                                bias = self.bias,
                                nums = self.nums,
                                dynamic = self.dynamic,
                                )
        self.norm = nn.BatchNorm2d(self.in_chans*self.patch_size*self.patch_size)
        

    def forward(self, x, coarse, px):
        # [B, embed_dim, h, w]
        output = self.mixer(self.norm(x), coarse, px) + x
        return output


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=8, in_chans=64):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = in_chans * patch_size * patch_size
        self.proj = nn.Conv2d(in_chans, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        return x   


class Mamba(nn.Module):
    def __init__(self,
                n_layer: int = 1,
                patch_size: int = 16,
                in_chans:int = 64,
                # vocab_size: int,
                d_state: int = 4,
                expand: int = 1,
                dt_rank: int = None,
                d_conv: int = 4,
                conv_bias: bool = True,
                bias: bool = False,
                nums: int = 1,
                dynamic: bool = True,
                ):
        super().__init__()
        self.n_layer = n_layer
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.expand = expand
        self.d_model = in_chans * patch_size * patch_size
        self.d_state = d_state
        self.dt_rank = dt_rank if(dt_rank is not None) else self.d_model / 16
        self.d_conv = d_conv
        self.conv_bias = conv_bias
        self.bias = bias
        self.nums = nums
        self.dynamic = dynamic
        self.patch_embedding = PatchEmbedding(patch_size = patch_size, in_chans = in_chans)

        self.layers = nn.ModuleList([ResidualBlock(patch_size=self.patch_size,
                                                   in_chans=self.in_chans,
                                                   expand=self.expand,
                                                   d_state=self.d_state,
                                                   dt_rank=self.dt_rank,
                                                   d_conv=self.d_conv,
                                                   conv_bias=self.conv_bias,
                                                   bias=self.bias,
                                                   nums=self.nums,
                                                   dynamic = self.dynamic,
                                                ) for _ in range(self.n_layer)])
        self.norm_f = RMSNorm(self.d_model)

    def forward(self, x, coarse, px):
        B, _, H, W = x.shape
        x = self.patch_embedding(x)
        if(px!=None):
            px = self.patch_embedding(px)

        for layer in self.layers:
            x = layer(x, coarse, px)
        x = x.reshape(B, self.d_model, -1).permute(0, 2, 1)
        x = self.norm_f(x)
        x = F.fold(x.transpose(1, 2), output_size=(H, W), kernel_size=self.patch_size, stride=self.patch_size)
        return x
        






# ------ point_prediction ------#
class Encoder(nn.Module):
    def __init__(self, in_channel=64, pre_path=None):
        super(Encoder, self).__init__()
        
        self.bkbone = timm.create_model('resnet50d', features_only=True, pretrained=True)
        
        # ------Transition Layer------
        self.aspp_x = ASPP(2048, 64)
        self.dem4 = nn.Sequential(nn.Conv2d(1024, in_channel, kernel_size=3, padding=1), nn.BatchNorm2d(in_channel), nn.ReLU(inplace=True))
        self.dem3 = nn.Sequential(nn.Conv2d(512, in_channel, kernel_size=3, padding=1), nn.BatchNorm2d(in_channel), nn.ReLU(inplace=True))
        self.dem2 = nn.Sequential(nn.Conv2d(256, in_channel, kernel_size=3, padding=1), nn.BatchNorm2d(in_channel), nn.ReLU(inplace=True))
        self.dem1 = nn.Sequential(nn.Conv2d(64, in_channel, kernel_size=3, padding=1), nn.BatchNorm2d(in_channel), nn.ReLU(inplace=True))

        self.aspp_p = ASPP(2048, 64)
        self.dem4_p = nn.Sequential(nn.Conv2d(1024, in_channel, kernel_size=3, padding=1), nn.BatchNorm2d(in_channel), nn.ReLU(inplace=True))
        self.dem3_p = nn.Sequential(nn.Conv2d(512, in_channel, kernel_size=3, padding=1), nn.BatchNorm2d(in_channel), nn.ReLU(inplace=True))
        self.dem2_p = nn.Sequential(nn.Conv2d(256, in_channel, kernel_size=3, padding=1), nn.BatchNorm2d(in_channel), nn.ReLU(inplace=True))
        self.dem1_p = nn.Sequential(nn.Conv2d(64, in_channel, kernel_size=3, padding=1), nn.BatchNorm2d(in_channel), nn.ReLU(inplace=True))
        

    def forward(self, x, prompt):
        
        e1, e2, e3, e4, e5 = self.bkbone(x)
        p1, p2, p3, p4, p5 = self.bkbone(prompt)
        
        # ------Transition Layer------
        e5 = self.aspp_x(e5)
        e4 = self.dem4(e4)
        e3 = self.dem3(e3)
        e2 = self.dem2(e2)
        e1 = self.dem1(e1)
        p5 = self.aspp_p(p5)
        p4 = self.dem4_p(p4)
        p3 = self.dem3_p(p3)
        p2 = self.dem2_p(p2)
        p1 = self.dem1_p(p1)
        
        # ------point_prediction------
        return e1, e2, e3, e4, e5, p1, p2, p3, p4, p5


# ------ counting_prediction ------#
class Counting_Predictor(nn.Module):
    def __init__(self):
        super(Counting_Predictor, self).__init__()
        self.regressor_fcn_neg = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, padding=0),
                                               nn.BatchNorm2d(64),
                                               nn.ReLU(inplace=True))
        self.regressor_fcn_pos = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, padding=0),
                                               nn.BatchNorm2d(64),
                                               nn.ReLU(inplace=True))
        self.regressor_neg = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1, padding=0), nn.ReLU(inplace=True))
        self.regressor_pos = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1, padding=0), nn.ReLU(inplace=True))

    def forward(self, neg_input, pos_input, t5):
        regression_features_neg = self.regressor_fcn_neg(
            F.adaptive_avg_pool2d(t5 * F.interpolate(neg_input, size=t5.size()[2:], mode='bilinear'), 1))
        regression_features_pos = self.regressor_fcn_pos(
            F.adaptive_avg_pool2d(t5 * F.interpolate(pos_input, size=t5.size()[2:], mode='bilinear'), 1))
        regression_neg = self.regressor_neg(regression_features_neg)
        regression_pos = self.regressor_pos(regression_features_pos)
        regression_neg = torch.squeeze(regression_neg)
        regression_pos = torch.squeeze(regression_pos)
        return regression_neg, regression_pos


class Line_Predictor(nn.Module):
    def __init__(self):
        super(Line_Predictor, self).__init__()
        self.T2_T1_fusion = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(inplace=True))
        self.neg_line_pre = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))
        self.pos_line_pre = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))

    def forward(self, neg_map, pos_map, T1, T2, x):
        T2_1 = self.T2_T1_fusion(F.interpolate(T2, size=T1.size()[2:], mode='bilinear') + T1)
        T2_1 = F.interpolate(T2_1, size=x.size()[2:], mode='bilinear')
        line_neg = self.neg_line_pre(T2_1 * neg_map + T2_1)
        line_pos = self.pos_line_pre(T2_1 * pos_map + T2_1)
        return line_neg, line_pos


class Point_Predictor(nn.Module):
    def __init__(self):
        super(Point_Predictor, self).__init__()

        self.output2_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output3_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output4_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output5_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.fusion_shallow = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.course_output = nn.Sequential(nn.Conv2d(64, 2, kernel_size=3, padding=1))
    def forward(self, f1,f2,f3,f4,f5,shallowfeature):

        D4_1 = self.output2_1(F.upsample(f5, size=f4.size()[2:], mode='bilinear')+f4)
        D3_1 = self.output3_1(F.upsample(D4_1, size=f3.size()[2:], mode='bilinear')+f3)
        D2_1 = self.output4_1(F.upsample(D3_1, size=f2.size()[2:], mode='bilinear')+f2)
        D1_1 = self.output5_1(F.upsample(D2_1, size=f1.size()[2:], mode='bilinear')+f1)
        d0 = self.fusion_shallow(F.interpolate(D1_1, size=shallowfeature.size()[2:], mode='bilinear') + shallowfeature)
        point_course = self.course_output(d0)
        
    
        return point_course,d0



class PFSM(nn.Module):
    def __init__(self, in_channel=64, n_layer=1, patch_size=2):
        super(PFSM, self).__init__()        
        self.mamba5 = Mamba(n_layer=n_layer, patch_size=patch_size, in_chans=in_channel, d_conv=3, nums=1)
        self.mamba4 = Mamba(n_layer=n_layer, patch_size=patch_size, in_chans=in_channel, d_conv=3, nums=1)
        self.mamba3 = Mamba(n_layer=n_layer, patch_size=patch_size, in_chans=in_channel, d_conv=3, nums=1)
        self.mamba2 = Mamba(n_layer=n_layer, patch_size=patch_size*2, in_chans=in_channel, d_conv=5, nums=1)
        self.mamba1 = Mamba(n_layer=n_layer, patch_size=patch_size*2, in_chans=in_channel, d_conv=5, nums=1)
    def forward(self, e1, e2, e3, e4, e5, p1, p2, p3, p4, p5):

        
        f5 = self.mamba5(e5, None, p5)
        f4 = self.mamba4(e4, None, p4)
        f3 = self.mamba3(e3, None, p3)
        f2 = self.mamba2(e2, None, p2)
        f1 = self.mamba1(e1, None, p1)

        return f1, f2, f3, f4, f5


# ------ MDCNeXt_point_line_counting ------#
class MDCNeXt(nn.Module):
    def __init__(self, pre_path=None):
        super(MDCNeXt, self).__init__()
        self.encoder = Encoder(64, pre_path)

        self.shallow_x = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.point_predictor = Point_Predictor()
        self.counting_predictor = Counting_Predictor()
        self.line_predictor = Line_Predictor()

        self.darsm = Mamba(n_layer=1, patch_size=4, in_chans=64, d_conv=5, dynamic=False)
        self.pfsm = PFSM()
        self.point_refine = nn.Sequential(nn.Conv2d(64, 2, kernel_size=3, padding=1))


    def forward(self, x, prompt):
        shallow_x = self.shallow_x(x)
        e1, e2, e3, e4, e5, p1, p2, p3, p4, p5 = self.encoder(x, prompt)
        f1,f2,f3,f4,f5 = self.pfsm(e1, e2, e3, e4, e5, p1, p2, p3, p4, p5)
        point_course,d0  = self.point_predictor(f1,f2,f3,f4,f5,shallow_x)
        point_refine = self.point_refine(self.darsm(F.interpolate(d0, size=point_course.size()[2:], mode='bilinear'), point_course, None)+shallow_x)


        if self.training:
            neg_map = torch.sigmoid(point_course[:, 0, :, :].unsqueeze(1))
            pos_map = torch.sigmoid(point_course[:, 1, :, :].unsqueeze(1))
            regression_neg, regression_pos = self.counting_predictor(neg_map, pos_map, f5)
            line_neg, line_pos = self.line_predictor(neg_map, pos_map, f1, f2, x)
            return point_refine, point_course, regression_neg, regression_pos, line_neg, line_pos
            
        return point_refine
        


if __name__ == "__main__":

    model = MDCNeXt().cuda()
    input = torch.randn(1, 3, 352, 352).cuda()
    anchor_input = torch.randn(1, 3, 352, 352).cuda()
    flops, params = profile(model, inputs=(input,anchor_input), verbose=False)
    print(f"🚀 Parameters: {params / 1e6:.2f} M")
    print(f"💡 FLOPs: {flops / 1e9:.2f} GFLOPs")



