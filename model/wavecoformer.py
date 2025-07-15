
from __future__ import annotations
from torchsummary import summary
import torch
from torch import einsum
import torch.nn as nn
import pywt
import numpy as np
from einops import rearrange, repeat
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR
from monai.networks.nets.swin_unetr import SwinTransformer, PatchMerging, PatchMergingV2, window_partition,window_reverse,WindowAttention,SwinTransformerBlock,MERGING_MODE,BasicLayer


import itertools
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, PatchEmbeddingBlock, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock, PatchEmbeddingBlock, Convolution
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from monai.networks.blocks.dynunet_block import get_conv_layer
rearrange, _ = optional_import("einops", name="rearrange")

class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, 
                 in_channels, out_channels, 
                 kernel_size, stride=1, padding='same'):
        super().__init__()

        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size, 
                                   stride=stride, padding=padding, 
                                   groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, 
                                   kernel_size=1)
    def forward(self, x):
        x = self.depthwise(x) #Perform depthwise convolution
        x = self.pointwise(x) #Perform pointwise convolution
        return x
    
class DepthwiseSeparableConv3ddilation(nn.Module):
    def __init__(self, 
                 in_channels, out_channels, 
                 kernel_size, groups,dilation, stride=1, padding='same'):
        super().__init__()

        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size, 
                                   stride=stride, padding=padding, 
                                   groups=groups,dilation=dilation)
        self.pointwise = nn.Conv3d(in_channels, out_channels, 
                                   kernel_size=1)
    def forward(self, x):
        x = self.depthwise(x) #Perform depthwise convolution
        x = self.pointwise(x) #Perform pointwise convolution
        return x
    
class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_conv_1 = nn.Sequential(DepthwiseSeparableConv3d(in_channels=7, 
                                                                   out_channels=16, 
                                                                   kernel_size=3),
                                          nn.InstanceNorm3d(num_features=16),
                                          nn.LeakyReLU(inplace=True))
        self.depth_conv_2 = nn.Sequential(DepthwiseSeparableConv3d(in_channels=16, 
                                                                   out_channels=16, 
                                                                   kernel_size=3),
                                          nn.InstanceNorm3d(num_features=16),
                                          nn.LeakyReLU(inplace=True))
        self.depth_conv_3 = nn.Sequential(DepthwiseSeparableConv3d(in_channels=16, 
                                                                   out_channels=8, 
                                                                   kernel_size=3),
                                          nn.InstanceNorm3d(num_features=8),
                                          nn.LeakyReLU(inplace=True))
        self.dil_conv_1 = nn.Sequential(nn.Conv3d(in_channels=1, 
                                                  out_channels=8, 
                                                  kernel_size=3, 
                                                  dilation=2, 
                                                  padding='same'),
                                        nn.InstanceNorm3d(num_features=8),
                                        nn.LeakyReLU(inplace=True))
        self.dil_conv_2 = nn.Sequential(nn.Conv3d(in_channels=8, 
                                                  out_channels=16, 
                                                  kernel_size=3, 
                                                  dilation=3, 
                                                  padding='same'),
                                        nn.InstanceNorm3d(num_features=16),
                                        nn.LeakyReLU(inplace=True))
        self.dil_conv_3 = nn.Sequential(nn.Conv3d(in_channels=16, 
                                                  out_channels=8, 
                                                  kernel_size=3, 
                                                  dilation=5, 
                                                  padding='same'),
                                        nn.InstanceNorm3d(num_features=8),
                                        nn.LeakyReLU(inplace=True))
        self.point_conv = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=1)
        self.norm = nn.InstanceNorm3d(num_features=8)
        self.relu = nn.LeakyReLU(inplace=True)
        
    def discrete_wavelet_transform(self, x,wavelet): #Perform DWT
        dwt_coeffs = []
        xhat=x.cpu()    
        for image in xhat.detach().numpy():
            coeffs = pywt.dwtn(image[0],wavelet, mode='periodization')
            dwt_coeffs.append(coeffs)
            combined_coeffs = []
            approx_coeffs = []            
            for coeffs in dwt_coeffs:
                for key in coeffs:
                        #print(f"{key}: {coeffs[key].shape}")
                        
                        if key=='aaa':
                            approx_coeffs.append(coeffs[key])
                            #print(type(combined_coeffs))
                            #combined_coeffs=np.array(combined_coeffs)
                        else:
                            combined_coeffs.append(coeffs[key])
        combined_coeffs=torch.tensor(np.array(combined_coeffs))
        approx_coeffs=np.array(approx_coeffs)
        approx_coeffs=torch.from_numpy(approx_coeffs)
        approx_coeffs=torch.unsqueeze(approx_coeffs,dim=1)
        combined_coeffs=combined_coeffs.reshape([x.shape[0],7,coeffs[key].shape[0],coeffs[key].shape[1],coeffs[key].shape[0]])
        approx_coeffs=approx_coeffs.to(x.device)
        combined_coeffs=combined_coeffs.to(x.device)
        return combined_coeffs,approx_coeffs
    def forward(self, x):
        wavelet='haar'
        hf,lf=self.discrete_wavelet_transform(x, wavelet) #High Frequency and low frequency components
        de1=self.depth_conv_1(hf) #Depthwise conv 1 on hf
        de2=self.depth_conv_2(de1) #Depthwise conv 2
        de3=self.depth_conv_3(de2) #Depthwise conv 3
        dil1=self.dil_conv_1(lf)  #Dilated conv 1 on lf rate=2
        dil2=self.dil_conv_2(dil1) #Dilated conv 2 on lf rate=3
        dil3=self.dil_conv_3(dil2) #Dilated conv 3 on lf rate=5
        p1=self.point_conv(lf)   #1x1 conv on lf
        p1=self.norm(p1)  #Instance norm on p1
        p1=self.relu(p1)
        lf_out=p1+dil3 #Adding output of dilated conv with 1x1 conv
        block_out=lf_out+de3 #Adding output of two branches
        return block_out,lf
    
class SalientChannelAmplification(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size_agg,kernel_size=3,ratio=16):
        super().__init__()
        self.depthwise_conv = nn.Sequential(
                                nn.Conv3d(in_channels = in_channels, 
                                    out_channels = in_channels,
                                    kernel_size = kernel_size_agg,
                                    groups = in_channels),
                                    nn.LeakyReLU(inplace=True))
        self.stacked_dil_conv = nn.Sequential(
                                      DepthwiseSeparableConv3ddilation(in_channels=in_channels,
                                                out_channels=in_channels,
                                                kernel_size=3,dilation=2,
                                                groups=in_channels,
                                                padding='same'),
                                       nn.InstanceNorm3d(num_features=in_channels),
                                       nn.LeakyReLU(inplace=True),
                                      DepthwiseSeparableConv3ddilation(in_channels=in_channels,
                                                out_channels=in_channels,
                                                kernel_size=3,dilation=3,
                                                groups=in_channels,padding='same'),
                                       nn.InstanceNorm3d(num_features=in_channels),
                                       nn.LeakyReLU(inplace=True),
                                      DepthwiseSeparableConv3ddilation(in_channels=in_channels,
                                                out_channels=in_channels,
                                                kernel_size=3,dilation=5,
                                                groups=in_channels,
                                                padding='same'),
                                      nn.InstanceNorm3d(num_features=in_channels),
                                       nn.LeakyReLU(inplace=True))
        self.squeeze_n_excite = nn.Sequential(nn.Conv3d(in_channels=in_channels,
                                                        out_channels=in_channels//ratio,
                                                        kernel_size=1),
                                              nn.LeakyReLU(inplace=True),
                                              nn.Conv3d(in_channels=in_channels//ratio,
                                                        out_channels=out_channels,
                                                        kernel_size=1),
                                              nn.Sigmoid())
    def forward(self,x):
        x = self.stacked_dil_conv(x)
        x = self.depthwise_conv(x)
        x = self.squeeze_n_excite(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, in_channels, 
                 in_channels_2, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels=in_channels, 
                                             out_channels=out_channels, 
                                             kernel_size=3,
                                             padding = 'same'),
                                   nn.InstanceNorm3d(num_features=out_channels),
                                   nn.LeakyReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv3d(in_channels=in_channels_2, 
                                             out_channels=out_channels, 
                                             kernel_size=3,
                                             padding = 'same'),
                                   nn.InstanceNorm3d(num_features=out_channels),
                                   nn.LeakyReLU(inplace=True))
        self.relu = nn.ReLU(inplace=True)
        self.weights = nn.Sequential(nn.Conv3d(in_channels=out_channels,
                                    out_channels=1,
                                    kernel_size=1),
                                     nn.Sigmoid())
    def forward(self,x,y):
            x1 = self.conv1(x)
            y1 = self.conv2(y)
            mul_v = x1 * y1
            mul_v = self.relu(mul_v)
            weights = self.weights(mul_v)
            #print(weights.shape)
            
            weighted_x = weights * x
            x1 = self.conv1(weighted_x)
            mul_v = x1 * y1
            mul_v = self.relu(mul_v)
            weights = self.weights(mul_v)
            weighted_y = weights * y
            #print(weighted_y.shape)
            
            return weighted_y

class SwinBlock(nn.Module):
    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        window_size: Sequence[int],
        patch_size: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        use_v2=False,
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            spatial_dims=spatial_dims,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.use_v2 = use_v2
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        self.dwt_conv_block=ConvBlock()
        if self.use_v2:
            self.layers1c = nn.ModuleList()
            self.layers2c = nn.ModuleList()
            self.layers3c = nn.ModuleList()
            self.layers4c = nn.ModuleList()
        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=down_sample_mod,
                use_checkpoint=use_checkpoint,
            )
            if i_layer == 0:
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)
            elif i_layer == 3:
                self.layers4.append(layer)
            if self.use_v2:
                layerc = UnetrBasicBlock(
                    spatial_dims=3,
                    in_channels=embed_dim * 2**i_layer,
                    out_channels=embed_dim * 2**i_layer,
                    kernel_size=3,
                    stride=1,
                    norm_name="instance",
                    res_block=True,
                )
                if i_layer == 0:
                    self.layers1c.append(layerc)
                elif i_layer == 1:
                    self.layers2c.append(layerc)
                elif i_layer == 2:
                    self.layers3c.append(layerc)
                elif i_layer == 3:
                    self.layers4c.append(layerc)

            self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
    
        self.stage1_cross_att = CrossAttention(in_channels=8, 
                                               in_channels_2=96, 
                                               out_channels = 96)
        self.stage2_cross_att = CrossAttention(in_channels=8, 
                                               in_channels_2=192, 
                                               out_channels = 192)
        self.stage3_cross_att = CrossAttention(in_channels=8, 
                                               in_channels_2=384, 
                                               out_channels = 384)
        self.stage4_cross_att = CrossAttention(in_channels=8, 
                                               in_channels_2=768, 
                                               out_channels = 768)
        self.stage_0_conv = Convolution(spatial_dims=3,
                            in_channels=56,
                            out_channels=48,
                            act="leakyrelu",
                            norm="instance")
        self.channel_att_2 = SalientChannelAmplification(in_channels=96,
                                                        out_channels=96,kernel_size_agg=24)
        self.channel_att_3 = SalientChannelAmplification(in_channels=192,
                                                        out_channels=192,kernel_size_agg=12)
        self.channel_att_4 = SalientChannelAmplification(in_channels=384,
                                                        out_channels=384,kernel_size_agg=6)
    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x
    
    def forward(self, x, normalize=True):
        x0 = self.patch_embed(x)
        x0 = self.pos_drop(x0)
        x0_out = self.proj_out(x0, normalize)
        conv_block_out_0, lf_0 = self.dwt_conv_block(x)
        x0_out_c = torch.cat([conv_block_out_0,x0_out],1)
        x0_out_n = self.stage_0_conv(x0_out_c)
        conv_block_out_1, lf_1 = self.dwt_conv_block(lf_0)
        '''
        if self.use_v2:
            x0 = self.layers1c[0](x0.contiguous())
        '''
        x1 = self.layers1[0](x0.contiguous())
        #print(conv_block_out_1.shape,x1.shape)
        #x1_out = self.proj_out(x1, normalize)
        x1_out = self.stage1_cross_att(conv_block_out_1,x1)
        x1_out_n = self.proj_out(x1_out, normalize)
        '''
        if self.use_v2:
            x1 = self.layers2c[0](x1.contiguous())
        '''
        x_c = self.channel_att_2(x1_out_n)
        x1_out = x_c*x1_out_n
        conv_block_out_2, lf_2 = self.dwt_conv_block(lf_1)
        x2 = self.layers2[0](x1_out.contiguous())
        #x2_out = self.proj_out(x2, normalize)
        x2_out = self.stage2_cross_att(conv_block_out_2,x2)
        x2_out_n = self.proj_out(x2_out, normalize)
        '''
        if self.use_v2:
            x2 = self.layers3c[0](x2.contiguous())
        '''
        x_c = self.channel_att_3(x2_out_n)
        x2_out = x_c*x2_out_n
        conv_block_out_3, lf_3 = self.dwt_conv_block(lf_2)
        x3 = self.layers3[0](x2_out.contiguous())
        #x3_out = self.proj_out(x3, normalize)
        x3_out = self.stage3_cross_att(conv_block_out_3,x3)
        x3_out_n = self.proj_out(x3_out, normalize)
        '''
        if self.use_v2:
            x3 = self.layers4c[0](x3.contiguous())
        '''
        x_c = self.channel_att_4(x3_out_n)
        x3_out = x_c*x3_out_n
        x4 = self.layers4[0](x3_out.contiguous())
        x4_out_n = self.proj_out(x4, normalize)
        return [x0_out_n, x1_out_n, x2_out_n, x3_out_n, x4_out_n]
    
class DwtformerUpBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 in_channels_2: int,
                 out_channels: int,
                 upsample_kernel_size: int,
                 spatial_dims: int=3,
                 norm_name: str = "instance"):
        super().__init__()
        self.upsample_stride = upsample_kernel_size
        self.upsample_block = Convolution(spatial_dims=spatial_dims, 
                                          in_channels=in_channels, 
                                          out_channels=out_channels,
                                          is_transposed=True,
                                          strides=2,
                                          )
        self.depth_conv_1 = nn.Sequential(DepthwiseSeparableConv3d(in_channels=out_channels*2, 
                                                                   out_channels=out_channels, 
                                                                   kernel_size=3),
                                          nn.InstanceNorm3d(num_features=out_channels),
                                          nn.LeakyReLU(inplace=True))
        self.depth_conv_2 = nn.Sequential(DepthwiseSeparableConv3d(in_channels=out_channels, 
                                                                   out_channels=out_channels, 
                                                                   kernel_size=3),
                                          nn.InstanceNorm3d(num_features=out_channels),
                                          nn.LeakyReLU(inplace=True))
        self.depth_conv_3 = nn.Sequential(DepthwiseSeparableConv3d(in_channels=out_channels, 
                                                                   out_channels=out_channels, 
                                                                   kernel_size=3),
                                          nn.InstanceNorm3d(num_features=out_channels),
                                          nn.LeakyReLU(inplace=True))
        self.point_conv = nn.Conv3d(in_channels=out_channels*2, out_channels=out_channels, kernel_size=1)
        self.norm = nn.InstanceNorm3d(num_features=out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        
    def forward(self,x,y):
        xu = self.upsample_block(x)
        #print(xu.shape,y.shape)
        cat = torch.cat([xu,y],dim=1)
        x = self.depth_conv_1(cat)
        x = self.depth_conv_2(x)
        x = self.depth_conv_3(x)
        cat = self.point_conv(cat)
        cat = self.norm(cat)
        cat = self.relu(cat)
        out = cat + x
        return out

class Wavecoformer(nn.Module):
    def __init__(
        self,
        img_size: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        use_v2=False,
    ):
        super().__init__()
        self.swinViT=SwinBlock(in_chans=in_channels, 
                      embed_dim=feature_size, 
                      window_size=(7,7,7), 
                      patch_size=(2,2,2), 
                      depths=depths, 
                      num_heads=num_heads)
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.decoder5 = DwtformerUpBlock(in_channels=16 * feature_size, 
                                         in_channels_2=8 * feature_size, 
                                         out_channels=8 * feature_size, 
                                         upsample_kernel_size=2)
        self.decoder4 = DwtformerUpBlock(in_channels=8 * feature_size, 
                                         in_channels_2=4 * feature_size, 
                                         out_channels=4 * feature_size, 
                                         upsample_kernel_size=2)
        self.decoder3 = DwtformerUpBlock(in_channels=4 * feature_size, 
                                         in_channels_2=2 * feature_size, 
                                         out_channels=2 * feature_size, 
                                         upsample_kernel_size=2)
        self.decoder2 = DwtformerUpBlock(in_channels=2 * feature_size, 
                                         in_channels_2=feature_size, 
                                         out_channels=feature_size, 
                                         upsample_kernel_size=2)
        self.decoder1 = DwtformerUpBlock(in_channels=feature_size, 
                                         in_channels_2=feature_size, 
                                         out_channels=feature_size, 
                                         upsample_kernel_size=2)
       

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)


    def load_from(self, weights):
        with torch.no_grad():
            self.swinViT.patch_embed.proj.weight.copy_(weights["state_dict"]["module.patch_embed.proj.weight"])
            self.swinViT.patch_embed.proj.bias.copy_(weights["state_dict"]["module.patch_embed.proj.bias"])
            for bname, block in self.swinViT.layers1[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers1")
            self.swinViT.layers1[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.reduction.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers2[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers2")
            self.swinViT.layers2[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.reduction.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers3[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers3")
            self.swinViT.layers3[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.reduction.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers4[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers4")
            self.swinViT.layers4[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.reduction.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.bias"]
            )

    def forward(self, x_in):
        hidden_states_out = self.swinViT(x_in)
        
        
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        return logits
