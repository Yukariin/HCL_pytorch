from typing import Mapping, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from modules import ConvNormAct, ResBlock, Downsample, PatchResizing2d, Upsample, GatedConv2d
from modules import confidence_mask_hierarchy
from swin import BasicLayer2d


class Encoder(nn.Module):
    def __init__(self,
                 img_size: int = 256,
                 dim: int = 64,
                 n_conv_stages: int = 0,
                 dim_mults: list[int] = [1, 2, 4],
                 depths: list[int] = [6, 4, 2],

                 window_size: int = 8,
                 legacy_v: int = 4
    ):
        super().__init__()

        assert len(dim_mults) == len(depths)

        self.n_stages = len(dim_mults)
        self.dims = [dim * dim_mults[i] for i in range(self.n_stages)]
        self.n_heads = dim_mults
        res = img_size

        # first conv
        self.first_conv = ConvNormAct(3, dim, kernel_size=5, stride=1, padding=2, activation='gelu')

        # convolution stages
        self.conv_down_blocks = nn.ModuleList([])
        for i in range(n_conv_stages):
            if legacy_v != 3:
                self.conv_down_blocks.append(nn.Sequential(
                    Downsample(dim, dim),
                    nn.GELU(),
                ))
            else:
                self.conv_down_blocks.append(nn.Sequential(
                    nn.GELU(),
                    Downsample(dim, dim),
                ))
            res = res // 2
        
        # transformer stages
        self.down_blocks = nn.ModuleList([])
        for i in range(self.n_stages):
            self.down_blocks.append(nn.ModuleList([
                BasicLayer2d(
                    dim=self.dims[i],
                    input_resolution=(res, res),
                    depth=depths[i],
                    num_heads=self.n_heads[i],
                    window_size=window_size,
                ),
                PatchResizing2d(
                    in_channels=self.dims[i],
                    out_channels=self.dims[i+1],
                    down=True,
                ) if i < self.n_stages - 1 else nn.Identity(),
            ]))
            if i < self.n_stages - 1:
                res = res // 2
    
    def forward(self, X: Tensor):
        X = self.first_conv(X)

        for blk in self.conv_down_blocks:
            X = blk(X)

        projs, skips = [], []
        for blk, down in self.down_blocks:
            X = blk(X)
            skips.append(X)
            X = down(X)

        return X, skips
    

class ProjectHeads(nn.Module):
    def __init__(self,
                 dim: int = 64,
                 dim_mults: list[int] = [1, 2, 4],
                 proj_dim: int = 128,
                 fuse: bool = True):
        super().__init__()

        self.n_stages = len(dim_mults)
        self.dims = [dim * dim_mults[i] for i in range(self.n_stages)] + [0]

        self.proj_heads = nn.ModuleList([])
        for i in range(self.n_stages):
            in_dim = (self.dims[i] + self.dims[i+1] // 4) if fuse else self.dims[i]
            self.proj_heads.append(nn.Sequential(
                nn.Conv2d(in_dim, proj_dim * 2, kernel_size=1, stride=1),
                nn.GELU(),
                nn.Conv2d(proj_dim * 2, proj_dim, kernel_size=1, stride=1),
            ))
        
        self.fuse = None
        if fuse is True:
            self.fuse = nn.ModuleList([])
            for i in range(1, self.n_stages):
                self.fuse.append(nn.Sequential(
                    nn.Conv2d(self.dims[i], self.dims[i] // 4, kernel_size=1),
                    nn.UpsamplingNearest2d(scale_factor=2),
                ))
    
    def forward(self, features: list[Tensor]):
        projs = []
        for i in range(self.n_stages-1, -1, -1):
            if self.fuse is not None and i < self.n_stages - 1:
                concatX = torch.cat((features[i], self.fuse[i](features[i+1])), dim=1)
            else:
                concatX = features[i]
            projX = self.proj_heads[i](concatX)
            projX = F.normalize(projX, dim=1)
            projs.append(projX)
        projs = list(reversed(projs))
        return projs


class Bottleneck(nn.Module):
    def __init__(
            self,
            img_size: int = 256,
            dim: int = 64,
            n_conv_stages: int = 0,
            dim_mults: list[int] = (1, 2, 4),
            depth: int = 2,
            window_size: int = 8,
    ):
        super().__init__()
        n_stages = len(dim_mults)
        res = img_size // (2 ** (n_stages - 1 + n_conv_stages))
        self.bottleneck = BasicLayer2d(
            dim=dim * dim_mults[-1],
            input_resolution=(res, res),
            depth=depth,
            num_heads=dim_mults[-1],
            window_size=window_size,
        )

    def forward(self, X: Tensor):
        return self.bottleneck(X)


class Decoder(nn.Module):
    def __init__(self,
                 img_size: int = 256,
                 dim: int = 64,
                 n_conv_stages: int = 0,
                 dim_mults: list[int] = [1, 2, 4],
                 depths: list[int] = [6, 4, 2],
                 window_size: int = 8,
                 legacy_v: int = 4
    ):
        super().__init__()

        assert len(dim_mults) == len(depths)

        self.n_stages = len(dim_mults)
        self.dims = [dim * dim_mults[i] for i in range(self.n_stages)]
        self.n_heads = dim_mults
        res = img_size // (2 ** n_conv_stages)

        # transformer stages
        self.up_blocks = nn.ModuleList([])
        for i in range(self.n_stages):
            self.up_blocks.append(nn.ModuleList([
                BasicLayer2d(
                    dim=self.dims[i] * 2,
                    input_resolution=(res, res),
                    depth=depths[i],
                    num_heads=self.n_heads[i],
                    window_size=window_size,
                    partial=True,
                ),
                PatchResizing2d(
                    in_channels=self.dims[i] * 2,
                    out_channels=self.dims[i-1],
                    up=True,
                ) if i > 0 else nn.Identity(),
            ]))
            res = res // 2
        
        # convolution stages
        self.conv_up_blocks = nn.ModuleList([])
        for i in range(n_conv_stages):
            self.conv_up_blocks.append(nn.Sequential(
                Upsample(dim * 2, dim * 2, legacy_v=legacy_v),
                nn.GELU(),
            ))
        
        # last convolution
        self.last_conv = ConvNormAct(dim * 2, 3, kernel_size=1, stride=1, padding=0, activation='tanh')
    
    def forward(self, X: Tensor, skips: list[Tensor], masks: list[Tensor]):
        for (blk, up), skip, mask in zip(reversed(self.up_blocks), reversed(skips), reversed(masks)):
            X, mask = blk(torch.cat((X, skip), dim=1), mask.float())
            X = up(X)
        for blk in self.conv_up_blocks:
            X = blk(X)
        X = self.last_conv(X)
        return X


class Classifier(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.cls = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def forward(self, X: Tensor):
        return self.cls(X)


class RefineNet(nn.Module):
    def __init__(
            self,
            dim: int = 64,
            dim_mults: list[int] = (1, 2, 4, 8, 8),
            legacy_v: int = 4,
    ):
        super().__init__()
        n_stages = len(dim_mults)
        dims = [dim * dim_mults[i] for i in range(n_stages)]

        self.refine_first = GatedConv2d(3 + 1, dims[0], 5, stride=1, padding=2, activation='gelu')
        self.refine_encoder = nn.ModuleList([])
        for i in range(n_stages):
            self.refine_encoder.append(
                nn.ModuleList([
                    ResBlock(dims[i], dims[i], 3, stride=1, padding=1, activation='gelu', gated=True),
                    Downsample(dims[i], dims[i+1]) if i < n_stages - 1 else nn.Identity(),
                ])
            )
        self.refine_bottleneck = ResBlock(dims[-1], dims[-1], 3, stride=1, padding=1, activation='gelu', gated=True)
        self.refine_decoder = nn.ModuleList([])
        for i in range(n_stages-1, -1, -1):
            self.refine_decoder.append(
                nn.ModuleList([
                    ResBlock(dims[i] * 2, dims[i], 3, stride=1, padding=1, activation='gelu', gated=True),
                    Upsample(dims[i], dims[i-1], legacy_v=legacy_v) if i > 0 else nn.Identity(),
                ])
            )
        self.refine_last = ConvNormAct(dims[0], 3, 1, stride=1, padding=0, activation='tanh')

    def forward(self, X: Tensor, mask: Tensor):
        skips = []
        X = self.refine_first(torch.cat((X, mask.float()), dim=1))
        for blk, down in self.refine_encoder:
            X = blk(X)
            skips.append(X)
            X = down(X)
        X = self.refine_bottleneck(X)
        for blk, up in self.refine_decoder:
            X = blk(torch.cat((X, skips.pop()), dim=1))
            X = up(X)
        X = self.refine_last(X)
        return X


class MaskPredictor(nn.Module):
    def __init__(self,
                 img_size: int = 256,
                 dim: int = 64,
                 n_conv_stages: int = 0,
                 dim_mults: list[int] = (1, 2, 4),
                 proj_dim: int = 128,
                 fuse: bool = True,
                 encoder_depths: list[int] = (6, 4, 2),
                 window_size: int = 8,

                 conf_threshs: list[float] = (1.0, 0.95, 0.95),
                 temperature: float = 0.1,
                 kmeans_n_iters: int = 10,
                 kmeans_repeat: int = 3,

                 legacy_v: int = 4
    ):
        super().__init__()

        assert len(dim_mults) == len(encoder_depths)

        # kmeans params
        self.conf_threshs = conf_threshs
        self.temperature = temperature
        self.kmeans_n_iters = kmeans_n_iters
        self.kmeans_repeat = kmeans_repeat

        self.encoder = Encoder(
            img_size=img_size,
            dim=dim,
            n_conv_stages=n_conv_stages,
            dim_mults=dim_mults,
            depths=encoder_depths,
            window_size=window_size,
            legacy_v=legacy_v,
        )
        self.project_heads = ProjectHeads(
            dim=dim,
            dim_mults=dim_mults,
            proj_dim=proj_dim,
            fuse=fuse,
        )

    def forward(self, X: Tensor, gt_mask: Tensor = None, classifier: nn.Module = None):
        # encoder
        _, skips = self.encoder(X)

        # project heads
        projs = self.project_heads(skips)

        # kmeans
        conf_mask_hier = confidence_mask_hierarchy(
            projs=[p.detach() for p in projs],
            gt_mask=gt_mask.float() if gt_mask is not None else None,
            conf_threshs=self.conf_threshs,
            temperature=self.temperature,
            kmeans_n_iters=self.kmeans_n_iters,
            kmeans_repeat=self.kmeans_repeat,
            classifier=classifier,
        )
        return projs, conf_mask_hier
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        self.encoder.load_state_dict(state_dict['encoder'], strict=strict)
        self.project_heads.load_state_dict(state_dict['project_heads'], strict=strict)

    def my_state_dict(self):
        return dict(
            encoder=self.encoder.state_dict(),
            project_heads=self.project_heads.state_dict(),
        )


class HCL(nn.Module):
    def __init__(self,
                 img_size: int = 256,
                 dim: int = 64,
                 n_conv_stages: int = 0,
                 dim_mults: list[float] = [1, 2, 3],
                 proj_dim: int = 128,
                 fuse: bool = True,
                 encoder_depths: list[int] = [6, 4, 2],
                 decoder_depths: list[int] = [2, 2, 2],
                 window_size: int = 8,
                 bottleneck_window_size: int = 16,
                 bottleneck_depth: int = 2,
                 
                 conf_threshs: list[float] = [1.0, 0.95, 0.95],
                 temperature: float = 0.1,
                 kmeans_n_iters: int = 10,
                 kmeans_repeat: int = 3,
                 
                 gt_mask_to_decoder: bool = False,
                 dual_inpainting: bool = False,

                 legacy_v: int = 4
    ):
        super().__init__()

        assert len(dim_mults) == len(encoder_depths) == len(decoder_depths)

        self.img_size = img_size
        self.n_conv_stages = n_conv_stages
        self.n_stages = len(dim_mults)
        # kmeans params
        self.conf_threshs = conf_threshs
        self.temperature = temperature
        self.kmeans_n_iters = kmeans_n_iters
        self.kmeans_repeat = kmeans_repeat
        # control flow
        self.gt_mask_to_decoder = gt_mask_to_decoder
        self.dual_inpainting = dual_inpainting

        self.encoder = Encoder(
            img_size=img_size,
            dim=dim,
            n_conv_stages=n_conv_stages,
            dim_mults=dim_mults,
            depths=encoder_depths,
            window_size=window_size,
            legacy_v=legacy_v,
        )
        self.project_heads = ProjectHeads(
            dim=dim,
            dim_mults=dim_mults,
            proj_dim=proj_dim,
            fuse=fuse,
        )
        self.bottleneck = Bottleneck(
            img_size=img_size,
            dim=dim,
            n_conv_stages=n_conv_stages,
            dim_mults=dim_mults,
            depth=bottleneck_depth,
            window_size=bottleneck_window_size,
        )
        self.decoder = Decoder(
            img_size=img_size,
            dim=dim,
            n_conv_stages=n_conv_stages,
            dim_mults=dim_mults,
            depths=decoder_depths,
            window_size=window_size,
            legacy_v=legacy_v,
        )

    def forward(self, X: Tensor, gt_mask: Tensor = None, classifier: nn.Module = None):
        # encoder
        X, skips = self.encoder(X)

        # project heads
        projs = self.project_heads(skips)

        # kmeans
        conf_mask_hier = confidence_mask_hierarchy(
            projs=[p.detach() for p in projs],
            gt_mask=gt_mask.float() if gt_mask is not None else None,
            conf_threshs=self.conf_threshs,
            temperature=self.temperature,
            kmeans_n_iters=self.kmeans_n_iters,
            kmeans_repeat=self.kmeans_repeat,
            classifier=classifier,
        )
        pred_masks = conf_mask_hier['pred_masks']

        # bottleneck
        X = self.bottleneck(X)

        # decoder
        if self.gt_mask_to_decoder:
            assert gt_mask is not None
            masks_to_decoder = [
                F.interpolate(gt_mask.float(), size=self.img_size // (2 ** i))
                for i in range(self.n_conv_stages, self.n_conv_stages + self.n_stages)
            ]
        else:
            masks_to_decoder = pred_masks
        out = self.decoder(X, skips, masks_to_decoder)

        if not self.dual_inpainting:
            return out, projs, conf_mask_hier

        # dual inpainting (bidirectional inpainting)
        if self.gt_mask_to_decoder:
            assert gt_mask is not None
            masks_to_decoder = [
                (1. - F.interpolate(gt_mask.float(), size=self.img_size // (2 ** i)))  # reverse
                for i in range(self.n_conv_stages, self.n_conv_stages + self.n_stages)
            ]
        else:
            masks_to_decoder = [~m for m in pred_masks]  # reverse
        out2 = self.decoder(X, skips, masks_to_decoder)
        return (out, out2), projs, conf_mask_hier

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        self.encoder.load_state_dict(state_dict['encoder'], strict=strict)
        self.project_heads.load_state_dict(state_dict['project_heads'], strict=strict)
        self.bottleneck.load_state_dict(state_dict['bottleneck'], strict=strict)
        self.decoder.load_state_dict(state_dict['decoder'], strict=strict)
    
    def my_state_dict(self):
        return dict(
            encoder=self.encoder.state_dict(),
            project_heads=self.project_heads.state_dict(),
            bottleneck=self.bottleneck.state_dict(),
            decoder=self.decoder.state_dict(),
        )


if __name__ == '__main__':
    inpaintnet = HCL(
        dim=64,
        proj_dim=64,
        encoder_depths=[6, 4, 2],
        decoder_depths=[2, 2, 2],
        bottleneck_window_size=8
    )

    dummy_input = torch.randn(1, 3, 256, 256)
    inpaintnet(dummy_input)
    
    print(sum(p.numel() for p in inpaintnet.parameters()))
    print(sum(p.numel() for p in inpaintnet.encoder.parameters()))
    print(sum(p.numel() for p in inpaintnet.project_heads.parameters()))
    print(sum(p.numel() for p in inpaintnet.bottleneck.parameters()))
    print(sum(p.numel() for p in inpaintnet.decoder.parameters()))
