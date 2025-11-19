#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means2D,
    colors,
    opacities,
    scales,
    rots,
    negative,
    raster_settings
):
    return _RasterizeGaussians.apply(
        means2D,
        colors,
        opacities,
        scales,
        rots,
        negative,
        raster_settings
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means2D,
        colors,
        opacities,
        scales,
        rots,
        negative,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means2D,
            colors,
            opacities,
            scales,
            rots,
            negative,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors, means2D, scales, rots, radii, opacities, geomBuffer, binningBuffer, imgBuffer,negative)
        return color, radii

    @staticmethod
    def backward(ctx, grad_out_color, _):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors, means2D, scales, rots, radii, opacities, geomBuffer, binningBuffer, imgBuffer ,negative= ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means2D,
                radii,
                colors,
                opacities,
                scales,
                rots,
                negative,
                grad_out_color,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.antialiasing,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        grad_means2D, grad_colors, grad_opacities,  grad_scales, grad_rots, grad_negative = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means2D,
            grad_colors,
            grad_opacities,
            grad_scales,
            grad_rots,
            grad_negative,
            None
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    bg : torch.Tensor
    prefiltered : bool
    debug : bool
    antialiasing : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def forward(self,  means2D, opacities,  colors = None, scale=None, rots=None,negative=None):
        raster_settings = self.raster_settings
        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means2D,
            colors,
            opacities,
            scale,
            rots,
            negative,
            raster_settings,
        )

def compute_relocation(opacity_old, scale_old, N, binoms, n_max):
    new_opacity, new_scale = _C.compute_relocation(opacity_old, scale_old, N.int(), binoms, n_max)
    return new_opacity, new_scale 