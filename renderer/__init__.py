from scene.gaussian_models import Model
from arguments import ModelParams, ArgumentParser, OptimizationParams, PipelineParams
import torch
from picturegs import GaussianRasterizationSettings, GaussianRasterizer
from utils.general_utils import build_scaling_rotation


def render(pc : Model, opt : OptimizationParams, background, resoulation):
    H, W = resoulation

    raster_settings = GaussianRasterizationSettings(
        H, W,
        torch.zeros(3),
        False,
        False,
        False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = pc.get_xyz
    opacity = pc.get_opacity
    rgb = pc.get_rgb

    scale = pc.get_scaling
    rot = pc.get_rotation
    nega = pc.get_negative

    # cov_inv = pc.get_cov_inv()

    # print("xy shape:", means2D.shape)
    # print("color shape::", rgb.shape)
    # print("opa shape:", opacity.shape)
    # print("cov shape:", cov_inv.shape)

    render_img, radii = rasterizer(
        means2D=means2D,
        colors=rgb,
        opacities=opacity,
        scale=scale,
        rots=rot,
        negative=nega
    )

    # print(render_img.device)
    # print(render_img.shape)
    render_img = render_img.clamp(0., 1.)

    return {"render" : render_img}
