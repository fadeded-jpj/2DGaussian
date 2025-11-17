from scene.gaussian_models import Model
from arguments import ModelParams, ArgumentParser, OptimizationParams, PipelineParams
import torch
from picturegs import GaussianRasterizationSettings, GaussianRasterizer
from utils.general_utils import build_scaling_rotation


def render(pc : Model, opt : OptimizationParams, background, resoulation):
    H, W = resoulation
    render_img = torch.zeros((resoulation[1], resoulation[0], 3))  # W, H, 3

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

    cov_inv = pc.get_cov_inv()

    render_img, radii = rasterizer(
        means2D=means2D,
        colors=rgb,
        opacities=opacity,
        conic=cov_inv
    )

    # render_img = render_img.clamp(0, 1)
    print(render_img.device)

    return {"render" : render_img}
