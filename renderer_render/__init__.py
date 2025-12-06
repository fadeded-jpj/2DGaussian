from scene.gaussian_models import Model
from arguments import ModelParams, ArgumentParser, OptimizationParams, PipelineParams
import torch
from picturegs import GaussianRasterizationSettings, GaussianRasterizer



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

    means2D = pc._xyz
    opacity = pc._opacity
    rgb = pc._rgb

    scale = pc._scaling
    rot = pc._rotation
    nega = pc._negative

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
    render_img = render_img.clamp(0., 114514.)

    return {"render" : render_img}
