import torch
from scene import Scene_for_Render
import os
from tqdm import tqdm
from os import makedirs
from renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.gaussian_models import Model
from utils.loss_utils import normalize
from utils.image_utils import write_exr_image, exr_to_png_opencv
from utils.system_utils import searchForAllIteration

def render_set(dataset : ModelParams, pipeline : PipelineParams, W, H, id, time):
    with torch.no_grad():
        primitives = Model()
        ply_path = os.path.join(dataset.model_path)
        scene = Scene_for_Render(ply_path, primitives, time=time, id=id)
        print("opa max:", primitives.get_opacity.max())
        print("opa min:", primitives.get_opacity.min())
        print("scale max:", primitives.get_scaling.max())
        print("scale min:", primitives.get_scaling.min())
        print("rot min:", primitives.get_rotation.min())
        print("rot max:", primitives.get_rotation.max())
        render_path = os.path.join(dataset.model_path)
        makedirs(render_path, exist_ok=True)
        bg = torch.zeros(3)

        render_pkg = render(primitives, pipeline, bg, [H, W])
        rendering = render_pkg["render"]

        write_exr_image(render_path, rendering, f"lightmap_{id}_{time}.exr")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--W", type=int, default=2048)
    parser.add_argument("--H", type=int, default=1024)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_set(model.extract(args), pipeline.extract(args), args.W, args.H, args.id, args.time)
