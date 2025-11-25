import torch
from scene import Scene
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
import time as Time
import re

total_time = 0.0

def render_set(dataset : ModelParams, pipeline : PipelineParams, W, H, time):
    global total_time
    with torch.no_grad():
        for fname in os.listdir(dataset.model_path):
            pattern = r'(\d+)_(\d+)\.ply'
            matches = re.findall(pattern, fname)

            try:
                _, id = matches[0]
            except IndexError:
                return

            primitives = Model()
            scene = Scene(dataset, primitives, load_iteration=-1, shuffle=False, render=True, time=time, id=id)
            render_path = os.path.join(dataset.model_path)
            makedirs(render_path, exist_ok=True)
            bg = torch.zeros(3)

            time_begin = Time.time()
            render_pkg = render(primitives, pipeline, bg, [H, W])
            rendering = render_pkg["render"]
            time_end = Time.time()
            write_exr_image(render_path, rendering, f"lightmap_{id}_{time}.exr")

            total_time += time_end - time_begin


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

    render_set(model.extract(args), pipeline.extract(args), args.W, args.H, args.time)

    print("total time cost: ", total_time)
