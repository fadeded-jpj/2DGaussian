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

def render_set(model_path, iteration, primitives, pipeline, gt_image):
    render_path = os.path.join(model_path, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    resoulation = (gt_image.shape[1], gt_image.shape[2])
    bg = torch.zeros(3)
    render_pkg = render(primitives, pipeline, bg, resoulation)
    rendering = render_pkg["render"]

    write_exr_image(render_path, rendering)
    write_exr_image(gts_path, gt_image)
    exr_to_png_opencv(rendering, render_path)
    exr_to_png_opencv(gt_image, gts_path)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        primitives = Model()
        scene = Scene(dataset, primitives, load_iteration=iteration, shuffle=False, render=True)
        gt = scene.getImages().cuda()
        render_set(dataset.model_path, scene.loaded_iter, primitives, pipeline, gt)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if args.iteration != -1:
        render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
    else:
        model_path = os.path.join(args.model_path, "point_cloud")
        save_iterations = searchForAllIteration(model_path)
        for iter in save_iterations:
            print("render ours_{}".format(iter))
            render_sets(model.extract(args), iter, pipeline.extract(args), args.skip_train, args.skip_test)
