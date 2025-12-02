import torch
from scene import Scene
from arguments import ModelParams, ArgumentParser, OptimizationParams, PipelineParams
from scene.gaussian_models import Model
from utils.image_utils import read_image_from_data, psnr, write_exr_image
import sys
from utils.general_utils import safe_state, get_expon_lr_func
from tqdm import tqdm
from renderer import render
from utils.loss_utils import l1_loss, ssim, DWT_loss, l1_loss_part
from scene.gaussian_models import build_scaling_rotation
from utils.general_utils import safe_state, op_sigmoid
import json
import os
import uuid
import torchvision
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from PIL import Image
from os import makedirs
import torchvision.transforms.functional as tf
from argparse import ArgumentParser, Namespace
from lpipsPyTorch import lpips
from utils.test_utils import test

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    primitives = Model()
    scene = Scene(dataset, primitives)
    primitives.training_setup(opt)
    tb_writer = prepare_output_and_logger(dataset)
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    pixels = scene.getImages().shape[1] * scene.getImages().shape[2]
    args.cap_max = int(30_000 * (pixels / (2048 * 1024)))

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    gt_image = scene.getImages().cuda()

    offset = 1.
    gt_image = offset * gt_image

    resoulation = (gt_image.shape[1], gt_image.shape[2])
    image = torch.ones_like(gt_image.max())
    print("xyz: ", primitives.get_xyz.max())
    print("rgb:", primitives.get_rgb.max())
    print("opa:", primitives.get_opacity.max())
    print("scale:", primitives.get_scaling.max())
    print("rot:", primitives.get_rotation.max())
    print("gt max:", gt_image.max())

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        primitives.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            torch.cuda.empty_cache()

        render_pkg = render(primitives, opt, background, resoulation)

        image = render_pkg["render"]

        if iteration == 1:
            with torch.no_grad():
                write_exr_image("output", image, "1.exr")

        # print("\n point color max:", primitives.get_rgb.max())
        # print(" pixel color max:", image.max())
        # if iteration % 10 == 0:
        #     # print(" cur opa max:", primitives.get_opacity.max())
        #     # print(" point color max:", primitives.get_rgb.max())
        #     # print(" color max:", image.max())
        #     # print("scale:", primitives.get_scaling.max())
        #     print("xyz: ", primitives.get_xyz.max())
        #     print("rgb:", primitives.get_rgb.max())
            # print("opa min:", primitives.get_opacity.min())
            # print("opa max:", primitives.get_opacity.max())
        #     print("scale:", primitives.get_scaling.max())
        #     print("rot:", primitives._rotation.max())


        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim - opt.lambda_fre) * Ll1 + (opt.lambda_dssim - opt.lambda_fre) * (1.0 - ssim(image, gt_image)) + opt.lambda_fre * DWT_loss(image, gt_image)

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "points:": '{}'.format(scene.get_points_size())})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Optimizer step
            if iteration < opt.iterations:
                # NOTE: SGHMC optimization
                sig = (op_sigmoid(1 - torch.abs(primitives.get_opacity)))

                L = build_scaling_rotation(primitives.get_scaling, primitives.get_rotation)
                actual_covariance = L @ L.transpose(1, 2)

                if dataset.optimizer_type == "sghmc":
                    _, total_max, total_mean, total_min = primitives.optimizer.step(sig=sig.detach(), cov=actual_covariance.detach())
                    primitives.optimizer.zero_grad(set_to_none = True)
                    if tb_writer:
                        tb_writer.add_scalar('sghmc_total_max', total_max.item(), iteration)
                        tb_writer.add_scalar('sghmc_total_mean', total_mean.item(), iteration)
                        tb_writer.add_scalar('sghmc_total_min', total_min.item(), iteration)

            if iteration > 500 and iteration < 5000 and iteration % 500 == 0 and False:
                primitives.scale_control(0.005, scene.cameras_extent)

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), dataset)
            if (iteration in saving_iterations or iteration == opt.iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                # scene.update_opacity(opt.dropout)
                scene.save_for_rec()
                test(gt_image.unsqueeze(0), image.unsqueeze(0))
                write_exr_image("output", image)


            if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                dead_mask = (primitives.get_opacity < dataset.opacity_threshold).squeeze(-1)
                dead_mask2 = (primitives.get_opacity > -dataset.opacity_threshold).squeeze(-1)
                dead_mask =  torch.logical_and(dead_mask, dead_mask2)

                primitives.recycle_components(dead_mask=dead_mask)
                add_num = primitives.add_components(cap_max=args.cap_max)

                torch.cuda.empty_cache()

            # if (iteration in checkpoint_iterations):
            #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
            #     torch.save((primitives.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, dataset):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

        positive_pts_mask = torch.where(scene.primitives.get_opacity >= 0, True, False).squeeze(-1)
        print("positive componets number: ", (positive_pts_mask==True).sum().item())
        print("negative componets number: ", (positive_pts_mask==False).sum().item())

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.primitives.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.primitives.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config


if __name__ == "__main__":

    # read_image_from_data()
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--config', type=str, default="setting.json")
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])

    print("Optimizing " + args.model_path)

    if args.config is not None:
        # Load the configuration file
        config = load_config(args.config)
        # Set the configuration parameters on args, if they are not already set by command line arguments
        for key, value in config.items():
            setattr(args, key, value)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)
    # primitives = Model()
    # scene = Scene(lp.extract(args), primitives, render=True)

