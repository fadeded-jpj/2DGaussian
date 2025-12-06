from utils.sampling_util import get_sample_init_xy
from scene.gaussian_models import Model

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from arguments import ModelParams
from utils.image_utils import read_images, get_image_color, create_image_tensor, plot_tensor_image, Image_source
import torch

from scene.gaussian_models import Model


class Scene:

    primitives : Model

    def __init__(self, args : ModelParams, primitives : Model, load_iteration=None, shuffle=True, resolution_scales=[1.0], render=False, time=None, id=None,level =None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.primitives = primitives
        self.images = read_images(os.path.join(args.source_path), args.id, args.time,args.level)
        _, H, W = self.images.images.shape



        if load_iteration and id is None:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))


        if render is False:
            # sample_xy = get_sample_init_xy(n_points=int(4096 * 6 *(4*(1/ init_ratio)/9+5/9))).cuda() # N, 2  [0, 1]
            args.cap_max = int(25_000)
            if(H * W==512*1024):
                args.cap_max = int(18_000)
            if(H * W==256*512):
                args.cap_max = int(10000)
            if(H * W==128*256):
                args.cap_max = int(5000)
            if(H * W==64*128):
                args.cap_max = int(3000)
            sample_xy = get_sample_init_xy(n_points=int(  args.cap_max )).cuda() # N, 2  [0, 1]
            rgb = get_image_color(self.images, sample_xy).cuda() # N, 3
            sample_xy[:, 0] *= W
            sample_xy[:, 1] *= H
            self.primitives.create_from_samlpe_points(sample_xy, rgb, spatial_lr_scale=1.0, optimizer_type=args.optimizer_type)
        elif id is None:
            self.primitives.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.primitives.load_ply(os.path.join(self.model_path,
                                                           f"{time}_{id}__{level}.ply"))



    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.primitives.save(os.path.join(point_cloud_path, "point_cloud.ply"))


    def save_for_rec(self):
        point_cloud_path = os.path.join(self.model_path)
        self.primitives.save(os.path.join(point_cloud_path, f"{self.images.time}_{self.images.id}__{self.images.level}.ply"))


    def update_opacity(self, dropout):
        op = self.primitives.get_opacity
        op = (1. - dropout) * op
        self.primitives.update_opacity(op)

    def get_points_size(self):
        return self.primitives.get_xyz.shape[0]

    def getImages(self):
        return self.images.images


class Scene_for_Render:
    primitives : Model
    def __init__(self, model_path : str, primitives : Model, time=None, id=None, level=None, W=1024, H=512):
        self.model_path = model_path
        self.primitives = primitives
        self.primitives.load_ply(os.path.join(self.model_path, f"{time}_{id}__{level}.ply"))
