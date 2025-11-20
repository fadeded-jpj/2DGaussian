from utils.sampling_util import get_sample_init_xy
from scene.gaussian_models import Model

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from arguments import ModelParams
from utils.image_utils import read_images, get_image_color, create_image_tensor, plot_tensor_image
import torch

from scene.gaussian_models import Model
class Scene:

    primitives : Model

    def __init__(self, args : ModelParams, primitives : Model, load_iteration=None, shuffle=True, resolution_scales=[1.0], render=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.primitives = primitives
        self.images = read_images(os.path.join(args.source_path))
        _, H, W = self.images.images.shape

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))



        if render is False:
            sample_xy = get_sample_init_xy(n_points=2048).cuda() # N, 2  [0, 1]
            rgb = get_image_color(self.images, sample_xy).cuda() # N, 3
            sample_xy[:, 0] *= W
            sample_xy[:, 1] *= H
            self.primitives.create_from_samlpe_points(sample_xy, rgb, spatial_lr_scale=1.0)
        else:
            self.primitives.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))



    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.primitives.save(os.path.join(point_cloud_path, "point_cloud.ply"))

    def update_opacity(self, dropout):
        op = self.primitives.get_opacity
        op = (1. - dropout) * op
        self.primitives.update_opacity(op)

    def get_points_size(self):
        return self.primitives.get_xyz.shape[0]

    def getImages(self):
        return self.images.images
