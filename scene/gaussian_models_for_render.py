import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, inverse_translated_sigmoid
from torch import nn
from plyfile import PlyData


class Model:

    def setup_functions(self):
        pass
        # self.scaling_activation = torch.exp
        # self.scaling_inverse_activation = torch.log

        # self.rotation_activation = torch.sigmoid
        # self.rotation_inverse_activation = inverse_sigmoid

        # self.opacity_activation = torch.sigmoid
        # self.inverse_opacity_activation = inverse_sigmoid
        # # self.opacity_activation = torch.tanh
        # # self.inverse_opacity_activation = inverse_tanh

        # # self.rotation_activation = torch.nn.functional.normalize

        # # NOTE: make sure freedom of degree is always >= 1.0f
        # self.nu_degree_activation = nn.Hardtanh(1, 10000)

        # self.sigmoid_function = torch.sigmoid
        # self.inverse_translated_sigmoid = inverse_translated_sigmoid

    def __init__(self):
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)      # scale_x, scale_y
        self._rotation = torch.empty(0)     # theta
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        # self._rgb = torch.empty(0)
        self.setup_functions()

        # NOTE: This is always 1
        self._negative = torch.empty(0)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'r', 'g', 'b']
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        # l.append('negative')
        return l


    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"])),  axis=1)
        rgb = np.stack((np.asarray(plydata.elements[0]["r"]),
                        np.asarray(plydata.elements[0]["g"]),
                        np.asarray(plydata.elements[0]["b"])), axis=1)
        # opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        opacities = torch.ones(len(xyz), 1).float().cuda()

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._rgb = nn.Parameter(torch.tensor(rgb, dtype=torch.float, device="cuda").requires_grad_(False))
        # self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))

        # negative = np.asarray(plydata.elements[0]["negative"])[..., np.newaxis]
        # self._negative = nn.Parameter(torch.tensor(negative, dtype=torch.float, device="cuda")).requires_grad_(True)
        negatives = torch.ones_like(self._opacity).float().cuda()
        self._negative = nn.Parameter(negatives).requires_grad_(False)


    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return self._opacity

    @property
    def get_negative(self):
        return self._negative

    @property
    def get_scaling(self):
        return self._scaling

    @property
    def get_rotation(self):
        return self._rotation

    @property
    def get_rgb(self):
        return self._rgb
