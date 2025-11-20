import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, inverse_tanh, inverse_translated_sigmoid, get_step_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, strip_2D
from utils.reloc_utils import compute_relocation_cuda
from utils.sghmc import AdamSGHMC
from utils.system_utils import mkdir_p
from simple_knn._C import distCUDA2

class Model:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, rotation, scaling_modifier=1.0):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_2D(actual_covariance)
            return symm

        def build_covariance_inv_from_scaling_rotation(scaling, rotation, scaling_modifier=1.0):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            actual_covariance_inv = torch.inverse(actual_covariance)
            symm = strip_2D(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation # translate
        self.covariance_inv_activation = build_covariance_inv_from_scaling_rotation

        self.opacity_activation = torch.tanh
        # self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_tanh

        self.rotation_activation = torch.nn.functional.normalize

        # NOTE: make sure freedom of degree is always >= 1.0f
        self.nu_degree_activation = nn.Hardtanh(1, 10000)

        self.sigmoid_function = torch.sigmoid
        self.inverse_translated_sigmoid = inverse_translated_sigmoid

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


        l.append('negative')

        return l

    def save(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        # xyz = torch.cat([xyz, torch.zeros(xyz.shape[0], 1, device="cuda")], dim=1).cpu().numpy()
        color = self._rgb.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        negative = self._negative.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, color, opacities, scale, rotation, negative), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"])),  axis=1)
        rgb = np.stack((np.asarray(plydata.elements[0]["r"]),
                        np.asarray(plydata.elements[0]["g"]),
                        np.asarray(plydata.elements[0]["b"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

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

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rgb = nn.Parameter(torch.tensor(rgb, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        negatives = np.asarray(plydata.elements[0]["negative"])[..., np.newaxis]
        self._negative = nn.Parameter(torch.tensor(negatives, dtype=torch.float, device="cuda").requires_grad_(False))


    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_negative(self):
        return self._negative

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    def get_cov(self):
        return self.covariance_activation(self.get_scaling, self.get_rotation)

    def get_cov_inv(self):
        return self.covariance_inv_activation(self.get_scaling, self.get_rotation)

    @property
    def get_rgb(self):
        return self._rgb

    def create_from_samlpe_points(self, sample_xy : torch.Tensor, rgb : torch.Tensor, spatial_lr_scale):
        # 过滤掉rgb全为0的点
        mask = rgb.sum(dim=1) > 0.00
        rgb = rgb[mask]

        sample_xy = sample_xy[mask]

        self.spatial_lr_scale = spatial_lr_scale

        points = torch.cat([sample_xy, torch.zeros(sample_xy.shape[0], 1, device=sample_xy.device)], dim=1)
        dist2 = torch.clamp_min(distCUDA2(points.float().cuda()), 0.000001)
        if torch.any(dist2.isinf()):
            dist2 = torch.where(dist2.isinf(), 10., dist2)

        scales = torch.log(torch.sqrt(dist2)*0.1)[...,None].repeat(1, 3)[:, :2]
        rots = torch.zeros((sample_xy.shape[0], 1), device="cuda").float().cuda()

        opacities = self.inverse_opacity_activation(0.5 * torch.ones((sample_xy.shape[0], 1), dtype=torch.float, device="cuda"))
        negatives = torch.full_like(opacities, 1.0).float().cuda()

        self._xyz = nn.Parameter(sample_xy.float().cuda().requires_grad_(True))
        self._rgb = nn.Parameter(rgb.float().cuda().requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._negative = nn.Parameter(negatives.requires_grad_(True))


    def training_setup(self, training_args, C_burnin=5e3, C=1.3e2, burnin_iterations=7000):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._rgb], 'lr': training_args.color_lr, "name": "rgb"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._negative], 'lr': training_args.negetive_lr, "name": "negative"},
        ]

        self.optimizer = AdamSGHMC(params=l, eps=1e-15, mdecay=C, scale_grad=1.0, mdecay_burnin=C_burnin, burnin_iterations=burnin_iterations)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.degree_scheduler_args = get_step_lr_func(lr_init=training_args.degree_lr, step_size=5000, gamma=0.5)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "degree":
                lr = self.degree_scheduler_args(iteration)
                param_group['lr'] = lr
                break

        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def cat_tensors_to_optimizer(self, tensors_dict, inds):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["momentum"] = torch.cat((stored_state["momentum"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                # print()
                # print(group["name"])
                # print("Existing tensor shape:", group["params"][0].shape)
                # print("Extension tensor shape:", extension_tensor.shape)
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_opacities, new_scaling, new_rotation, new_negative, new_color ,indices=None, reset_params=True):
        d = {"xyz": new_xyz,
        "rgb": new_color,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "negative": new_negative}

        # print("xyz", new_xyz.shape)
        # print("rgb", new_color.shape)
        # print("opa:", new_opacities.shape)
        # print("sca", new_scaling.shape)
        # print("rot", new_rotation.shape)
        # print("nega", new_negative.shape)

        optimizable_tensors = self.cat_tensors_to_optimizer(d, indices)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._rgb = optimizable_tensors["rgb"]
        self._negative = optimizable_tensors["negative"]
        self._negative.requires_grad_(False)

        if reset_params:
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def replace_tensors_to_optimizer(self, inds=None):
        tensors_dict = {
            "xyz": self._xyz,
            "rgb": self.get_rgb,
            "opacity": self._opacity,
            "scaling" : self._scaling,
            "rotation" : self._rotation,
            "negative": self._negative
            }

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)

            if stored_state is not None:
                if inds is not None:
                    stored_state["exp_avg"][inds] = 0
                    stored_state["exp_avg_sq"][inds] = 0
                else:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._rgb = optimizable_tensors["rgb"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._negative = optimizable_tensors["negative"]
        self._negative.requires_grad_(False)

        return optimizable_tensors

    def replace_tensors_to_optimizer_momentum(self, inds=None):
        tensors_dict = {
            "xyz": self._xyz,
            "rgb": self._rgb,
            "opacity": self._opacity,
            "scaling" : self._scaling,
            "rotation" : self._rotation,
            "negative": self._negative
            }

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)

            if stored_state is not None:
                if inds is not None:
                    # NOTE: do not reset Adam momentum to avoid overlap (like adding noise)
                    # stored_state["exp_avg"][inds] = 0
                    # stored_state["exp_avg_sq"][inds] = 0
                    stored_state["momentum"][inds] = 0

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._rgb = optimizable_tensors["rgb"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._negative = optimizable_tensors["negative"]
        self._negative.requires_grad_(False)

        return optimizable_tensors

    def _update_params(self, idxs, ratio):
        new_opacity, new_scaling = compute_relocation_cuda(
            opacity_old = self.get_opacity[idxs, 0] * self.get_negative[idxs, 0],
            scale_old=self.get_scaling[idxs],
            N=ratio[idxs, 0] + 1
        )

        # new_opacity = self.get_opacity[idxs] * self.get_negative[idxs]

        # new_scaling = self.get_scaling[idxs]

        new_opacity = torch.clamp(new_opacity.unsqueeze(-1), max = 1.0 - torch.finfo(torch.float32).eps, min = -1.0 + torch.finfo(torch.float32).eps)

        new_opacity = torch.where((new_opacity >= 0) & (new_opacity < 0.005), 0.005, new_opacity)

        new_opacity = torch.where((new_opacity < 0) & (new_opacity > -0.005), -0.005, new_opacity)

        new_opacity = new_opacity / self.get_negative[idxs]
        new_opacity = self.inverse_opacity_activation(new_opacity)
        new_scaling = self.scaling_inverse_activation(new_scaling.reshape(-1, 2))

        # print("xyz device", self._xyz.device)
        # print("rotation device", self._rotation.device)
        # print("negative device", self._negative.device)
        # print("idxs device", idxs.device)

        return self._xyz[idxs], self._rgb[idxs], new_opacity, new_scaling, self._rotation[idxs], self._negative[idxs]

    def _sample_alives(self, probs, num, alive_indices=None):
            # 添加调试信息
        print(f"probs shape: {probs.shape}")
        print(f"probs min: {probs.min()}, max: {probs.max()}")
        print(f"probs sum: {probs.sum()}")

    # # 检查 NaN 和 Inf
    #     if torch.isnan(probs).any():
    #         print("❌ NaN values found in probs!")
    #         probs = torch.nan_to_num(probs, nan=0.0)

    #     if torch.isinf(probs).any():
    #         print("❌ Inf values found in probs!")
    #         probs = torch.nan_to_num(probs, posinf=1.0, neginf=0.0)

    # # 确保所有值非负
    #     if (probs < 0).any():
    #         print("❌ Negative values found in probs!")
    #         probs = torch.clamp(probs, min=0.0)

    # # 检查是否所有概率都为0
    #     if (probs.sum() == 0).all():
    #         print("⚠️ All probabilities are zero, using uniform distribution")
    #         probs = torch.ones_like(probs) / probs.size(0)


        probs = probs.abs() / (probs.abs().sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs).unsqueeze(-1)
        return sampled_idxs, ratio

    def recycle_components(self, dead_mask=None):
        if dead_mask.sum() == 0:
            return

        alive_mask = ~dead_mask
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        # only recycle 5% of all components at max
        if dead_mask.sum() > int(0.05 * self.get_opacity.shape[0]):
            sorted, indices = torch.sort(torch.abs(self.get_opacity.squeeze(-1)))
            dead_indices = indices[0: int(0.05 * self.get_opacity.shape[0])]

        if alive_indices.shape[0] <= 0:
            return

        # sample from alive ones based on opacity
        probs = (self.get_opacity[alive_indices, 0])
        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])

        (
            self._xyz[dead_indices],
            self._rgb[dead_indices],
            self._opacity[dead_indices],
            self._scaling[dead_indices],
            self._rotation[dead_indices],
            _
        ) = self._update_params(reinit_idx, ratio=ratio)

        self._opacity[reinit_idx] = self._opacity[dead_indices]
        self._scaling[reinit_idx] = self._scaling[dead_indices]

        self.replace_tensors_to_optimizer(inds=reinit_idx)
        self.replace_tensors_to_optimizer_momentum(inds=dead_indices)

    def add_components(self, cap_max):
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(1.05 * current_num_points))
        min_gs = 10 if current_num_points < 100 else 0
        num_gs = max(min_gs, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        print("opa", self.get_opacity)
        probs = self.get_opacity.squeeze(-1)
        print("probs", probs)
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)

        (
            new_xyz,
            new_rgb,
            new_opacity,
            new_scaling,
            new_rotation,
            new_negative,
        ) = self._update_params(add_idx, ratio=ratio)

        # print(self._opacity.shape)
        # print(add_idx.shape)
        # print(new_opacity.shape)
        self._opacity[add_idx] = new_opacity
        self._scaling[add_idx] = new_scaling

        # print("new_xyz shape:", new_xyz.shape)
        # print("new_opacity shape:", new_opacity.shape)
        # print("new_scaling shape:", new_scaling.shape)
        # print("new_rotation shape:", new_rotation.shape)
        # print("new_negative shape:", new_negative.shape)

        self.densification_postfix(new_xyz, new_opacity, new_scaling, new_rotation, new_negative, new_rgb, add_idx, reset_params=False)
        self.replace_tensors_to_optimizer(inds=add_idx)

        return num_gs

    def update_opacity(self, op):
        self._opacity = op
