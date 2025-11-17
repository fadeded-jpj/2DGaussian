#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random
import re
import os

def getDepthPath(image_name, img_folder):
    depth_path = os.path.join(os.path.dirname(img_folder), "depth", image_name + '.png')
    print("######## depth path: {}".format(depth_path))
    return depth_path

def getMaskPath(image_name, scene_name):
    match = re.search(r'.*rect_(\d+)_', image_name)
    # print("image_name:{}".format(image_name))
    try:
        mask_id = int(match.group(1))
    except AttributeError:
        print(match)
        mask_id = 0
    mask_id = mask_id - 1
    mask_name = f"{mask_id:03d}" + ".png"

    mask_path = os.path.join("data/idrmasks", scene_name, mask_name)

    return mask_path

def op_sigmoid(x, k=100, x0=0.995):
    return 1 / (1 + torch.exp(-k * (x - x0)))

def inverse_sigmoid(x):
    return torch.log(x/((1-x)))

def inverse_translated_sigmoid(x, scale, bias, scale_x):
    x = (x-bias)/scale
    return inverse_sigmoid(x) * scale_x

def inverse_tanh(x):
    return 0.5 * torch.log((1+x)/((1-x)))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def get_step_lr_func(lr_init, step_size=5000, gamma=0.1, min_lr=0.01):
    def helper(step):
        times = int(step / step_size)
        lr = lr_init * pow(gamma, times)
        lr = max(min_lr, lr)
        return lr

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def strip_2D(L):
    uncertainty = torch.zeros((L.shape[0], 3), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 1, 1]
    return uncertainty

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_rotation_2D(r):
    R = torch.zeros((r.size(0), 2, 2), device='cuda')

    R[:, 0, 0] = torch.cos(r[:, 0])
    R[:, 0, 1] = -torch.sin(r[:, 0])
    R[:, 1, 0] = torch.sin(r[:, 0])
    R[:, 1, 1] = torch.cos(r[:, 0])

    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 2, 2), dtype=torch.float, device="cuda")
    R = build_rotation_2D(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

def normalize(map):
    return (map - map.min()) / (map.max() - map.min())


def transform_point4x4_batch_torch(points, matrix):
    """
    使用4x4矩阵批量变换3D点（PyTorch版本）
    """
    n = points.shape[0]

    # 将点转换为齐次坐标 (n, 4)
    points_homogeneous = torch.cat([points, torch.ones(n, 1, device=points.device)], dim=1)

    # 构建变换矩阵 (4, 4)
    transform_matrix = matrix
    # 批量变换: (n, 4) @ (4, 4)^T = (n, 4)
    transformed = points_homogeneous @ transform_matrix.T

    return transformed

def transform_point4x3_batch_torch(points, matrix):
    """
    使用4x3矩阵批量变换3D点（PyTorch版本）
    """
    n = points.shape[0]

    # 将点转换为齐次坐标 (n, 4)
    points_homogeneous = torch.cat([points, torch.ones(n, 1, device=points.device)], dim=1)

    # 构建变换矩阵 (3, 4)
    transform_matrix = matrix

    # 批量变换: (n, 4) @ (3, 4)^T = (n, 3)
    transformed = points_homogeneous @ transform_matrix.T

    return transformed

def world_to_screen_space_batch_torch(world_points, viewmatrix, projmatrix, screen_width, screen_height):
    """
    将批量世界空间坐标转换到屏幕空间坐标（PyTorch版本）
    """
    # 转换到齐次裁剪空间
    p_hom = transform_point4x4_batch_torch(world_points, projmatrix)  # (n, 4)

    # 透视除法
    p_w = 1.0 / (p_hom[:, 3] + 1e-7)  # (n,)
    p_proj = p_hom[:, :3] * p_w.unsqueeze(1)  # (n, 3)

    # 转换到视图空间
    p_view = transform_point4x3_batch_torch(world_points, viewmatrix)  # (n, 3)

    # 根据p_view.z生成缩放系数


    # 将NDC坐标[-1, 1]转换到屏幕坐标[0, screen_width/screen_height]
    screen_x = (p_proj[:, 0] + 1.0) * 0.5 * screen_width
    screen_y = (1.0 - p_proj[:, 1]) * 0.5 * screen_height  # Y轴翻转
    screen_coords = torch.stack([screen_x, screen_y], dim=1)  # (n, 2)

    return screen_coords, p_view, p_proj

def depth_offsets(p_view_z, far_scale=1.2):
    p_view_z = normalize(p_view_z)

    far_threshold = 0.3
    near_threshold = 0.7

    far_size, near_size = (p_view_z < far_threshold).sum().item(), (p_view_z > near_threshold).sum().item()

    far_near_ratio = float(far_size) / near_size
    # print(near_far_ratio)

    near_scale = 1. + far_near_ratio * (1.0 - far_scale)
    assert(abs(far_size * far_scale + near_size * near_scale - (far_size + near_size)) < 0.0001)
    scale_tensor = torch.where(
        p_view_z < far_threshold,
        torch.tensor(far_scale, device=p_view_z.device, dtype=p_view_z.dtype), # far
        torch.where(
            p_view_z > near_threshold,
            torch.tensor(near_scale, device=p_view_z.device, dtype=p_view_z.dtype), # near
            torch.tensor(1.0, device=p_view_z.device, dtype=p_view_z.dtype)
        )
    )
    return scale_tensor, scale_tensor.mean()


def compute_gaussian_value(xy : torch.Tensor, r, s):
    """
    Args:
        xy: 二维坐标
        r: 旋转 (B, 1)
        s: 缩放 (B, 2)
    return
        res: 高斯分布对应的值
    """
    cov = build_scaling_rotation(s, r)

    cov_inv = cov
    res = torch.exp(0.5 * xy.T * cov * xy)
    return res
