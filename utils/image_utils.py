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
import json
import os
import numpy as np
import OpenEXR
import Imath
from matplotlib import pyplot as plt
import torch.nn.functional as F
import cv2
from torch import nn


class Image_source(nn.Module):
    def __init__(self, image : torch.Tensor, id, time ,level):
        super(Image_source, self).__init__()
        self.images = image
        self.id = id
        self.time = time
        self.level = level

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):

    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def write_exr_image(path, img : torch.Tensor, name="img.exr"):
     # 你可以将这些数据保存为exr文件以供查看：
    _, H, W = img.shape
    lightmap = img.permute(1, 2, 0).float().cpu().numpy()

    R = lightmap[:, :, 0].tobytes()
    G = lightmap[:, :, 1].tobytes()
    B = lightmap[:, :, 2].tobytes()

    exr_file = OpenEXR.OutputFile(os.path.join(path, name),
                                  OpenEXR.Header(W, H))
    exr_file.writePixels({'R': R, 'G': G, 'B': B})
    exr_file.close()

def read_image_from_data(dataset_path='E:\\tx contest\\HPRC_Test1\\Data\\Data_HPRC', time=0, config_file='config.json'):
    save_path = os.path.join("data", str(time))
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(dataset_path, config_file), 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取lightmap数量
    lightmap_count = data['lightmap_count']
    print(f"Lightmap count: {lightmap_count}")


    for lightmap in data['lightmap_list']:
        id = lightmap['id']
        level= lightmap['level']
        lightmap_names = lightmap['lightmaps']
        masks_names = lightmap['masks']
        resolution = lightmap['resolution']

        # 获取path信息
        lightmap_path = os.path.join(dataset_path, 'Data', lightmap_names[str(time)])
        mask_path = os.path.join(dataset_path, 'Data', masks_names[str(time)])

        # lightmap数据类型为float32
        lightmap_data = np.fromfile(lightmap_path, dtype=np.float32)
        # mask数据类型为int8
        mask_data = np.fromfile(mask_path, dtype=np.int8)

        # lightmap每个像素有R G B三通道
        lightmap = lightmap_data.reshape(resolution['height'], resolution['width'], 3)
        # mask每个像素有1通道
        mask = mask_data.reshape(resolution['height'], resolution['width'])
        mask = torch.from_numpy(mask)
        print(mask.shape)


        # mask数据为-1时，表示该数据为无效数据，为127时，表示该数据为有效数据
        # 获取有效lightmap数据可以这样做：
        # valid_lightmap = lightmap[mask >= 127]

        # 你可以将这些数据保存为exr文件以供查看：
        R = lightmap[:, :, 0].tobytes()
        G = lightmap[:, :, 1].tobytes()
        B = lightmap[:, :, 2].tobytes()

        print("R max:", lightmap[:, :, 0].max())


        exr_file = OpenEXR.OutputFile(os.path.join('data', str(time), f'lightmap_{id}_{time}__{level}.exr'),
                                  OpenEXR.Header(resolution['width'], resolution['height']))
        exr_file.writePixels({'R': R, 'G': G, 'B': B})
        exr_file.close()

def read_image_from_exr_to_tensor(exr_path):
    print("Reading EXR image from:", exr_path)
    exr_file = OpenEXR.InputFile(exr_path)
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    R = np.frombuffer(exr_file.channel('R', FLOAT), dtype=np.float32).reshape(size[1], size[0])
    G = np.frombuffer(exr_file.channel('G', FLOAT), dtype=np.float32).reshape(size[1], size[0])
    B = np.frombuffer(exr_file.channel('B', FLOAT), dtype=np.float32).reshape(size[1], size[0])

    # print("R:", R.max())
    # print("G:", G.max())
    # print("B:", B.max())

    img = np.stack([R, G, B], axis=-1)
    img = torch.from_numpy(img).float().permute(2, 0, 1)  # C, H, W
    return img

def read_images(image_path, id=0, time=0,level=""):
    img_tensor = read_image_from_exr_to_tensor(os.path.join(image_path, f'lightmap_{id}_{time}__{level}.exr'))
    img_sor = Image_source(img_tensor.cuda(), int(id), int(time),level)

    images = img_sor

    return images  # C, H, W

def get_image_color(img : Image_source, sample_xy):
    rgb = []
    res = img.images.shape
    grid = (sample_xy[:, 1] * res[1], sample_xy[:, 0] * res[2])  # N,
    grid = torch.stack(grid, dim=-1).int()  # N, 2

    image = img.images
    # 确保坐标是整数
    x_coords = grid[:, 1].long()
    y_coords = grid[:, 0].long()

    # print("x_coords:", x_coords.max())
    # print("y_coords:", y_coords.max())

    # 直接索引
    rgb_values = image[:, y_coords, x_coords].permute(1, 0)  # (N, 3)

    return rgb_values


def create_image_tensor(coords_tensor, colors_tensor, img_size, device='cuda'):
    """
    使用 PyTorch 创建图像 tensor（避免 CPU-GPU 传输）

    Parameters:
    - coords_tensor: 形状为 (N, 2) 的坐标 tensor
    - colors_tensor: 形状为 (N, 3) 的颜色 tensor
    - img_size: (height, width)
    - device: 设备

    Returns:
    - image_tensor: 形状为 (3, H, W) 的图像 tensor
    """
    H, W = img_size

    # 创建空白图像
    image_tensor = torch.zeros(3, H, W, device=device)

    # 确保坐标是整数且在有效范围内
    coords_tensor[:, 0] *= img_size[1]  # width
    coords_tensor[:, 1] *= img_size[0]  # height
    coords_int = coords_tensor.long()

    # 创建有效掩码
    valid_mask = (coords_int[:, 0] >= 0) & (coords_int[:, 0] < W) & \
                 (coords_int[:, 1] >= 0) & (coords_int[:, 1] < H)

    coords_valid = coords_int[valid_mask]
    colors_valid = colors_tensor[valid_mask]

    # 将颜色值赋给图像（注意坐标顺序）
    # coords_valid[:, 0] 是 x, coords_valid[:, 1] 是 y
    image_tensor[:, coords_valid[:, 1], coords_valid[:, 0]] = colors_valid.T

    return image_tensor


def plot_tensor_image(image_tensor):
    """
    绘制 PyTorch tensor 图像
    """
    # 转换为 numpy 并调整维度
    if image_tensor.dim() == 3 and image_tensor.shape[0] == 3:
        # (3, H, W) -> (H, W, 3)
        image_np = image_tensor.cpu().permute(1, 2, 0).numpy()
    else:
        image_np = image_tensor.cpu().numpy()

    # 如果值范围是 [0,1]，转换为 [0,255]
    image_np = (image_np * 255).astype(np.uint8)


    plt.figure(figsize=(12, 8))
    plt.imshow(image_np)
    plt.axis('off')
    plt.title(f"Image from Tensor")
    plt.show()


def exr_to_png_opencv(exr_tensor, output_path=None):
    """
    将EXR tensor转换为PNG格式

    Args:
        exr_tensor: shape (H, W, 3) 的tensor
        output_path: 保存路径，如果为None则返回numpy数组

    Returns:
        PNG格式的numpy数组或保存文件
    """
    # 转换为numpy数组
    print("image shape:", exr_tensor.shape)
    if isinstance(exr_tensor, torch.Tensor):
        exr_array = exr_tensor.permute(1, 2, 0).cpu().numpy()
    else:
        exr_array = exr_tensor

    # 方法A: 简单的线性映射（适合范围在0-1之间的数据）
    png_array = np.clip(exr_array * 255, 0, 255).astype(np.uint8)

    # 方法B: 使用Reinhard色调映射（适合HDR数据）
    # png_array = reinhard_tonemap(exr_array)

    if output_path:
        # 注意：OpenCV使用BGR格式，需要转换
        print("save path: ", output_path)
        png_bgr = cv2.cvtColor(png_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_path, "image.png"), png_bgr)
        print(f"PNG图像已保存到: {output_path}")
    else:
        return png_array
