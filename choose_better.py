from argparse import ArgumentParser
import torch
from scene.gaussian_models_for_render import Model
from scene import Scene_for_Render
from renderer import render_for_rec
import os
import plyfile
import numpy as np
from Utils import cal_lpips, cal_psnr, cal_ssim
import json
import sys
from utils.image_utils import write_exr_image

psnr_worst_value = 20.0
psnr_best_value = 55.0
ssim_worst_value = 0.73
ssim_best_value = 0.999
lpips_worst_value = 0.3
lpips_best_value = 0.0003

def get_size_score(file_path, source_name):
    compression_worst_value=0.5
    compression_best_value=0.
    source_path = os.path.join("../HPRC_Test1/Data/Data_HPRC/Data", source_name)
    data_size= os.path.getsize(source_path)
    model_size = os.path.getsize(file_path)

    compression_score = (compression_worst_value - np.clip(model_size / data_size, compression_best_value, compression_worst_value)) / (compression_worst_value - compression_best_value) * 100

    return compression_score * 0.2

def get_image_score(ply_path, lightmap, time, id, resolution, mask):
    psnr_list = []
    ssim_list = []
    lpips_list = []

    primitive = Model()
    scene = Scene_for_Render(ply_path, primitive, time=time, id=id)
    lightmap_reconstruct = render_for_rec(primitive, None, torch.zeros(3), resolution)["render"].cuda().unsqueeze(0)
    lightmap_reconstruct[:, :, mask <= 0] = 0

    if id == 0:
        write_exr_image("output", lightmap_reconstruct[0], "render.exr")
        write_exr_image("output", lightmap[0], "gt.exr")

    part_size = 256
    rows = (lightmap.shape[2] + part_size - 1) // part_size
    cols = (lightmap.shape[3] + part_size - 1) // part_size

    for i in range(rows):
        for j in range(cols):
            start_row = i * part_size
            end_row = min((i + 1) * part_size, lightmap.shape[2])
            start_col = j * part_size
            end_col = min((j + 1) * part_size, lightmap.shape[3])

            lightmap_part = lightmap[:, :, start_row:end_row, start_col:end_col]
            lightmap_reconstruct_part = lightmap_reconstruct[:, :,start_row:end_row, start_col:end_col]
            mask_part = mask[start_row:end_row, start_col:end_col]
            valid_mask = mask_part >= 127

            if (np.any(valid_mask) and lightmap_part.max() != 0):
                psnr_list.append(cal_psnr(lightmap_part, lightmap_reconstruct_part, mask_part))
                ssim_list.append(cal_ssim(lightmap_part, lightmap_reconstruct_part))
                lpips_list.append(cal_lpips(lightmap_part, lightmap_reconstruct_part))


    psnr_score = ((np.clip(np.mean(psnr_list), psnr_worst_value, psnr_best_value) - psnr_worst_value) / (psnr_best_value - psnr_worst_value)) * 100
    ssim_score = ((np.clip(np.mean(ssim_list), ssim_worst_value, ssim_best_value) - ssim_worst_value) / (ssim_best_value - ssim_worst_value)) * 100
    lpips_score = ((lpips_worst_value - np.clip(np.mean(lpips_list), lpips_best_value, lpips_worst_value)) / (lpips_worst_value - lpips_best_value)) * 100

    return 0.05 * psnr_score + 0.05 * ssim_score + 0.05 * lpips_score

def get_score(pri_path, time, id, source_name, mask, resolution, gt_image):
    ply_path = os.path.join(pri_path)

    compression_score = get_size_score(ply_path, source_name)
    # compression_score = 0.
    image_score = get_image_score(ply_path, gt_image, time, id, resolution, mask)

    print(f"Compression Score:{compression_score}", f" image score:{image_score}")

    return compression_score + image_score

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--time", "-t", type=str, default='0')
    args = parser.parse_args(sys.argv[1:])

    dataset_path = f'../HPRC_Test1/Data/Data_HPRC'
    config_file = 'config.json'
    with open(os.path.join(dataset_path, config_file), 'r', encoding='utf-8') as f:
        data = json.load(f)
    lightmap_count = data['lightmap_count']

    # 这里填两个不同的路径
    ply_path_1 = os.path.join("output", args.time)
    ply_path_2 = ""
    output_path = ""

    for lightmap_config in data['lightmap_list']:
        # 从配置文件中获取lightmap的id、lightmap路径、mask路径、分辨率
        lightmap_names = lightmap_config['lightmaps']
        mask_names = lightmap_config['masks']
        resolution = lightmap_config['resolution']

        # 计算数据大小，3通道，24个时刻，每个通道4个字节
        data_size = resolution['height'] * resolution['width'] * 3 * 4  # bit
        id = lightmap_config.get('id')

        lightmap_path = os.path.join(dataset_path, "Data", lightmap_names[args.time])
        mask_path = os.path.join(dataset_path, "Data", mask_names[args.time])

        lightmap_data = np.fromfile(lightmap_path, dtype=np.float32)
        mask_data = np.fromfile(mask_path, dtype=np.int8)
        lightmap = lightmap_data.reshape(resolution['height'], resolution['width'], 3)
        mask = mask_data.reshape(resolution['height'], resolution['width'])

        # 每256*256分辨率计算一次指标，最后取平均值
        # 我们在计算指标的时候不会考虑无效的像素，所以你可以在你的训练中不考虑这些无效像素

        lightmap = torch.from_numpy(lightmap).permute(2, 0, 1).unsqueeze(0).to("cuda")

        resolu = [resolution['height'], resolution['width']]

        with torch.no_grad():
            score1 = get_score(ply_path_1, args.time, id, lightmap_names[args.time], mask, resolu, lightmap)
            score2 = get_score(ply_path_2, args.time, id, lightmap_names[args.time], mask, resolu, lightmap)
            ply_name = f"{args.time}_{id}.ply"

            if score1 >= score2:
                target_path = os.path.join(ply_path_1, ply_name)
            else:
                target_path = os.path.join(ply_path_2, ply_name)

            os.system(f"copy {target_path} {output_path}")
