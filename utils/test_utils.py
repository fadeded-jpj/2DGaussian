import numpy as np
import torch
import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure
import os
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_fn = lpips.LPIPS(net='alex').to(device)

def cal_psnr(lightmap, lightmap_reconstruct, mask=None):
    mse = torch.mean((lightmap[:, :, :] - lightmap_reconstruct[:, :, :]) ** 2)
    print("mes:", mse)
    max_value = torch.max(lightmap[:, :, :])
    print("max value:", max_value)
    psnr = 10 * torch.log10(max_value ** 2 / mse)
    return psnr.item()

def cal_ssim(lightmap, lightmap_reconstruct):
    with torch.no_grad():
        metric = StructuralSimilarityIndexMeasure(data_range=lightmap.max() - lightmap.min()).to(device)
        return metric(lightmap, lightmap_reconstruct).item()


def cal_lpips(lightmap, lightmap_reconstruct):
    with torch.no_grad():
        def normalize(image):
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                return (image - img_min) / (img_max - img_min)
            else:
                return image

        lpips_value = lpips_fn(normalize(lightmap), normalize(lightmap_reconstruct)).item()
        return lpips_value

def get_folder_size(folder_path):
    total_size = 0
    try:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    file_size = os.path.getsize(file_path)
                    total_size += file_size
                except OSError:
                    print(f"无法访问文件: {file_path}")
                    continue
    except OSError:
        print(f"无法访问文件夹: {folder_path}")
        return None

    return total_size

def extract_numeric_value(value_str):
    if isinstance(value_str, (int, float)):
        return float(value_str)

    value_str = str(value_str).strip()

    numeric_part = ""
    for i, char in enumerate(value_str):
        if char.isdigit() or char == '.' or (i == 0 and char in ['+', '-']):
            numeric_part += char
        else:
            break

    if numeric_part:
        numeric_value = float(numeric_part)

        if '%' in value_str:
            numeric_value = numeric_value / 100.0

        return numeric_value
    else:
        raise ValueError(f"Cannot extract numeric value from: {value_str}")

def test(gt, image):
    print("gt shape:", gt.shape)
    print("image shape", image.shape)
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        psnr_best_value = extract_numeric_value(config['metrics']['psnr']['best_value'])
        psnr_worst_value = extract_numeric_value(config['metrics']['psnr']['worst_value'])
        ssim_best_value = extract_numeric_value(config['metrics']['ssim']['best_value'])
        ssim_worst_value = extract_numeric_value(config['metrics']['ssim']['worst_value'])
        lpips_best_value = extract_numeric_value(config['metrics']['lpips']['best_value'])
        lpips_worst_value = extract_numeric_value(config['metrics']['lpips']['worst_value'])
        time_best_value = extract_numeric_value(config['metrics']['inference time']['best_value'])
        time_worst_value = extract_numeric_value(config['metrics']['inference time']['worst_value'])
        compression_best_value = extract_numeric_value(config['metrics']['compression ratio']['best_value'])
        compression_worst_value =extract_numeric_value(config['metrics']['compression ratio']['worst_value'])

    psnr_list = []
    ssim_list = []
    lpips_list = []
    part_size = 256
    rows = (gt.shape[2] + part_size - 1) // part_size
    cols = (gt.shape[3] + part_size - 1) // part_size
    print("rows:", rows)
    print("cols:", cols)
    for i in range(rows):
        for j in range(cols):
            start_row = i * part_size
            end_row = min((i + 1) * part_size, gt.shape[2])
            start_col = j * part_size
            end_col = min((j + 1) * part_size, gt.shape[3])

            gt_part = gt[:, :, start_row:end_row, start_col:end_col]
            image_part = image[:, :,start_row:end_row, start_col:end_col]

            psnr_list.append(cal_psnr(gt_part, image_part))
            ssim_list.append(cal_ssim(gt_part, image_part))
            lpips_list.append(cal_lpips(gt_part, image_part))

    print("PSNR:", psnr_list)
    print("SSIM:", ssim_list)
    print("LPIPS:", lpips_list)

    psnr_score = ((np.clip(np.mean(psnr_list), psnr_worst_value, psnr_best_value) - psnr_worst_value) / (psnr_best_value - psnr_worst_value)) * 100
    ssim_score = ((np.clip(np.mean(ssim_list), ssim_worst_value, ssim_best_value) - ssim_worst_value) / (ssim_best_value - ssim_worst_value)) * 100
    lpips_score = ((lpips_worst_value - np.clip(np.mean(lpips_list), lpips_best_value, lpips_worst_value)) / (lpips_worst_value - lpips_best_value)) * 100

    print("\n psnr:{}".format(np.mean(psnr_list)))
    print("\n ssim:{}".format(np.mean(ssim_list)))
    print("\n lpips:{}".format(np.mean(lpips_list)))
    print("get score:", psnr_score * 0.05 + 0.05 * ssim_score + lpips_score * 0.05)
