import os
from arguments import ModelParams, ArgumentParser, OptimizationParams, PipelineParams
from utils.image_utils import read_image_from_data
import re


times = [0, 100, 200, 300, 400, 500, 590, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1810, 1900, 2000, 2100, 2200, 2300]
# times = [0, 590]

if __name__ == "__main__":
    for time in times:
        data_path = os.path.join("..", "HPRC_Test1", "Data", "Data_HPRC") # 到Data_HPRC的路径
        read_image_from_data(data_path, time)

        save_path = os.path.join("data", str(time))

        for fname in os.listdir(save_path):
            pattern = r'lightmap_(\d+)_(\d+)\.exr'
            matches = re.findall(pattern, fname)
            id, _ = matches[0]

            os.system(f"python train.py -s {save_path} -m output\{time} --id {id} --time {time}")
