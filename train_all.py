import os
from arguments import ModelParams, ArgumentParser, OptimizationParams, PipelineParams
from utils.image_utils import read_image_from_data
import re


# times = [0, 100, 200, 300, 400, 500, 590, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1810, 1900, 2000, 2100, 2200, 2300]
times = [0, 100, 200, 300, 400, 500, 590, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1810, 1900, 2000, 2100, 2200, 2300]

if __name__ == "__main__":
    for time in times:
        data_path = os.path.join("D:/resource/data/dataset/tx/Data/Data_HPRC") # 到Data_HPRC的路径
        if os.path.exists(os.path.join("data", str(time)))==False:
            print("尚未读取原始数据")
            read_image_from_data(data_path, time)

        save_path = os.path.join("data", str(time))

        for fname in os.listdir(save_path):
            pattern = r'lightmap_(\d+)_(\d+)__([a-zA-Z0-9_]+)\.exr'
            matches = re.findall(pattern, fname)
            print(matches)
            id, _ ,level= matches[0]
            file_path = os.path.join("output", str(time),f"{time}_{id}__{level}.ply")
            print(file_path)
            if os.path.exists(file_path):
                print("已经存在了"+id)
                continue

            os.system(f"python train.py -s {save_path} -m output\{time} --id {id} --time {time} --level {level}")
