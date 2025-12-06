import torch
import os
import numpy as np

from scene.gaussian_models import Model
from scene import Scene_for_Render
from renderer import render


class BasicInterface:
    def __init__(self, lightmap_config, device):
        self.device = device
        self.id = lightmap_config.get('id')
        self.level = lightmap_config.get('level')

        self.scene = {}
        # test_time = list(range(24)) + [5.9, 18.1]
        test_time = [0]

        for time in test_time:
            time = int(time * 100)
            ply_path = os.path.join("output", str(time))
            # ply_path = os.path.join("supplement", 'test')

            primitives = Model()
            self.scene[time] = Scene_for_Render(ply_path, primitives, time=time, id=self.id, level=self.level)

        resolution = lightmap_config['resolution']
        self.resolution = resolution
        self.height = resolution['height']
        self.width = resolution['width']

    def reconstruct(self, current_time):
        H, W = self.height, self.width

        self.result = render(self.scene[current_time].primitives, None, torch.zeros(3).to(self.device), [H, W])["render"].to(self.device)

    def get_result(self):
        print("self.result shape:", self.result.shape)
        return self.result.unsqueeze(0)

    def random_test(self, coord):
        # print("self.result shape:", self.result.shape)
        result = self.result[:, coord[:, 0], coord[:, 1]]
        return result.permute(1, 0)


def get(lightmap_config, device):
    return BasicInterface(lightmap_config, device)
