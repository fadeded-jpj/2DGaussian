from argparse import ArgumentParser
import sys
import numpy as np
import os
from plyfile import PlyData, PlyElement
from utils.general_utils import inverse_sigmoid
import torch
from utils.system_utils import mkdir_p

def read_ply(path):
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

    return xyz, rgb, opacities, scales, rots

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--time", "-t", type=int, default = 0)
    parser.add_argument("--plypath", "-p", type=str, default="")
    parser.add_argument("--outputpath", "-o", type=str, default="")
    args = parser.parse_args(sys.argv[1:])

    type_list = ['x', 'y', 'r', 'g', 'b', 'scale_0', 'scale_1', 'rot_0']

    os.makedirs(args.outputpath, exist_ok=True)

    for fname in os.listdir(args.plypath):
        if fname.endswith('.ply') is False:
            continue
        ply_path = os.path.join(args.plypath, fname)
        save_path = os.path.join(args.outputpath, fname)

        xyz, rgb, opacities, scales, rots = read_ply(ply_path)
        print(rgb.size)
        print(opacities.size)
        rgb = torch.from_numpy(rgb).reshape(-1, 3).float().cuda() * torch.sigmoid(torch.from_numpy(opacities).float().cuda())
        rgb = rgb.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in type_list]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, rgb, scales, rots), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(save_path)

