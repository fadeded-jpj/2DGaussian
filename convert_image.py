from utils.image_utils import read_image_from_data
from arguments import ArgumentParser
import sys


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--config', type=str, default="setting.json")
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--time", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="E:\\tx contest\\HPRC_Test1\\Data\\Data_HPRC")
    args = parser.parse_args(sys.argv[1:])

    read_image_from_data(args.data_path, time=args.time)
