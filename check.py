import os
import sys
from arguments import ArgumentParser
import re

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--time", "-t", type=int, default = 0)
    args = parser.parse_args(sys.argv[1:])

    miss = []
    source_path = os.path.join("data", str(args.time))
    output_path = os.path.join("output", str(args.time))

    for fname in os.listdir(source_path):
        pattern = r'lightmap_(\d+)_(\d+)\.exr'
        matches = re.findall(pattern, fname)
        id, _ = matches[0]

        ply_path = os.path.join(output_path, f"{args.time}_{id}.ply")

        if not os.path.exists(ply_path):
            miss.append(int(id))

    print(miss)


