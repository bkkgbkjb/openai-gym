import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--c", type=int, default=35)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--env", type=str, default="antmaze-umaze-diverse-v2")
# parser.add_argument('--exp-name', type=str, default="")
parser.add_argument('--log-video', type=bool, default=False)
parser.add_argument("--high-discount", type=float, default=0.99)


args = parser.parse_args()

exp_name = json.dumps(vars(args), separators=(",", ":"))
