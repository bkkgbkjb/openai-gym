import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--z-loss-ratio', type=float, default=1.0)
parser.add_argument('--k', type=int, default=40)

args = parser.parse_args()
exp_name = json.dumps(vars(args), indent=4, sort_keys=True)