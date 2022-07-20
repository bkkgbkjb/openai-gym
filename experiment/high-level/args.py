import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--c', type=int, default=35)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--env', type=str, default="antmaze-umaze-diverse-v2")
parser.add_argument('--exp-name', type=str, default="")
parser.add_argument('--high-discount', type=float, default=None)


args = parser.parse_args()