import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='hopper')
parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
parser.add_argument('--K', type=int, default=20)
parser.add_argument('--pct_traj', type=float, default=1.)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
parser.add_argument('--embed_dim', type=int, default=128)
parser.add_argument('--n_layer', type=int, default=3)
parser.add_argument('--n_head', type=int, default=1)
parser.add_argument('--activation_function', type=str, default='relu')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
parser.add_argument('--warmup_steps', type=int, default=10000)
parser.add_argument('--num_eval_episodes', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--max_iters', type=int, default=100)
parser.add_argument('--num_steps_per_iter', type=int, default=1000)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
parser.add_argument('--double-x', type=bool, default=False)

args = parser.parse_args()
exp_name = json.dumps(vars(args), indent=4, sort_keys=True)