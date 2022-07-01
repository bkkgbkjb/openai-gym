import setup
import plotly.graph_objects as go
import h5py
import random
from utils.episode import Episodes
import numpy as np
import os
import torch
from typing import List
from tqdm import tqdm

episodes: List[Episodes] = []

NAME = './AntFall.hdf5'
with h5py.File(NAME, 'r') as f:
    s = torch.from_numpy(np.asarray(f['state']))
    a = torch.from_numpy(np.asarray(f['action']))
    r = np.asarray(f['reward'])
    d = np.asarray(f['done'])
    pos = np.asarray(f['info']['pos'])
    ns = torch.from_numpy(np.asarray(f['next_state']))

states = []
actions = []
rewards = []
infos = []

for (s, a, r, d, pos, ns) in tqdm(zip(s, a, r, d, pos, ns), total=len(s)):

    states.append(s)
    actions.append(a)
    rewards.append(r)
    infos.append(dict(pos=pos, end=False))

    if d:
        states.append(ns)
        infos.append(dict(pos=ns[:3], end=True))
        episodes.append(Episodes.from_list((states, actions, rewards, infos)))

        states = []
        actions = []
        rewards = []
        infos = []

idxs = np.random.choice(len(episodes), size=100)

batch = []
for i in idxs:
    batch.append(episodes[i])

fig = go.Figure()
for i, e in enumerate(batch):
    x = [s.info['pos'][0] for s in e.steps]
    y = [s.info['pos'][1] for s in e.steps]
    if 'Fall' in NAME:
        z = [s.info['pos'][2] for s in e.steps]
    fig.add_trace((go.Scatter if not "Fall" in NAME else go.Scatter3d)(
        x=x, y=y, z=z if "Fall" in NAME else None, mode='lines', name=str(i)))

fig.show()