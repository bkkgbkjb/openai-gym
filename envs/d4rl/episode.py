import gym
import setup
import d4rl
import numpy as np
from utils.env_sb3 import flat_to_episode
import plotly.graph_objects as go

env = gym.make('antmaze-umaze-diverse-v2')

ds = env.get_dataset()

end_index = (np.argwhere(ds["timeouts"] == True))[-1].item() + 1

init = (0.0, 0.0)

states = ds["observations"][:end_index]
actions = ds["actions"][:end_index]
pos = ds["observations"][:end_index, :2]
goals = ds["infos/goal"][:end_index]

rewards = ds["rewards"][:end_index]

dones = ds["timeouts"][:end_index]
infos = [dict(pos=pos[i], goal=goals[i]) for i in range(len(pos))]

episodes = flat_to_episode(states, actions, rewards, dones, infos)

start_pos = np.asarray([e[0].state[:2].numpy() for e in episodes])
end_pos = np.asarray([e.last_state[:2].numpy() for e in episodes])

fig = go.Figure()
for i in range(start_pos.shape[0]):
	fig.add_trace(go.Scatter(x=[start_pos[i, 0], end_pos[i, 0]], y=[start_pos[i, 1], end_pos[i, 1]], mode='lines+markers'))
fig.show()
