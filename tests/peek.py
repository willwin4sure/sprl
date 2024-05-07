import torch
import numpy as np

RUN_NAME = "iguana_retry"

all_states = []
all_distributions = []
all_outcomes = []
all_timestamps = []

timestamp = 1
for i in range(1):
    states = torch.Tensor(np.load(f"./data/games/{RUN_NAME}/{RUN_NAME}_iteration_{i}_states.npy"))
    distributions = torch.Tensor(np.load(f"./data/games/{RUN_NAME}/{RUN_NAME}_iteration_{i}_distributions.npy"))
    outcomes = torch.Tensor(np.load(f"./data/games/{RUN_NAME}/{RUN_NAME}_iteration_{i}_outcomes.npy"))
    timestamps = torch.Tensor([timestamp for _ in range(states.shape[0])])
    
    timestamp += 1
    
    all_states.append(states)
    all_distributions.append(distributions)
    all_outcomes.append(outcomes)
    all_timestamps.append(timestamps)

state_tensor = torch.cat(all_states, dim=0)
distribution_tensor = torch.cat(all_distributions, dim=0)
outcome_tensor = torch.cat(all_outcomes, dim=0).unsqueeze(1)
timestamp_tensor = torch.cat(all_timestamps, dim=0).unsqueeze(1)

for i in range(17):
    print(state_tensor[i])
    print(distribution_tensor[i])
    print(outcome_tensor[i])
    print(timestamp_tensor[i])
