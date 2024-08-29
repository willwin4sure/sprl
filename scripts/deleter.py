# in folder ~/sprl/data/games/quail_gamma/i/j where i ranges from 0...8 and j ranges from 48*i...48*(i+1)
# there are files that look like
# quail_gamma_iteration_9_distributions.npy
# quail_gamma_iteration_9_outcomes.npy
# quail_gamma_iteration_9_states.npy

# iterate over all of these files and show the minimum iteration which has a file with zero filesize.

import os

for i in range(8):
    for j in range(48*i, 48*(i+1)):
        path = f"/home/gridsan/rzhong/sprl/data/games/quail_gamma/{i}/{j}"
        k = 0
        while True:
            if os.path.isfile(f"{path}/quail_gamma_iteration_{k}_distributions.npy") == False:
                print(f"Found no file at iteration {i}.{j}.{k}")
                break
            if os.path.getsize(f"{path}/quail_gamma_iteration_{k}_distributions.npy") == 0:
                print(f"Found zero size file at iteration {i}.{j}.{k}")
                break
            k += 1
        # delete all files after k
        while True:
            if os.path.isfile(f"{path}/quail_gamma_iteration_{k}_distributions.npy") == False:
                break
            os.remove(f"{path}/quail_gamma_iteration_{k}_distributions.npy")
            os.remove(f"{path}/quail_gamma_iteration_{k}_outcomes.npy")
            os.remove(f"{path}/quail_gamma_iteration_{k}_states.npy")
            k += 1

# this time, delete every file which is before iteration 400.

for i in range(8):
    for j in range(48*i, 48*(i+1)):
        path = f"/home/gridsan/rzhong/sprl/data/games/quail_gamma/{i}/{j}"
        for k in range(400):
            if os.path.isfile(f"{path}/quail_gamma_iteration_{k}_distributions.npy") == False:
                print(f"Skipping {i}.{j}.{k}")
                continue
            os.remove(f"{path}/quail_gamma_iteration_{k}_distributions.npy")
            os.remove(f"{path}/quail_gamma_iteration_{k}_outcomes.npy")
            os.remove(f"{path}/quail_gamma_iteration_{k}_states.npy")
