"""
pentago_worker.py
"""

import os
import sys
import time

from src.interface.run_self_play import run_self_play

from src.pentago_constants import (
    NUM_ITERS,
    EXEC_PATH,

    NUM_WORKER_TASKS,
    NUM_GROUPS,

    NUM_GAMES_PER_WORKER,
    UCT_ITERATIONS,
    MAX_TRAVERSALS,
    MAX_QUEUE_SIZE,

    INIT_NUM_GAMES_PER_WORKER,
    INIT_UCT_ITERATIONS,
    INIT_MAX_TRAVERSALS,
    INIT_MAX_QUEUE_SIZE,

    RUN_NAME,
)

# def get_latest_model_path() -> str:
#     model_dir = f"data/models/{RUN_NAME}"

#     model_files = os.listdir(model_dir)
#     model_files = [f for f in model_files if f.startswith("{RUN_NAME}_iteration_") and f.endswith(".pt")]
#     model_files = sorted(model_files, key=lambda f: int(f.split("_")[-1].split(".")[0]))

#     return os.path.join(model_dir, model_files[-1])

def wait_model_path(iteration: int) -> str:
    """
    Gets the model path, when the model is actually ready.
    """

    if iteration == -1:
        return "random"
    
    while not os.path.exists(f"data/models/{RUN_NAME}/traced_{RUN_NAME}_iteration_{iteration}.pt"):
        print(f"Spinning on traced model from iteration {iteration}...")
        time.sleep(5)

    time.sleep(5)
    
    return f"data/models/{RUN_NAME}/traced_{RUN_NAME}_iteration_{iteration}.pt"


def main():
    assert len(sys.argv) == 3, "Usage: python pentago_worker.py <task_id> <num_tasks>"

    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])

    assert num_tasks == NUM_WORKER_TASKS

    my_group = my_task_id // (num_tasks // NUM_GROUPS)

    print(f"I am worker {my_task_id} in group {my_group}.")

    thread_save_path = f"data/games/{RUN_NAME}/{my_group}/{my_task_id}"
    os.makedirs(thread_save_path, exist_ok=True)

    for iteration in range(NUM_ITERS):
        print(f"Iteration {iteration}...")

        network_path = wait_model_path(iteration - 1)
        num_games = INIT_NUM_GAMES_PER_WORKER if iteration == 0 else NUM_GAMES_PER_WORKER
        uct_iterations = INIT_UCT_ITERATIONS if iteration == 0 else UCT_ITERATIONS
        max_traversals = INIT_MAX_TRAVERSALS if iteration == 0 else MAX_TRAVERSALS
        max_queue_size = INIT_MAX_QUEUE_SIZE if iteration == 0 else MAX_QUEUE_SIZE

        run_self_play(EXEC_PATH, network_path,
                      f"{thread_save_path}/{RUN_NAME}_iteration_{iteration}",
                      num_games, uct_iterations, max_traversals, max_queue_size,
                      do_print_tqdm=True)


if __name__ == "__main__":
    main()
