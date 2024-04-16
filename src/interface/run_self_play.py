"""
run_self_play.py

This file contains functionality for running a C++ executable
to generate self-play data.
"""

from tqdm import tqdm
import subprocess

EXECUTABLE_PATH = "./cpp/build/Release/selfPlay.exe"

def run_self_play(model_path: str, save_path: str, num_games: int, num_iters: int, max_traversals: int, max_queue_size: int, do_print_tqdm: bool):
    """
    Args:
        model_path (str): path to the traced version of the PyTorch model, or "random" for a uniform policy
        save_path (str): save path for the data
        num_games (int): number of games to play
        num_iters (int): number of iterations of UCT search to run
        max_traversals (int): maximum number of traversals per batch
        max_queue_size (int): maximum number of NN evals per batch
    """
    process = subprocess.Popen([EXECUTABLE_PATH, model_path, save_path,
                                str(num_games), str(num_iters), str(max_traversals), str(max_queue_size)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if do_print_tqdm:
        with tqdm(total=num_games) as pbar:
            try:
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    
                    if output:
                        # Ad-hoc code for making the printing correct
                        if "Generating self-play data:" in output:
                            pbar.update(1)
                            pbar.set_description(f"Number of states: {output.strip().split(" ")[-1]}")
                            
                _, stderr = process.communicate()
                if stderr:
                    print("Errors from executable:")
                    print(stderr.strip())
                    
            except Exception as e:
                print(f"An error occurred: {e}")

    print("Exit Status:", process.returncode)
