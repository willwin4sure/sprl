"""
run_self_play.py

This file contains functionality for running a C++ executable
to generate self-play data.
"""

from tqdm import tqdm
import subprocess

def run_self_play(exec_path: str, model_path: str, save_path: str, num_games: int, num_iters: int, max_traversals: int, max_queue_size: int, symmetrize: bool, use_parent_q: bool, do_print_tqdm: bool):
    """
    Args:
        exec_path (str): path to the compiled cpp executable, e.g. something like "/build/Release/C4SelfPlay.exe"
        model_path (str): path to the traced version of the PyTorch model, or "random" for a uniform policy
        save_path (str): save path for the data
        num_games (int): number of games to play
        num_iters (int): number of iterations of UCT search to run
        max_traversals (int): maximum number of traversals per batch
        max_queue_size (int): maximum number of NN evals per batch
        symmetrize (bool): whether to symmetrize the data
        use_parent_q (bool): whether to use the parent's Q value
        do_print_tqdm (bool): whether to print a tqdm progress bar
    """
    process = subprocess.Popen([exec_path, model_path, save_path,
                                str(num_games), str(num_iters), str(max_traversals), str(max_queue_size), "1" if symmetrize else "0", "1" if symmetrize else "0", "1" if use_parent_q else "0"],
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
                        if "Generating self-play data: " in output:
                            # Syncing up with the internal C++ tqdm.
                            vals = output.strip().split(" ")

                            pbar.n = int(vals[-2]) + 1
                            pbar.refresh()
                            pbar.set_description(f"Number of states: {vals[-1]}")

                        else:
                            # Printing any other output.
                            print()
                            print(output.strip())
                            
                _, stderr = process.communicate()
                if stderr:
                    print("Errors from executable:")
                    print(stderr.strip())
                    
            except Exception as e:
                print(f"An error occurred: {e}")

    print("Exit Status:", process.returncode)
