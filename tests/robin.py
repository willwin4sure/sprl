"""
interactive_test.py

This is a script that allows you to play any game interactively
against yourself, in order to test it.
"""

import os
import sys

import torch
from tqdm import tqdm

from src.agents.agent import Agent
from src.agents.human_agent import HumanAgent
from src.agents.policy_agent import PolicyAgent
from src.agents.random_agent import RandomAgent
from src.evaluator.play import play
from src.games.connect_k import ConnectK
from src.games.game import Game, GameState
from src.networks.connect_four_network import ConnectFourNetwork
from src.policies.monte_carlo_policy import MonteCarloPolicy
from src.policies.network_policy import NetworkPolicy
from src.policies.random_policy import RandomPolicy
from src.policies.uct_policy import UCTPolicy
from src.utils.pretty_matrix import *


def win_statistics(win_matrix):
    # Print a chart of the win matrix

    statistics = {}

    for player in win_matrix.keys():
        total_wins = sum(win_matrix[player].values())
        total_losses = sum(win_matrix[other_player][player]
                           for other_player in win_matrix.keys())
        if total_wins + total_losses == 0:
            win_rate = 0
        else:
            win_rate = total_wins / (total_wins + total_losses)
        statistics[player] = {"total_wins": total_wins,
                              "total_losses": total_losses, "win_rate": round(win_rate, 2)}

    return pretty_dict_matrix(statistics, col_seperator=None, first_row_seperator="-", first_col_seperator="|", title="Statistics")


def handle_master(results_path, num_games, num_workers):
    import time

    print("I am responsible for checking the results periodically and writing them all to a big file.")
    results_file = f"{results_path}.txt"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # First, figure out who the players are by reading the first file.
    while True:
        time.sleep(1)
        # wait until this file exists
        if not os.path.exists(f"{results_path}_0.txt"):
            continue

        players = []
        with open(f"{results_path}_0.txt", "r") as f:
            lines = f.readlines()
            first_line = lines[0].strip()
            players = first_line.split(" ")
            players = [player for player in players if player != ""]

        total_scores = [[0 for __ in range(len(players))]
                        for _ in range(len(players))]

        total_games = 0
        for i in range(num_workers):
            results_file = f"{results_path}_{i}.txt"
            # wait until this file exists
            if not os.path.exists(results_file):
                continue
            try:
                with open(results_file, "r") as f:
                    lines = f.readlines()[1:len(players) + 1]
                    lines = [line.split(" ")[1:] for line in lines]
                    scores = [[float(x) for x in line if x.strip() != ""]
                              for line in lines]
                    total_scores = [[total_scores[i][j] + scores[i][j]
                                    for j in range(len(players))] for i in range(len(players))]
                    total_games += sum(sum(score) for score in scores)
            except Exception as e:
                print("Error reading file ", results_file)
                print(e)
        win_matrix = {player: {opponent: score for opponent, score in zip(
            players, scores)} for player, scores in zip(players, total_scores)}

        with open(results_path + ".txt", "w") as f:
            f.write("DASHBOARD\n")
            f.write("-"*100+"\n")
            f.write(pretty_dict_matrix(win_matrix))
            f.write("\n\n")
            f.write(win_statistics(win_matrix))
            f.write(
                f"\n\nTotal games played: {total_games} / {num_games * len(players) * (len(players) - 1)}")
        if total_games >= num_games * len(players) * (len(players) - 1):
            break


if __name__ == "__main__":
    connect4 = ConnectK()
    NUM_GAMES = 384

    RESULTS_PATH = "data/results/round_robin_results"

    if len(sys.argv) < 3:
        my_task_id = 0
        num_tasks = 1
    else:
        my_task_id = int(sys.argv[1])
        num_tasks = int(sys.argv[2])

        if my_task_id == -1:
            handle_master(RESULTS_PATH, NUM_GAMES, num_tasks)
            sys.exit(0)

    rounds = NUM_GAMES // (num_tasks)
    if my_task_id < NUM_GAMES % (num_tasks):
        rounds += 1

    print("We are worker ", my_task_id, " and we will play ", rounds, " games.")
    results_file = f"{RESULTS_PATH}_{my_task_id}.txt"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    players = {}
    for i in range(0, 100, 10):
        elephant_net = NetworkPolicy(torch.load(
            f"data/models/dragon/dragon_iteration_{i}.pt"))
        elephant = PolicyAgent(UCTPolicy(elephant_net, num_iters=1000,
                                         c=1.0, train=False, init_type="zero"), temperature=0.5)
        players.update({f"dr_{i}": elephant})
    for i in range(0, 30, 10):
        elephant_net = NetworkPolicy(torch.load(
            f"data/models/elephant/elephant_iteration_{i}.pt"))
        elephant = PolicyAgent(UCTPolicy(elephant_net, num_iters=1000,
                                         c=1.0, train=False, init_type="equal"), temperature=0.5)
        players.update({f"el_{i}": elephant})

    electron_net = NetworkPolicy(torch.load(
        "data/models/electron/electron_iteration_20.pt"))
    electron = PolicyAgent(UCTPolicy(electron_net, num_iters=1000,
                                     c=1.0, train=False, init_type="offset"), temperature=0.5)
    random_agent = RandomAgent()
    players.update({"electron": electron, "random": random_agent})
    # play a round robin tournament among the players.

    win_matrix = {player: {opponent: 0 for opponent in players.keys()}
                  for player in players.keys()}

    for _ in tqdm(range(1)):
        with tqdm(total=len(players) * (len(players) - 1)) as pbar:
            pbar.set_description("Playing games")
            for player1 in players.keys():
                for player2 in players.keys():
                    if player1 == player2:
                        continue
                    pbar.set_description(
                        f"Playing games: {player1} vs {player2}"
                    )
                    winner = play(connect4, (players[player1],
                                             players[player2]), do_print=False)

                    if winner == 0:
                        win_matrix[player1][player2] += 1
                    elif winner == 1:
                        win_matrix[player2][player1] += 1
                    else:
                        win_matrix[player1][player2] += 0.5
                        win_matrix[player2][player1] += 0.5
                    pbar.update(1)
                    with open(results_file, "w") as f:
                        f.write(pretty_dict_matrix(win_matrix))
                        f.write("\n\n")
                        f.write(win_statistics(win_matrix))
