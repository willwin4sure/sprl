#!/usr/bin/env python3
"""
interactive_test.py

This is a script that allows you to play any game interactively
against yourself, in order to test it.
"""

import os
import sys

import torch
from tqdm import tqdm

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


def handle_master(results_path, num_games, num_workers, group_size, players):
    import time

    import matplotlib.pyplot as plt

    print("I am responsible for checking the results periodically and writing them all to a big file.")
    results_file = f"{results_path}.txt"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # First, figure out who the players are by reading the first file.
    while True:
        time.sleep(1)
        # wait until this file exists
        if not os.path.exists(f"{results_path}/0/0/log.txt"):
            continue

        total_scores = [[0 for __ in range(len(players))]
                        for _ in range(len(players))]

        total_games = 0
        for i in range(num_workers):
            results_file = f"{results_path}/{i // group_size}/{i}/log.txt"
            # wait until this file exists
            if not os.path.exists(results_file):
                continue
            try:
                with open(results_file, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip() == "":
                            continue
                        player, opponent, winner = map(int, line.split())
                        if winner == 0:
                            total_scores[player][opponent] += 1
                        elif winner == 1:
                            total_scores[opponent][player] += 1
                        else:
                            total_scores[player][opponent] += 0.5
                            total_scores[opponent][player] += 0.5
                        total_games += 1
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

        # Now, matplotlib a heatmap win matrix. This should be a 1920 x 1080 image.
        fig, ax = plt.subplots()
        im = ax.imshow(total_scores, cmap="viridis")

        # We want to show all ticks...
        ax.set_xticks(range(len(players)))
        ax.set_yticks(range(len(players)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(players)
        ax.set_yticklabels(players)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        if len(players) <= 5:  # Else, the boxes are too small.
            for i in range(len(players)):
                for j in range(len(players)):
                    text = ax.text(j, i, total_scores[i][j],
                                   ha="center", va="center", color="w", fontsize=10)

        ax.set_title("Total scores of players")
        fig.tight_layout()
        # colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Scores", rotation=-90, va="bottom")

        plt.savefig(results_path + ".png", dpi=300)
        plt.close()

        if total_games >= num_games * len(players) * (len(players) - 1):
            break


if __name__ == "__main__":
    NUM_GAMES = 192
    RESULTS_PATH = "/home/gridsan/rzhong/sprl/data/robin/panda_fight"
    GROUP_SIZE = 48
    NUM_TASKS = 192

    player_names = ['random'] + [f'ps_{i}' for i in range(
        0, 200, 10)] + [f'pa_{i}' for i in range(0, 190, 10)]

    handle_master(RESULTS_PATH, NUM_GAMES, NUM_TASKS, GROUP_SIZE, player_names)
