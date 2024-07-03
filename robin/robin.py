#!/usr/bin/env python3
"""
interactive_test.py

This is a script that allows you to play any game interactively
against yourself, in order to test it.
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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


def nickName(teamName, iteration):
    """
    E.g. the iteration of panda_delta_replicate_slow_new_prime iteration 20 should be pdrsnp20.
    Split the teamName by underscores, take the first letter of each word, and append the iteration number.
    """
    nick = ""
    idx = 0
    while idx < len(teamName):
        nick += teamName[idx]
        while idx < len(teamName) and teamName[idx] != "_":
            idx += 1
        while idx < len(teamName) and teamName[idx] == "_":
            idx += 1
    nick += str(iteration)
    return nick


class Elo(nn.Module):
    def __init__(self, num_players, freeze_first=True):
        """
        freeze_first: If True, the first player's ELO is fixed at 1000.
        """
        super(Elo, self).__init__()
        self.elos = nn.Parameter(torch.ones(num_players - 1) * 1000)
        self.freeze_first = freeze_first

    def get_elos(self):
        if self.freeze_first:
            return torch.cat([torch.tensor([1000]), self.elos])
        else:
            return self.elos


def getEloLogprob(elos: torch.Tensor, total_scores: torch.Tensor):
    """
    Given a win_matrix, calculate the log probability of the win_matrix given the elos.
    """
    assert elos.shape[0] == total_scores.shape[0] == total_scores.shape[1]
    # elo_diffs[i][j] = elos[i] - elos[j]
    elo_diffs = elos[:, None] - elos[None, :]
    log_probs = torch.log(1 / (1 + torch.exp(-elo_diffs / 400)))
    return torch.sum(total_scores * log_probs)


def converge_on_elos(elo_model, total_scores, lr=0.1, max_iter=1000):
    """
    Given an Elo model and a win matrix, converge on the ELOs that maximize the likelihood of the win matrix.

    Modifies the elo model in place.
    """
    optimizer = torch.optim.Adam(elo_model.parameters(), lr=lr)
    for i in range(max_iter):
        optimizer.zero_grad()
        logprob = getEloLogprob(elo_model.get_elos(), total_scores)
        loss = -logprob
        loss.backward()
        optimizer.step()
    return elo_model.get_elos()


def plot_heatmap(total_scores, players, results_path):
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

    plt.savefig(results_path + "_heatmap.png", dpi=300)
    plt.close()


def compute_plot_elos(elo_model, team_names, player_iterations, total_scores, results_path, iterations=10):
    # Now, do an ELO computation for each player.
    elos = converge_on_elos(elo_model, total_scores)
    elos = elos.detach().numpy()
    idx = 0

    plt.figure()
    plt.title("ELOs of players")
    plt.xlabel("Iterations")
    plt.ylabel("ELO")

    for team, iterations in zip(team_names, player_iterations):
        tmp = []
        for iteration in iterations:
            tmp.append(elos[idx])
            idx += 1

        # plot iterations against ELO
        plt.plot(iterations, tmp, label=team, marker="o")

    plt.legend()
    plt.savefig(results_path + "_elo.png", dpi=300)
    plt.close()


def handle_master(num_games, num_workers, group_size, robin_config_path,
                  heatmap=True, elo=True, live_heatmap=True, live_elo=True):

    print("I am responsible for checking the results periodically and writing them all to a big file.")

    with open(robin_config_path, "r") as f:
        config = "\n".join(f.readlines())
    config = config.split()

    tournament_name = config[0]
    num_teams = int(config[1])
    team_names = []
    player_iterations = []
    idx = 2
    total_players = 0
    for _ in range(num_teams):
        team_names.append(config[idx])
        num_players = int(config[idx + 1])
        total_players += num_players
        idx += 2
        player_iterations.append([])
        for _ in range(num_players):
            player_iterations[-1].append(int(config[idx]))
            idx += 1

    results_path = f"./data/robin/{tournament_name}"

    players = []

    # The ELO of the random player is fixed at 1000; this is the baseline.
    elo_model = Elo(total_players)

    for team, iterations in zip(team_names, player_iterations):
        for iteration in iterations:
            players.append(nickName(team, iteration))

    results_file = f"{results_path}.txt"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # First, figure out who the players are by reading the first file.
    while True:
        time.sleep(1)
        # wait until this file exists
        if not os.path.exists(f"{results_path}/0/0/log.txt"):
            continue

        # total_scores = [[0 for __ in range(len(players))]
        #                 for _ in range(len(players))]
        total_scores = torch.zeros(len(players), len(players))

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
        win_matrix = {player: {opponent: score.item() for opponent, score in zip(
            players, scores)} for player, scores in zip(players, total_scores)}

        print("Total games played: ", total_games)
        with open(results_path + ".txt", "w") as f:
            f.write("DASHBOARD: " + tournament_name + "\n")
            f.write("-"*100+"\n")
            f.write(pretty_dict_matrix(win_matrix))
            f.write("\n\n")
            f.write(win_statistics(win_matrix))
            f.write(
                f"\n\nTotal games played: {total_games} / {num_games * len(players) * (len(players) - 1)}")

        if live_heatmap:
            plot_heatmap(total_scores, players, results_path)
        if live_elo:
            compute_plot_elos(elo_model, team_names, player_iterations,
                              total_scores, results_path)

        # Now, do an ELO computation for each player.

        if total_games >= num_games * len(players) * (len(players) - 1):
            break
    if heatmap:
        plot_heatmap(total_scores, players, results_path)
    if elo:
        compute_plot_elos(elo_model, team_names, player_iterations,
                          total_scores, results_path, iterations=1000)


if __name__ == "__main__":
    NUM_GAMES = 144
    GROUP_SIZE = 48
    NUM_TASKS = 144

    ROBIN_CONFIG_PATH = "/home/gridsan/rzhong/sprl/robin/robin_config.txt"
    handle_master(NUM_GAMES, NUM_TASKS,
                  GROUP_SIZE, ROBIN_CONFIG_PATH)
