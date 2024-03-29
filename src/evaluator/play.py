from typing import Tuple

from src.games.game import Game, GameState

from src.agents.agent import Agent

def play(game: Game, agents: Tuple[Agent, Agent]):
    """
    Play a game between two agents.
    """
    state: GameState = game.start_state()

    # main game loop
    while not game.is_terminal(state):
        # print(f"Current state:\n{game.display_state(state)}\n")
        # print(f"Player {state.player}'s turn")

        # action_mask = game.action_mask(state)
        # print(f"Legal action mask: {action_mask}")

        action = agents[state.player].action(game, state)

        state = game.next_state(state, action)

    return state.winner

    # print(f"Final state:\n{game.display_state(state)}\n\n")
    # if state.winner == -1:
    #     print("Game ended in a draw.")
    # else:
    #     print(f"Player {state.winner} wins!")

    # print(f"Rewards: {game.rewards(state)}")
