from typing import Tuple

from src.games.game import Game, GameState

from src.agents.agent import Agent

def play(game: Game, agents: Tuple[Agent, Agent], do_print: bool = False):
    """
    Play a game between two agents and returns the winner.
    """
    state: GameState = game.start_state()

    # main game loop
    while not game.is_terminal(state):
        if do_print:
            print(f"Current state:\n{game.display_state(state)}\n")
            print(f"Player {state.player}'s turn")

        action = agents[state.player].action(game, state)

        state = game.next_state(state, action)

    if do_print:
        print(f"Final state:\n{game.display_state(state)}\n")
        if state.winner == -1:
            print("Game ended in a draw.")
        else:
            print(f"Player {state.winner} wins!")

        print(f"Rewards: {game.rewards(state)}")
        
    return state.winner
