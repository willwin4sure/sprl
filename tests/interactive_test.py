"""
interactive_test.py

This is a script that allows you to play any game interactively
against yourself, in order to test it.
"""


from src.games.connect_k import ConnectK
from src.games.game import Game, GameState


def play_game(game: Game):
    """
    Play a game interactively against yourself.
    """
    state: GameState = game.start_state()

    # Main game loop
    while not game.is_terminal(state):
        print(f"Current state:\n{game.display_state(state)}\n")
        print(f"Player {state.player}'s turn")
        print(f"Legal action mask: {game.action_mask(state)}")

        action = int(input("Enter action: "))

        state = game.next_state(state, action)

    print(f"Final state:\n{game.display_state(state)}\n\n")
    if state.winner == -1:
        print("Game ended in a draw.")
    else:
        print(f"Player {state.winner} wins!")

    print(f"Rewards: {game.rewards(state)}")


if __name__ == "__main__":
    connect4 = ConnectK()
    play_game(connect4)
