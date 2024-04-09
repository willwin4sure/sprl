#include <iostream>
#include <memory>
#include <cassert>

#include "games/GameState.hpp"
#include "games/Game.hpp"
#include "games/ConnectFour.hpp"

int getHumanAction(const std::array<float, 7>& actionSpace) {
    int action = -1;
    while (action < 0 || action >= 7 || actionSpace[action] != 1.0f) {
        std::cout << "Enter a valid action: ";
        std::cin >> action;
    }

    return action;
}

template <int BOARD_SIZE, int ACTION_SIZE>
SPRL::Player play(SPRL::Game<BOARD_SIZE, ACTION_SIZE>* game) {
    SPRL::GameState<BOARD_SIZE> state = game->startState();

    while (!game->isTerminal(state)) {
        std::cout << game->stateToString(state) << std::endl;

        std::array<float, ACTION_SIZE> actionSpace = game->actionMask(state);
        std::cout << "Action mask: ";
        for (int i = 0; i < ACTION_SIZE; i++) {
            std::cout << actionSpace[i] << ' ';
        }
        std::cout << '\n';

        int action = getHumanAction(actionSpace);

        state = game->nextState(state, action);
    }

    std::cout << "Game over!" << '\n';
    std::cout << game->stateToString(state) << std::endl;

    std::cout << "The winner is Player " << static_cast<int>(state.getWinner()) << '\n';
    std::cout << "The rewards are " << game->rewards(state).first << " and " << game->rewards(state).second << '\n';
}

int main() { 
    auto game = std::make_unique<SPRL::ConnectFour>();
    play(game.get());
}