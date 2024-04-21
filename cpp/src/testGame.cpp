#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>

#include "games/GameState.hpp"
#include "games/Game.hpp"
#include "games/ConnectFour.hpp"
#include "games/Pentago.hpp"

template <int ACTION_SIZE>
int getHumanAction(const SPRL::GameActionDist<ACTION_SIZE>& actionSpace) {
    int action = -1;
    while (action < 0 || action >= actionSpace.size() || actionSpace[action] != 1.0f) {
        std::cout << "Enter a valid action: ";
        std::cin >> action;
    }

    return action;
}

template <int BOARD_SIZE, int ACTION_SIZE>
void testGame(SPRL::Game<BOARD_SIZE, ACTION_SIZE>* game) {
    using State = SPRL::GameState<BOARD_SIZE>;
    using ActionDist = SPRL::GameActionDist<ACTION_SIZE>;
    
    State state = game->startState();

    while (!game->isTerminal(state)) {
        std::cout << game->stateToString(state) << '\n';

        ActionDist actionMask = game->actionMask(state);
        std::cout << "Action mask: ";
        for (auto& action : actionMask) {
            std::cout << action << ' ';
        }
        std::cout << '\n';

        int action = getHumanAction(actionMask);

        state = game->nextState(state, action);
    }

    std::cout << "Game over!" << '\n';
    std::cout << game->stateToString(state) << '\n';

    std::cout << "The winner is Player " << static_cast<int>(state.getWinner()) << '\n';
    std::cout << "The rewards are " << game->rewards(state).first << " and " << game->rewards(state).second << '\n';
}

int main(int argc, char* argv[]) {
    auto game = std::make_unique<SPRL::Pentago>();
    testGame(game.get());

    return 0;
}
