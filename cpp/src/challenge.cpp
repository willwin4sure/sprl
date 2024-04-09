#include <iostream>
#include <memory>
#include <cassert>

#include "games/GameState.hpp"
#include "games/Game.hpp"
#include "games/ConnectFour.hpp"

template <class TActionSpace>
int getHumanAction(const TActionSpace& actionSpace) {
    int action = -1;
    while (action < 0 || action >= actionSpace.size() || actionSpace[action] != 1.0f) {
        std::cout << "Enter a valid action: ";
        std::cin >> action;
    }

    return action;
}

template <class TGame, class TState, class TActionSpace>
SPRL::Player play(TGame* game) {
    TState state = game->startState();

    while (!game->isTerminal(state)) {
        std::cout << game->stateToString(state) << std::endl;

        TActionSpace actionSpace = game->actionMask(state);
        std::cout << "Action mask: ";
        for (auto& action : actionSpace) {
            std::cout << action << ' ';
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
    play<SPRL::ConnectFour, SPRL::ConnectFour::State, SPRL::ConnectFour::ActionSpace>(game.get());
}