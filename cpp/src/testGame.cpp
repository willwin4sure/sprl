#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>

#include "games/GameState.hpp"
#include "games/Game.hpp"
#include "games/ConnectFour.hpp"
#include "games/Pentago.hpp"

int getPentagoAction(SPRL::Pentago::ActionDist& actionSpace) {
    int action = -1;
    while (action < 0 || action >= actionSpace.size() || actionSpace[action] != 1.0f) {
        std::cout << "Enter a square, followed by a quadrant, then a rotation (e.g. C5 0 1): ";
        char file, rank;
        int quadrant, rotation;
        std::cin >> file >> rank >> quadrant >> rotation;
        
        int boardIdx = (file - 'A') + (rank - '0') * 6;
        action = boardIdx + quadrant * 36 + rotation * 36 * 4;
    }

    return action;
}

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

    while (!state.isTerminal()) {
        std::cout << game->stateToString(state) << '\n';

        ActionDist actionMask = game->actionMask(state);

        for (int symmetry = 0; symmetry < 8; ++symmetry) {
            int8_t sym = static_cast<int8_t>(symmetry);
            ActionDist symmetrizedActionMask = game->symmetrizeActionDist(actionMask, {sym})[0];
            ActionDist symmetrizedActionMask2 = game->actionMask(game->symmetrizeState(state, {sym})[0]);

            // Assert they are the same
            for (int i = 0; i < ACTION_SIZE; ++i) {
                assert(symmetrizedActionMask[i] == symmetrizedActionMask2[i]);
            }
        }

        std::cout << "Action mask: ";
        for (auto& action : actionMask) {
            std::cout << action << ' ';
        }
        std::cout << '\n';

        int action = getPentagoAction(actionMask);

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
