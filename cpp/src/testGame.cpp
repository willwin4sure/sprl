#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>

#include "games/GameState.hpp"
#include "games/Game.hpp"
#include "games/ConnectFour.hpp"
#include "games/Pentago.hpp"

#include "random/Random.hpp"

#include "tqdm/tqdm.hpp"

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
    
    SPRL::Symmetry symmetry = static_cast<int8_t>(SPRL::GetRandom().UniformInt(0, game->numSymmetries() - 1));

    State state0 = game->startState();
    State state1 = game->symmetrizeState(game->startState(), {symmetry})[0];

    while (!state0.isTerminal()) {
        // std::cout << game->stateToString(state0) << '\n';
        // std::cout << game->stateToString(state1) << '\n';

        ActionDist actionMask = game->actionMask(state0);

        int numLegal = 0;
        for (int i = 0; i < ACTION_SIZE; ++i) {
            if (actionMask[i] == 1.0f) {
                numLegal++;
            }
        }

        int spike = SPRL::GetRandom().UniformInt(0, numLegal - 1);

        ActionDist actionValues;
        actionValues.fill(0.0f);

        int write = 0;
        for (int i = 0; i < ACTION_SIZE; ++i) {
            if (actionMask[i] == 1.0f) {
                if (write == spike) {
                    actionValues[i] = 1.0f;
                    break;
                }
                write++;
            }
        }

        // Pick maximum as action
        SPRL::ActionIdx action0 = std::distance(actionValues.begin(),
            std::max_element(actionValues.begin(), actionValues.end()));

        ActionDist symmetrizedActionValues = game->symmetrizeActionDist(actionValues, {symmetry})[0];
        SPRL::ActionIdx action1 = std::distance(symmetrizedActionValues.begin(),
            std::max_element(symmetrizedActionValues.begin(), symmetrizedActionValues.end()));

        // assert that inverse symmetrizing equals original
        ActionDist inverseSymmetrizedActionValues = game->symmetrizeActionDist(symmetrizedActionValues, {game->inverseSymmetry(symmetry)})[0];
        for (int i = 0; i < ACTION_SIZE; ++i) {
            assert(actionValues[i] == inverseSymmetrizedActionValues[i]);
        }

        state0 = game->nextState(state0, action0);
        state1 = game->nextState(state1, action1);

        SPRL::GameBoard<BOARD_SIZE> board0 = state0.getBoard();
        SPRL::GameBoard<BOARD_SIZE> board1 = game->symmetrizeState(state1, {game->inverseSymmetry(symmetry)})[0].getBoard();

        for (int i = 0; i < BOARD_SIZE; ++i) {
            if (board0[i] != board1[i]) {
                std::cout << "Error: " << i << " " << static_cast<int>(board0[i]) << " " << static_cast<int>(board1[i]) << '\n';

                // print out the game states and actions
                std::cout << game->stateToString(state0) << '\n';
                std::cout << game->stateToString(state1) << '\n';
                std::cout << game->stateToString(game->symmetrizeState(state1, {game->inverseSymmetry(symmetry)})[0]) << '\n';

                std::cout << "Action: " << static_cast<int>(action0) << " " << static_cast<int>(action1) << '\n';

                // print out the action values
                for (int i = 0; i < ACTION_SIZE; ++i) {
                    std::cout << i << " " << actionValues[i] << " " << symmetrizedActionValues[i] << '\n';
                }

                assert(false);
            }

        }
    }

    // std::cout << "Game over!" << '\n';
    // std::cout << game->stateToString(state0) << '\n';
    // std::cout << game->stateToString(state1) << '\n';

    // std::cout << "The winner is Player " << static_cast<int>(state0.getWinner()) << '\n';
    // std::cout << "The rewards are " << game->rewards(state0).first << " and " << game->rewards(state0).second << '\n';
}

int main(int argc, char* argv[]) {
    auto game = std::make_unique<SPRL::Pentago>();
    auto pbar = tq::trange(1000000);
    for (int t : pbar) {
        testGame(game.get());
    }

    // std::cout << static_cast<int>(game->actionToActionIdx(game->symmetrizeSingleAction(game->actionIdxToAction(83), 5))) << std::endl;

    return 0;
}
