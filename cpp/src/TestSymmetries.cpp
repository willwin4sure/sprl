#include "agents/HumanAgent.hpp"
#include "agents/HumanGoAgent.hpp"

#include "evaluate/play.hpp"

#include "games/GameNode.hpp"
#include "games/ConnectFourNode.hpp"
#include "games/GoNode.hpp"

#include "random/Random.hpp"

#include "symmetry/D4GridSymmetrizer.hpp"

#include <array>
#include <iostream>

void printState(SPRL::GridState<SPRL::GO_BOARD_SIZE, SPRL::GO_HISTORY_SIZE> state) {
    for (int t = 0; t < state.size(); ++t) {
        for (int i = 0; i < SPRL::GO_BOARD_WIDTH; ++i) {
            for (int j = 0; j < SPRL::GO_BOARD_WIDTH; ++j) {
                switch (state.getHistory()[t][i * SPRL::GO_BOARD_WIDTH + j]) {
                case SPRL::Piece::ZERO:
                    std::cout << "O ";
                    break;
                case SPRL::Piece::ONE:
                    std::cout << "X ";
                    break;
                case SPRL::Piece::NONE:
                    std::cout << ". ";
                    break;
                }
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }
}

void printMask(SPRL::GameActionDist<SPRL::GO_ACTION_SIZE> mask) {
    for (int i = 0; i < SPRL::GO_BOARD_SIZE; ++i) {
        if (mask[i] > 0) {
            std::cout << "1 ";
        } else {
            std::cout << "0 ";
        }
    }
    std::cout << '\n';
}

int main(int argc, char* argv[]) {
    SPRL::HumanGoAgent humanAgent {};
    std::array<SPRL::Agent<SPRL::GoNode, SPRL::GridState<SPRL::GO_BOARD_SIZE, SPRL::GO_HISTORY_SIZE>, SPRL::GO_ACTION_SIZE>*, 2> agents = { &humanAgent, &humanAgent };

    SPRL::D4GridSymmetrizer<SPRL::GO_BOARD_WIDTH, SPRL::GO_HISTORY_SIZE> symmetrizer {};
    
    SPRL::GoNode rootNode {};

    SPRL::GameNode<SPRL::GoNode, SPRL::GridState<SPRL::GO_BOARD_SIZE, SPRL::GO_HISTORY_SIZE>, SPRL::GO_ACTION_SIZE>* curNode = &rootNode;

    while (!curNode->isTerminal()) {
        std::cout << curNode->toString() << '\n';

        auto state = curNode->getGameState();

        auto mask = curNode->getActionMask();

        printState(state);
        printMask(mask);

        SPRL::SymmetryIdx sym = SPRL::GetRandom().UniformInt(0, symmetrizer.numSymmetries() - 1);
        std::cout << "Symmetry: " << static_cast<int>(sym) << '\n';

        printState(symmetrizer.symmetrizeState(state, { sym })[0]);
        printMask(symmetrizer.symmetrizeActionDist(mask, { sym })[0]);

        SPRL::ActionIdx action;

        int playerIdx = static_cast<int>(curNode->getPlayer());

        action = agents[playerIdx]->act(curNode, true);

        agents[1 - playerIdx]->opponentAct(action);

        curNode = curNode->getAddChild(action);
    }

    return 0;
}