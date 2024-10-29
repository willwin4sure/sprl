#ifndef SPRL_HUMAN_CHESS_AGENT_HPP
#define SPRL_HUMAN_CHESS_AGENT_HPP

#include "IAgent.hpp"

#include "../games/ChessNode.hpp"

#include <iostream>

namespace SPRL {

class HumanChessAgent : public IAgent<ChessNode, ChessState, CHESS_ACTION_SIZE> {
public:
    using State = ChessState;
    using ActionDist = GameActionDist<CHESS_ACTION_SIZE>;

    ActionIdx act(const GameNode<ChessNode, ChessState, CHESS_ACTION_SIZE>* gameNode,
                  bool verbose = false) const override {

        while (true) {
            ActionIdx action {};

            std::cout << "Enter a move (e.g. e2f4, e7e8n): ";

            std::string move;
            std::cin >> move;

            if (!std::cin) {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << "Invalid format. Please retry.\n";
                continue;
            }

            if (!CHESS_MOVE_STR_TO_IDX.contains(move)) {
                std::cout << "Invalid move string.\n";
                continue;
            }

            action = CHESS_MOVE_STR_TO_IDX.at(move);

            if (gameNode->getActionMask()[action] == 0) {
                std::cout << "Action is not legal in this position. Try again.\n";
                continue;
            }

            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

            return action;
        }
    }
};

} // namespace SPRL

#endif