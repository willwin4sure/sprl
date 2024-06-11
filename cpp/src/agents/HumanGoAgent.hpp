#ifndef SPRL_HUMAN_GO_AGENT_HPP
#define SPRL_HUMAN_GO_AGENT_HPP

#include "../games/GoNode.hpp"
#include "Agent.hpp"

#include <iostream>

namespace SPRL {

/**
 * Agent that prompts the terminal for playing Go, so a human can play.
 * 
 * Asks for a square on a Go board (e.g. `C5`), or `XX` for pass.
*/
class HumanGoAgent : public Agent<GoNode, GridState<GO_BOARD_SIZE, GO_HISTORY_SIZE>, GO_ACTION_SIZE> {
public:
    using State = GridState<GO_BOARD_SIZE, GO_HISTORY_SIZE>;
    using ActionDist = GameActionDist<GO_ACTION_SIZE>;

    ActionIdx act(const GameNode<GoNode, GridState<GO_BOARD_SIZE, GO_HISTORY_SIZE>, GO_ACTION_SIZE>* gameNode,
                  bool verbose = false) const override {

        while (true) {
            ActionIdx action {};

            std::cout << "Enter a square, or XX for pass (e.g. C5): ";
            char file, rank;
            std::cin >> file >> rank;

            if (!std::cin) {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << "Invalid format. Please retry." << '\n';
                continue;
            }

            if (file == 'X' && rank == 'X') {
                action = GO_BOARD_SIZE;
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                return action;
            }

            action = (file - 'A') + (rank - '0') * GO_BOARD_WIDTH;

            if (action < 0 || action >= GO_ACTION_SIZE) {
                std::cout << "Action not in bounds. Try again." << '\n';
                continue;
            }

            if (gameNode->getActionMask()[action] == 0.0f) {
                std::cout << "Action is not legal in this position. Try again." << '\n';
                continue;
            }

            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

            return action;
        }
    }
};

} // namespace SPRL

#endif
