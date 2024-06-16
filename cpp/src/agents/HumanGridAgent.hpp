#ifndef SPRL_HUMAN_GRID_AGENT_HPP
#define SPRL_HUMAN_GRID_AGENT_HPP

#include "IAgent.hpp"

#include "../games/GridState.hpp"

#include <iostream>

namespace SPRL {

/**
 * Agent that prompts the terminal for playing a grid game, so a human can play.
 * The game must also have exactly one pass action.
 * 
 * Asks for a square on the board (e.g. `C5`), or `XX` for pass.
 * 
 * @tparam ImplNode The implementation of the game node, e.g. `GoNode`.
 * @tparam NUM_ROWS The number of rows in the grid.
 * @tparam NUM_COLS The number of columns in the grid.
 * @tparam HISTORY_SIZE The length of the history in a state.
*/
template <typename ImplNode, int NUM_ROWS, int NUM_COLS, int HISTORY_SIZE>
class HumanGridAgent : public IAgent<ImplNode, GridState<NUM_ROWS * NUM_COLS, HISTORY_SIZE>, NUM_ROWS * NUM_COLS + 1> {
public:
    using State = GridState<NUM_ROWS * NUM_COLS, HISTORY_SIZE>;
    using ActionDist = GameActionDist<NUM_ROWS * NUM_COLS + 1>;

    ActionIdx act(const GameNode<ImplNode, GridState<NUM_ROWS * NUM_COLS, HISTORY_SIZE>, NUM_ROWS * NUM_COLS + 1>* gameNode,
                  bool verbose = false) const override {

        while (true) {
            ActionIdx action {};

            std::cout << "Enter a square, or XX for pass (e.g. C5): ";
            char file, rank;
            std::cin >> file >> rank;

            if (!std::cin) {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << "Invalid format. Please retry.\n";
                continue;
            }

            if (file == 'X' && rank == 'X') {
                action = NUM_ROWS * NUM_COLS;
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                return action;
            }

            int row = rank - '0';
            int col = file - 'A';

            if (row < 0 || row >= NUM_ROWS || col < 0 || col >= NUM_COLS) {
                std::cout << "Action not in bounds. Try again.\n";
                continue;
            }

            action = row * NUM_COLS + col;

            if (gameNode->getActionMask()[action] == 0.0f) {
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
