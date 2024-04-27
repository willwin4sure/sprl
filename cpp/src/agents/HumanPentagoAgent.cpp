#include "HumanPentagoAgent.hpp"

#include <iostream>

namespace SPRL {

ActionIdx HumanPentagoAgent::act(Game<PTG_BOARD_SIZE, PTG_NUM_ACTIONS>* game,
                                  const State& state,
                                  const ActionDist& actionMask,
                                  bool verbose) const {

    while (true) {
        ActionIdx action {};

        std::cout << "Enter a square, followed by a quadrant, then a rotation (e.g. C5 0 1): ";
        char file, rank;
        int quadrant, rotation;
        std::cin >> file >> rank >> quadrant >> rotation;

        if (!std::cin) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid format. Please retry." << '\n';
            continue;
        }

        int boardIdx = (file - 'A') + (rank - '0') * 6;
        action = boardIdx + quadrant * 36 + rotation * 36 * 4;

        if (action < 0 || action >= PTG_NUM_ACTIONS) {
            std::cout << "Action not in bounds. Try again." << '\n';
            continue;
        }

        if (actionMask[action] == 0.0) {
            std::cout << "Action is not legal in this position. Try again." << '\n';
            continue;
        }

        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        return action;
    }

}

} // namespace SPRL
