#include "HumanOthelloAgent.hpp"

#include <iostream>
#include <limits>

namespace SPRL {

ActionIdx HumanOthelloAgent::act(Game<OTH_SIZE * OTH_SIZE, OTH_SIZE * OTH_SIZE + 1>* game,
                                 const State& state,
                                 const ActionDist& actionMask,
                                 bool verbose) const {

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
            action = OTH_SIZE * OTH_SIZE;
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            return action;
        }

        action = (file - 'A') + (rank - '0') * OTH_SIZE;

        if (action < 0 || action >= OTH_SIZE * OTH_SIZE + 1) {
            std::cout << "Action not in bounds. Try again." << '\n';
            continue;
        }

        if (actionMask[action] == 0.0) {
            std::cout << "Action is not legal in this position. Try again." << '\n';
            continue;
        }

        return action;
    }

}

} // namespace SPRL
