#ifndef HUMAN_AGENT_HPP
#define HUMAN_AGENT_HPP

#include "Agent.hpp"

#include <iostream>

namespace SPRL {

/**
 * Agent that prompts the terminal for input.
*/
template <int BOARD_SIZE, int ACTION_SIZE>
class HumanAgent : public Agent<BOARD_SIZE, ACTION_SIZE> {
public:
    using State = GameState<BOARD_SIZE>;
    using ActionDist = GameActionDist<ACTION_SIZE>;

    ActionIdx act(Game<BOARD_SIZE, ACTION_SIZE>* game,
                  const State& state,
                  const ActionDist& actionMask,
                  bool verbose = false) const override {

        while (true) {
            ActionIdx action {};

            std::cout << "Enter an action index: ";
            std::cin >> action;

            if (!std::cin) {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << "Please enter an integer for the action index." << '\n';
                continue;
            }

            if (action < 0 || action >= ACTION_SIZE) {
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
};

} // namespace SPRL

#endif
