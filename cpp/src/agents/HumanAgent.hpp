#ifndef SPRL_HUMAN_AGENT_HPP
#define SPRL_HUMAN_AGENT_HPP

#include "Agent.hpp"

#include <iostream>

namespace SPRL {

/**
 * Agent that prompts the terminal for input.
*/
template <typename ImplNode, typename State, int AS>
class HumanAgent : public Agent<ImplNode, State, AS> {
public:
    using ActionDist = GameActionDist<AS>;

    ActionIdx act(const ImplNode* gameNode, bool verbose = false) const override {

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

            if (action < 0 || action >= AS) {
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
