#ifndef SPRL_HUMAN_AGENT_HPP
#define SPRL_HUMAN_AGENT_HPP

#include "IAgent.hpp"

#include <iostream>

namespace SPRL {

/**
 * Agent that prompts the terminal for input, so a human can play.
 * 
 * This basic agent just prompts the human for a valid action index.
 * More specialized agents can be created for specific games
 * that accept human-readable input.
 * 
 * @tparam ImplNode The implementation of the game node, e.g. `GoNode`.
 * @tparam State The state of the game, e.g. `GridState`.
 * @tparam ACTION_SIZE The number of possible actions in the game.
*/
template <typename ImplNode, typename State, int ACTION_SIZE>
class HumanAgent : public IAgent<ImplNode, State, ACTION_SIZE> {
public:
    using ActionDist = GameActionDist<ACTION_SIZE>;

    ActionIdx act(const GameNode<ImplNode, State, ACTION_SIZE>* gameNode,
                  bool verbose = false) const override {

        while (true) {
            ActionIdx action {};

            std::cout << "Enter an action index: ";
            std::cin >> action;

            if (!std::cin) {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << "Invalid format. Please retry.\n";
                continue;
            }

            if (action < 0 || action >= ACTION_SIZE) {
                std::cout << "Action not in bounds. Try again.\n";
                continue;
            }

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
