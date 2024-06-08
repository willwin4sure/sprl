#ifndef SPRL_AGENT_HPP
#define SPRL_AGENT_HPP

#include "../games/GameNode.hpp"

namespace SPRL {

/**
 * Interface class for agents that play a game.
 * 
 * @tparam ImplNode The implementation of the game node, e.g. GoNode.
 * @tparam State The state of the game, e.g. GoState.
 * @tparam ACTION_SIZE The number of possible actions in the game.
*/
template <typename ImplNode, typename State, int ACTION_SIZE>
class Agent {
public:
    using ActionDist = GameActionDist<ACTION_SIZE>;

    virtual ~Agent() = default;

    /**
     * @param gameNode The current game node, must be non-terminal.
     * @param verbose Whether to print debug information.
     * 
     * @returns The action to take given the current game node.
    */
    virtual ActionIdx act(const GameNode<ImplNode, State, ACTION_SIZE>* gameNode, bool verbose = false) const = 0;

    /**
     * Processes an opponent's action if necessary to update state,
     * e.g. advancing the decision node in a game tree.
     * 
     * @param action The opponent's action.
    */
    virtual void opponentAct(const ActionIdx action) const {}
};

} // namespace SPRL

#endif
