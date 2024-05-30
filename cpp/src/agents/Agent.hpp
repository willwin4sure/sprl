#ifndef SPRL_AGENT_HPP
#define SPRL_AGENT_HPP

#include "../games/GameNode.hpp"

namespace SPRL {

/**
 * Interface class for agents that play a game.
*/
template <typename State, int AS>
class Agent {
public:
    using GNode = GameNode<State, AS>;
    using ActionDist = GameActionDist<AS>;

    virtual ~Agent() = default;

    /**
     * Returns the action to take given the current game node.
     * 
     * Requires that the state is non-terminal.
    */
    virtual ActionIdx act(const GNode* gameNode, bool verbose = false) const = 0;

    /**
     * Processes an opponent's action if necessary to update state.
    */
    virtual void opponentAct(const ActionIdx action) const {}
};

} // namespace SPRL

#endif
