#ifndef SPRL_INETWORK_HPP
#define SPRL_INETWORK_HPP

#include "../games/GameNode.hpp"

namespace SPRL {

/**
 * Interface for evaluation of game states, e.g. using a neural network
 * or a heuristic.
 * 
 * @tparam State The state of the game.
 * @tparam ACTION_SIZE The number of possible actions in the game.
*/
template <typename State, int ACTION_SIZE>
class INetwork {
public:
    using ActionDist = GameActionDist<ACTION_SIZE>;

    virtual ~INetwork() = default;

    /**
     * @returns A pair of the action distribution and the value estimate for the given state.
    */
    virtual std::vector<std::pair<ActionDist, Value>> evaluate(
        const std::vector<State>& states,
        const std::vector<ActionDist>& masks) = 0;

    /**
     * @returns The number of evaluations made by the network, summed over batches.
    */
    virtual int getNumEvals() = 0;
};

} // namespace SPRL

#endif
