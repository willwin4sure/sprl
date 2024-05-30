#ifndef SPRL_NETWORK_HPP
#define SPRL_NETWORK_HPP

#include "../games/GameNode.hpp"

namespace SPRL {

/**
 * Interface for neural networks that evaluate game states.
*/
template <typename State, int AS>
class Network {
public:
    using ActionDist = GameActionDist<AS>;

    virtual ~Network() = default;

    /**
     * Returns a pair of the action distribution and the value estimate for the given state.
    */
    virtual std::vector<std::pair<ActionDist, Value>> evaluate(
        const std::vector<State>& states,
        const std::vector<ActionDist>& masks) = 0;

    /**
     * Returns the number of evaluations made by the network, summed over batches.
    */
    virtual int getNumEvals() = 0;
};

} // namespace SPRL

#endif
