#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "../games/GameState.hpp"
#include "../games/Game.hpp"

namespace SPRL {

/**
 * Interface for neural networks that evaluate game states.
*/
template <int BOARD_SIZE, int ACTION_SIZE>
class Network {
public:
    virtual ~Network() = default;

    /**
     * Returns a pair of the action distribution and the value estimate for the given state.
    */
    virtual std::vector<std::pair<GameActionDist<ACTION_SIZE>, Value>> evaluate(const std::vector<GameState<BOARD_SIZE>>& states,
                                                                                const std::vector<GameActionDist<ACTION_SIZE>>& masks) = 0;

    /**
     * Returns the number of evaluations made by the network, summed over batches.
    */
    virtual int getNumEvals() = 0;
};

} // namespace SPRL

#endif
