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
    /**
     * Returns a pair of the action distribution and the value estimate for the given state.
    */
    virtual std::vector<std::pair<GameActionDist<ACTION_SIZE>, Value>> evaluate(SPRL::Game<BOARD_SIZE, ACTION_SIZE>* game,
                                                                                const std::vector<GameState<BOARD_SIZE>>& states) = 0;
};

} // namespace SPRL

#endif
