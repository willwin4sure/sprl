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
    virtual std::pair<std::array<float, ACTION_SIZE>, float> evaluate(SPRL::Game<BOARD_SIZE, ACTION_SIZE>* game, const GameState<BOARD_SIZE>& state) = 0;
};

} // namespace SPRL

#endif
