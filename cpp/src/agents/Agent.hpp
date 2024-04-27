#ifndef AGENT_HPP
#define AGENT_HPP

#include "../games/GameState.hpp"
#include "../games/Game.hpp"

namespace SPRL {

/**
 * Interface class for agents that play a game.
*/
template <int BOARD_SIZE, int ACTION_SIZE>
class Agent {
public:
    using State = GameState<BOARD_SIZE>;
    using ActionDist = GameActionDist<ACTION_SIZE>;

    virtual ~Agent() = default;

    /**
     * Returns the action to take given the current state.
     * 
     * Requires that the state is non-terminal.
    */
    virtual ActionIdx act(Game<BOARD_SIZE, ACTION_SIZE>* game,
                          const State& state,
                          const ActionDist& actionMask,
                          bool verbose = false) const = 0;

    /**
     * Processes an opponent's action if necessary to update state.
    */
    virtual void opponentAct(const ActionIdx action) {}
};

} // namespace SPRL

#endif
