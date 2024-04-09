#ifndef GAME_HPP
#define GAME_HPP

#include "GameState.hpp"

#include <string>

namespace SPRL {

/// Type alias for the action index.
using ActionIdx = int8_t;

/// Type alias for the symmetry index.
using Symmetry = int8_t;

/**
 * Interface class for two-player, zero-sum, abstract strategy games.
*/
template <int BOARD_SIZE, int ACTION_SIZE>
class Game {
public:
    /// Type alias for the game state.
    using State = GameState<BOARD_SIZE>;
    
    /// Type alias for the action space.
    using ActionSpace = std::array<float, ACTION_SIZE>;
    
    virtual ~Game() = default;

    /**
     * Returns the initial state of the game.
    */
    virtual State startState() const = 0;

    /**
     * Returns the next state of the game given the current state and an action.
    */
    virtual State nextState(const State& state, const ActionIdx action) const = 0;

    /**
     * Returns True if the game is over, False otherwise.
    */
    virtual bool isTerminal(const State& state) const = 0;

    /**
     * Returns a mask of valid actions for the current state.
    */
    virtual ActionSpace actionMask(const State& state) const = 0;

    /**
     * Returns the rewards for the two players given the current state.
    */
    virtual std::pair<float, float> rewards(const State& state) const = 0;

    /**
     * Returns the number of symmetries for the game.
    */
    virtual int numSymmetries() const = 0;

    /**
     * Returns the inverse of a given symmetry.
    */
    virtual Symmetry inverseSymmetry(const Symmetry& symmetry) const = 0;

    /**
     * Returns a vector of symmetrized states given a state and a vector of symmetries.
    */
    virtual std::vector<State> symmetrizeState(const State& state, const std::vector<Symmetry>& symmetries) const = 0;

    /**
     * Returns a vector of symmetrized action spaces given an action space and a vector of symmetries.
    */
    virtual std::vector<ActionSpace> symmetrizeActionSpace(const ActionSpace& actionSpace, const std::vector<Symmetry>& symmetries) const = 0;

    /**
     * Returns a string representation of the game state.
    */
    virtual std::string stateToString(const State& state) const = 0;
};

} // namespace SPRL

#endif