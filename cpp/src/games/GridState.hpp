#ifndef SPRL_GRID_STATE_HPP
#define SPRL_GRID_STATE_HPP

/**
 * @file GridState.hpp
 * 
 * Contains a particular game state representation for grid games
 * with exactly two types of pieces.
*/

#include "GameNode.hpp"

#include <vector>

namespace SPRL {

/**
 * Represents a piece on the grid game board.
*/
enum class Piece : int8_t {
    NONE = -1,
    ZERO = 0,
    ONE  = 1
};

/**
 * Represents a grid game board of pieces.
 * 
 * @tparam BS The size of the board.
*/
template <int BS>
using GridBoard = std::array<Piece, BS>;

/**
 * Immutable state of a grid game as a short history of board states.
 * 
 * Used as input into the neural network.
 * 
 * @tparam BS The size of the board.
*/
template <int BS>
class GridState {
public:
    /**
     * Constructs a new grid state with the given history.
    */
    GridState(std::vector<GridBoard<BS>>&& history, Player player)
        : m_history { std::move(history) }, m_player { player } {
    }

private:
    /// `history[0]` is the current state and higher indices move back in time.
    std::vector<GridBoard<BS>> m_history;

    /// The current player to move.
    Player m_player { Player::NONE };
};

} // namespace SPRL

#endif