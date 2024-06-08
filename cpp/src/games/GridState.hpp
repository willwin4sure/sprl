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
 * @returns The other piece.
*/
constexpr Piece otherPiece(Piece piece) {
    switch (piece) {
    case Piece::ZERO: return Piece::ONE;
    case Piece::ONE:  return Piece::ZERO;
    default:          return Piece::NONE;
    }
}

/**
 * @returns The piece corresponding to the player.
*/
constexpr Piece pieceFromPlayer(Player player) {
    switch (player) {
    case Player::ZERO: return Piece::ZERO;
    case Player::ONE:  return Piece::ONE;
    default:          return Piece::NONE;
    }
}

/**
 * @returns The player corresponding to the piece.
*/
constexpr Player playerFromPiece(Piece piece) {
    switch (piece) {
    case Piece::ZERO: return Player::ZERO;
    case Piece::ONE:  return Player::ONE;
    default:          return Player::NONE;
    }
}

/**
 * Represents a grid game board of pieces.
 * 
 * @tparam BOARD_SIZE The size of the board.
*/
template <int BOARD_SIZE>
using GridBoard = std::array<Piece, BOARD_SIZE>;

/**
 * Immutable state of a grid game as a short history of board states.
 * 
 * Used as input into the neural network.
 * 
 * @tparam BOARD_SIZE The size of the board.
*/
template <int BOARD_SIZE>
class GridState {
public:
    /**
     * Constructs a new grid state with the given history.
    */
    GridState(std::vector<GridBoard<BOARD_SIZE>>&& history, Player player)
        : m_history { std::move(history) }, m_player { player } {
    }

    /**
     * @returns A readonly reference to the history of board states.
    */
    const std::vector<GridBoard<BOARD_SIZE>>& getHistory() const {
        return m_history;
    }

    /**
     * @returns The player to move.
    */
    Player getPlayer() const {
        return m_player;
    }

private:
    /// `history[0]` is the current state and higher indices move back in time.
    std::vector<GridBoard<BOARD_SIZE>> m_history;

    /// The current player to move.
    Player m_player { Player::NONE };
};

} // namespace SPRL

#endif