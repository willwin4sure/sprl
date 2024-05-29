#ifndef SPRL_GAME_STATE_HPP
#define SPRL_GAME_STATE_HPP

#include "GameActionDist.hpp"

#include <cstdint>
#include <vector>

namespace SPRL {

/**
 * Represents a player in the game.
*/
enum class Player : int8_t {
    NONE = -1,
    ZERO = 0,
    ONE  = 1
};

/**
 * Represents a piece on the game board.
*/
enum class Piece : int8_t {
    NONE = -1,
    ZERO = 0,
    ONE  = 1
};

/**
 * Represents a game board of pieces.
 * 
 * @tparam BS The size of the board.
*/
template <int BS>
using GameBoard = std::array<Piece, BS>;

/**
 * Immutable state of a game as a short history of board states.
 * 
 * Used as input into the neural network.
 * 
 * @tparam BS The size of the board.
*/
template <int BS>
class GameState {
private:
    /// `history[0]` is the current state and higher indices move back in time.
    std::vector<GameBoard<BS>> m_history;

    /// The current player to move.
    Player m_player { Player::NONE };
};

} // namespace SPRL

#endif