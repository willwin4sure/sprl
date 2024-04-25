#ifndef GAME_STATE_HPP
#define GAME_STATE_HPP

#include <cstdint>
#include <array>
#include <vector>

namespace SPRL {

/// Type alias for the player (0 or 1), or -1 to represent no player.
using Player = int8_t;

/// Type alias for the piece (0 or 1), or -1 to represent no piece.
using Piece = int8_t;

/// Templated type alias for some board.
template <int BOARD_SIZE>
using GameBoard = std::array<Piece, BOARD_SIZE>;


/**
 * Immutable class for game states, templated on the size of the (flattened) board.
 * 
 * Represents the current state of a game in a light-weight fashion.
*/
template <int BOARD_SIZE>
class GameState {
public:
    using Board = GameBoard<BOARD_SIZE>;

    /**
     * Constructs a new game state with an empty board, player 0 to move, no winner.
    */
    GameState() : m_board {}, m_player { 0 }, m_winner { -1 }, m_isTerminal { false } {
        m_board.fill(-1);  // empty board
    }

    /**
     * Constructs a new game state with the given board, player to move, and winner.
    */
    GameState(const Board& board, Player player, Player winner, bool isTerminal)
        : m_board { board }, m_player { player }, m_winner { winner }, m_isTerminal { isTerminal } {}

    /**
     * Returns a readonly reference to the underlying board.
    */
    const Board& getBoard() const {
        return m_board;
    }

    /**
     * Returns the player to move.
    */
    Player getPlayer() const {
        return m_player;
    }

    /**
     * Returns the winner of the game.
    */
    Player getWinner() const {
        return m_winner;
    }

    /**
     * Returns whether the state is terminal.
    */
    bool isTerminal() const {
        return m_isTerminal;
    }

private:
    Board m_board {};
    Player m_player {};
    Player m_winner {};
    bool m_isTerminal {};
};

} // namespace SPRL

#endif