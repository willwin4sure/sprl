#ifndef SPRL_GO_NODE_HPP
#define SPRL_GO_NODE_HPP

#include "GameNode.hpp"
#include "GridState.hpp"

#include "../random/Zobrist.hpp"

#include <cassert>
#include <vector>

namespace SPRL {

constexpr int GO_BOARD_WIDTH = 9; 
constexpr int GO_BOARD_SIZE = GO_BOARD_WIDTH * GO_BOARD_WIDTH;
constexpr int GO_ACTION_SIZE = GO_BOARD_SIZE + 1; // + 1 for pass
constexpr int GO_HISTORY_LENGTH = 8;

/**
 * GoState needs to also include a history of 8 numbers
 * (which will be Zobrist hashes of the the current state and 7 preceding states for PSK)
 * and a coordinate for fast ko detection
 * 
 * GoState also contains some logic for DSU queries.
*/
class GoNode : public GameNode<GridState<GO_BOARD_SIZE>, GO_ACTION_SIZE> {
public:
    using Board = GridBoard<GO_BOARD_SIZE>;
    using State = GridState<GO_BOARD_SIZE>;
    using GNode = GameNode<State, GO_ACTION_SIZE>;

    using Coord = int32_t;
    using LibertyCount = int32_t;

    /// Special value meaning there is no one-move ko to potentially break
    static constexpr Coord NO_KO_COORD = -1;

    /// Special value meaning >1 stone has been captured
    static constexpr Coord MORE_THAN_ONE_CAPTURE_KO_COORD = -2;

    GoNode() {
        setStartNode();
    }

    GoNode(GoNode* parent, ActionIdx action, const ActionDist& actionMask,
           Player player, Player winner, bool isTerminal, const Board& board,
           Coord koCoord, const std::array<ZobristHash, GO_HISTORY_LENGTH>& zobristHistory,
           int8_t passes)
        : GNode ( parent, action, actionMask, player, winner, isTerminal ),
          m_board { board }, m_koCoord { koCoord }, m_zobristHistory { zobristHistory }, m_passes { passes } {
    }

    void setStartNode() override;
    std::unique_ptr<GNode> getNextNode(ActionIdx action) override;
    
    State getGameState() const override;
    std::array<Value, 2> getRewards() const override;

    std::string toString() const override;

private:
    /**
     * Checks is placing a piece in this coordinate would exhaust all liberties of the region it is part of.
    */
    bool checkSuicide(const Coord coord, const Player player) const;

    /**
     * Slow O(component size) modification which resets all nodes
     * in the connected component of coord. m_board[coord] must be player.
     * 
     * Edits the underlying board and koCoord. In addition, updates the liberties of all adjacent groups of the opposite player.
     * 
     * @returns The Zobrist modification you need to xor the current hash
     * with in order to get the hash of the new board.
    */
    void clearComponent(Coord coord, Player player);

    /**
     * Places a piece in the given coordinate.
     * 
     * Edits the underlying board and koCoord.
     * 
     * @returns The Zobrist modification you need to xor the current hash
     * with in order to get the hash of the new board.
    */
    ZobristHash placePiece(const Coord coord, const Player player);
    
    /**
    * Simulates a pass.
    */
    void pass();
    
    /**
     * Returns the territory score for player 0 and 1, which are integers in [0, GO_BOARD_SIZE].
     * 
     * The algorithm is a WYSIWYG implementation of Go scoring. All stones of a particular color belong to that player.
     * A blank cell belongs to a player if all its neighbors belong to that player, computed using a BFS.
    */
    std::array<int32_t, 2> computeScore();
    
    Coord parent(const Coord coord) const {
        if (m_dsu[coord] == coord) return coord;
        return m_dsu[coord] = parent(m_dsu[coord]);
    }

    LibertyCount getLiberties(Coord coord) const {
        return m_liberties[parent(coord)];
    }

    LibertyCount& getLiberties(Coord coord) {
        return m_liberties[parent(coord)];
    }

    std::vector<Coord> neighbors(const Coord coord) const{
        std::vector<Coord> result;

        if (coord % GO_BOARD_WIDTH > 0) {
            result.push_back(coord - 1);
        }

        if (coord % GO_BOARD_WIDTH < GO_BOARD_WIDTH - 1) {
            result.push_back(coord + 1);
        }

        if (coord >= GO_BOARD_WIDTH) {
            result.push_back(coord - GO_BOARD_WIDTH);
        }
        
        if (coord < GO_BOARD_SIZE - GO_BOARD_WIDTH) {
            result.push_back(coord + GO_BOARD_WIDTH);
        }
    }

    ZobristHash getPieceHash(const Coord coordinate, const Player who) {
        return s_zobrist[coordinate + static_cast<int>(who) * GO_BOARD_SIZE];
    }

    ActionDist actionMask(const GoNode& state) const;

private:
    static const Zobrist<GO_BOARD_SIZE * 2> s_zobrist;

    Board m_board;

    Coord m_koCoord;  // The single position that would violate one-move ko.
    std::array<ZobristHash, GO_HISTORY_LENGTH> m_zobristHistory;  // Zobrist hashes of recent boards for superko.

    int8_t m_passes { 0 };  // Number of passes; 0, 1, or 2. If 2, the game is over.

    mutable std::array<Coord, GO_BOARD_SIZE> m_dsu;  // TODO: make this a separate DSU class
    std::array<LibertyCount, GO_BOARD_SIZE> m_liberties;

    // Zobrist values for each connected component.
    std::array<ZobristHash, GO_BOARD_SIZE> m_componentZobristValues; 
};

} // namespace SPRL

#endif