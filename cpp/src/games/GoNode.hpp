#ifndef SPRL_GO_NODE_HPP
#define SPRL_GO_NODE_HPP

#include "GameNode.hpp"
#include "GridState.hpp"

#include "../utils/DSU.hpp"
#include "../utils/Zobrist.hpp"

#include <cassert>
#include <unordered_set>
#include <vector>

namespace SPRL {

constexpr int GO_BOARD_WIDTH = 9; 
constexpr int GO_BOARD_SIZE = GO_BOARD_WIDTH * GO_BOARD_WIDTH;
constexpr int GO_ACTION_SIZE = GO_BOARD_SIZE + 1;  // Last index represents pass.
constexpr int GO_HISTORY_LENGTH = 8;
constexpr float GO_KOMI = 7.5;

class GoNode : public GameNode<GridState<GO_BOARD_SIZE>, GO_ACTION_SIZE> {
public:
    using Board = GridBoard<GO_BOARD_SIZE>;
    using State = GridState<GO_BOARD_SIZE>;
    using GNode = GameNode<State, GO_ACTION_SIZE>;

    // Following data types need to be increased in size if the board size is increased.
    
    using Coord = int8_t;
    using LibertyCount = int8_t;

    GoNode() {
        setStartNode();
    }

    GoNode(GoNode* parent, ActionIdx action, ActionDist&& actionMask,
           Player player, Player winner, bool isTerminal,
           Board&& board, ZobristHash hash, int depth,
           std::unordered_set<ZobristHash>&& zobristHistorySet,
           DSU<Coord, GO_BOARD_SIZE>&& dsu,
           std::array<LibertyCount, GO_BOARD_SIZE>&& liberties,
           std::array<ZobristHash, GO_BOARD_SIZE>&& componentZobristValues)

        : GNode { parent, action, std::move(actionMask), player, winner, isTerminal },
          m_board { std::move(board) }, m_hash { hash }, m_depth { depth },
          m_zobristHistorySet { std::move(zobristHistorySet) },
          m_dsu { std::move(dsu) }, m_liberties { std::move(liberties) },
          m_componentZobristValues { std::move(componentZobristValues) } {

    }

    void setStartNode() override;
    std::unique_ptr<GNode> getNextNode(ActionIdx action) override;
    
    State getGameState() const override;
    std::array<Value, 2> getRewards() const override;

    std::string toString() const override;

private:
    /**
     * @param row The row from the top, must be in the range `[0, GO_BOARD_WIDTH)`.
     * @param col The column from the left, must be in the range `[0, GO_BOARD_WIDTH)`.
     * 
     * @returns The coordinate index of the given row and column.
    */
    static Coord toCoord(int row, int col) {
        assert(row >= 0 && row < GO_BOARD_WIDTH);
        assert(col >= 0 && col < GO_BOARD_WIDTH);
        return row * GO_BOARD_WIDTH + col;
    }

    /**
     * @param coord The coordinate index, must be in the range `[0, GO_BOARD_SIZE)`.
     * 
     * @returns The row and column of the given coordinate.
    */
    static std::pair<int, int> toRowCol(Coord coord) {
        assert(coord >= 0 && coord < GO_BOARD_SIZE);
        return { coord / GO_BOARD_WIDTH, coord % GO_BOARD_WIDTH };
    }

    /**
     * @returns All the in-bounds neighbors of a coordinate.
    */
    std::vector<Coord> neighbors(Coord coord) const {
        std::vector<Coord> result;
        result.reserve(4);

        auto [row, col] = toRowCol(coord);

        if (row > 0) result.push_back(toCoord(row - 1, col));
        if (col > 0) result.push_back(toCoord(row, col - 1));
        
        if (row < GO_BOARD_WIDTH - 1) result.push_back(toCoord(row + 1, col));
        if (col < GO_BOARD_WIDTH - 1) result.push_back(toCoord(row, col + 1));

        return result;
    }

    /**
     * @returns The Zobrist hash for a piece at a particular coordinate.
    */
    ZobristHash getPieceHash(Coord coord, Player player) const {
        return s_zobrist[coord + static_cast<int>(player) * GO_BOARD_SIZE];
    }

    /**
     * @returns The liberty count of the group of a coordinate.
    */
    LibertyCount getLiberties(Coord coord) const {
        return m_liberties[m_dsu.find(coord)];
    }

    /**
     * @returns A reference to the liberty count of the group of a coordinate.
    */
    LibertyCount& liberties(Coord coord) {
        return m_liberties[m_dsu.find(coord)];
    }

    /**
     * @returns The Zobrist hash of the group of a coordinate.
    */
    ZobristHash getComponentZobristValue(Coord coord) const {
        return m_componentZobristValues[m_dsu.find(coord)];
    }

    /**
     * @returns A reference to the Zobrist hash of the group of a coordinate.
    */
    ZobristHash& componentZobristValue(Coord coord) {
        return m_componentZobristValues[m_dsu.find(coord)];
    }

    /**
     * @returns The number of liberties of the group of a coordinate,
     * given the current board state.
    */
    LibertyCount computeLiberties(Coord coord) const;

    /**
     * Observer helper function that detects illegal suicides and violations of PSK.
     * 
     * @returns False if the placement of a piece at `coord` by `player`
     * would immediately result in that piece being captured,
     * or if the move violates the PSK rule.
    */
    bool checkLegalPlacement(Coord coord, Player player) const;

    /**
     * Mutator helper function that removes the group of a particular coodinate.
     * `m_board[coord]` must be a piece owned by `player`.
     * 
     * Edits the board, hash, DSU, and liberty/Zobrist values.
    */
    void clearComponent(Coord coord, Player player);

    /**
     * Places a piece in the given coordinate.
     * 
     * Edits the underlying state.
    */
    void placePiece(Coord coord, Player player);

    /**
     * Observer helper function that computes Tromp-Taylor scoring:
     * all stones count as points to respective players, and empty
     * cells count as points for a color if and only if
     * there is no path of empty cells to a stone of the opposite color.
     * 
     * @returns The territory scores for the two players, which
     * should be integers in the range `[0, GO_BOARD_SIZE]`.
    */
    std::array<int, 2> countTerritory() const;

    ActionDist computeActionMask() const;

private:
    /// Static Zobrist hashes for (Coord, Piece) pairs.
    static inline const Zobrist<GO_BOARD_SIZE * 2> s_zobrist {};

    int m_depth;  // The depth of the node in the tree, starting at 0.
    
    Board m_board;       // The current board state.
    ZobristHash m_hash;  // The hash of the current board.

    /// Set of Zobrist hashes along path to root, inclusive. 
    /// Non-empty sets will only exist when m_depth % L == 0,
    /// for O(L) query time complexity. For now, L = 1.
    std::unordered_set<ZobristHash> m_zobristHistorySet;

    DSU<Coord, GO_BOARD_SIZE> m_dsu;  // DSU holding connected groups of stones.

    /// Liberty count for each group, indexed by representatives.
    std::array<LibertyCount, GO_BOARD_SIZE> m_liberties;

    /// Total Zobrist hash for each group, indexed by representatives.
    std::array<ZobristHash, GO_BOARD_SIZE> m_componentZobristValues; 
};

} // namespace SPRL

#endif