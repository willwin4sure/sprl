#include "GoNode.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <queue>

namespace SPRL {

GoNode::LibertyCount GoNode::computeLiberties(Coord coord) const {
    Player player = playerFromPiece(m_board[coord]);
    
    if (player == Player::NONE) {
        return 0;
    }

    // Run a BFS.
    
    std::array<bool, GO_BOARD_SIZE> visited;
    visited.fill(false);

    std::deque<Coord> q;
    visited[coord] = true;
    q.push_back(coord);

    LibertyCount liberties = 0;
    
    while (!q.empty()) {
        Coord current = q.front();
        q.pop_front();
        
        assert(visited[current]);
        assert(m_board[current] == pieceFromPlayer(player));

        for (Coord neighbor : neighbors(current)) {
            if (m_board[neighbor] == pieceFromPlayer(player)) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    q.push_back(neighbor);
                }
                
            } else if (m_board[neighbor] == Piece::NONE) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;  // Don't double count liberties.
                    ++liberties;
                }
            }
        }
    }

    return liberties;
}

void GoNode::clearComponent(Coord coord, Player player) {
    assert(m_board[coord] == pieceFromPlayer(player));
    
    m_board[coord] = Piece::NONE;

    // Update all the DSU state to denote empty.

    m_dsu.setParent(coord, coord);
    liberties(coord) = 0;
    componentZobristValue(coord) = 0;

    /*
     * Check for stones which belong to the opposite player
     * adjacent to the current stone in the removed component.
     * Each of these components should gain a liberty.
    */

    std::vector<Coord> oppNeighborGroups;
    oppNeighborGroups.reserve(4);

    for (Coord neighbor : neighbors(coord)) {
        if (m_board[neighbor] == Piece::NONE) {
            continue;
        }

        if (m_board[neighbor] == pieceFromPlayer(player)) {
            clearComponent(neighbor, player);
            
        } else {
            Coord group = m_dsu.find(neighbor);

            if (std::find(oppNeighborGroups.begin(), oppNeighborGroups.end(), group) != oppNeighborGroups.end()) {
                continue;  // Already counted.
            }
            oppNeighborGroups.push_back(group);
            
            ++liberties(group);
        }
    }
}

void GoNode::placePiece(Coord coord, Player player){
    assert(m_board[coord] == Piece::NONE);
    m_board[coord] = pieceFromPlayer(player);
    
    /*
     * Phase One: check for friendly neighbors, and merge all of them together.
     * In addition, we will compute the Zobrist hash of the new component.
    */

    ZobristHash newComponentHash = getPieceHash(coord, player);
    
    for (Coord neighbor : neighbors(coord)) {
        if (m_board[neighbor] == pieceFromPlayer(player)) {
            if (m_dsu.sameSet(neighbor, coord)) {
                continue;  // Already counted.
            }
            newComponentHash ^= getComponentZobristValue(neighbor);
            m_dsu.unite(neighbor, coord);
        }
    }

    componentZobristValue(coord) = newComponentHash;

    /*
     * Update the liberties of the new component.
     * It is too difficult to update the liberties in O(1), so instead
     * we will BFS to re-evaluate the liberties of the new component.
     * At this stage, the liberty count for component is correct,
     * assuming that no captured enemy stones have been removed yet.
     *
     * (The reason we don't add in the new liberties from the
     * captured components at this stage is because in the next
     * stage we'll add them, plus possible other liberties
     * to other unrelated friendly components).
    */

    liberties(coord) = computeLiberties(coord);

    /*
     * Phase Two: check for enemy neighbors, and deduct
     * a liberty from those components.
     * If a component has no liberties left, it dies,
     * which also updates friendly liberties.
    */
    
    // Hash update for entire state, begins with the new piece we placed.
    ZobristHash stateHashUpdate = getPieceHash(coord, player);

    std::vector<Coord> oppNeighborGroups;
    oppNeighborGroups.reserve(4);

    for (Coord neighbor : neighbors(coord)) {
        if (m_board[neighbor] == pieceFromPlayer(otherPlayer(player))) {
            Coord group = m_dsu.find(neighbor);

            if (std::find(oppNeighborGroups.begin(), oppNeighborGroups.end(), group) != oppNeighborGroups.end()) {
                continue;  // Already counted.
            }
            oppNeighborGroups.push_back(group);

            // Remove a liberty from this gruop.            
            --liberties(group);

            // Kill the component if necessary.
            if (getLiberties(group) == 0) {
                stateHashUpdate ^= getComponentZobristValue(group);
                clearComponent(group, otherPlayer(player));
            }
        }
    }

    // Update the Zobrist value.
    m_hash ^= stateHashUpdate;

    // For positional super-ko detection.
    assert(m_zobristHistorySet.find(m_hash) == m_zobristHistorySet.end());
    m_zobristHistorySet.insert(m_hash);
}

bool GoNode::checkLegalPlacement(const Coord coordinate, const Player player) const {
    assert(player == m_player);  // Correct player.

    if (m_board[coordinate] != Piece::NONE) {
        // Position is already occupied.
        return false;
    }

    ZobristHash newHash = m_hash ^ getPieceHash(coordinate, player);
    bool hasLiberties = false;

    std::vector<Coord> oppNeighborGroups;
    oppNeighborGroups.reserve(4);

    for (Coord neighbor : neighbors(coordinate)) {
        if (m_board[neighbor] == Piece::NONE) {
            // Empty neighbor
            hasLiberties = true;

        } else if (m_board[neighbor] == pieceFromPlayer(player)) {
            // Friendly neighbor
            if (getLiberties(neighbor) > 1) {

                // We have a liberty because we're attached to a
                // friendly piece with more than one liberty
                // (and we've only consumed one of them, leaving at least one)

                hasLiberties = true;
            }

        } else {
            // Enemy neighbor
            if (getLiberties(neighbor) == 1) {
                // We would capture the enemy piece, so we must have a liberty.
                hasLiberties = true;
                Coord group = m_dsu.find(neighbor);
                if (std::find(oppNeighborGroups.begin(), oppNeighborGroups.end(), group) == oppNeighborGroups.end()) {
                    oppNeighborGroups.push_back(group);

                    // Hash update
                    newHash ^= getComponentZobristValue(group);
                }
            }
        }
    }

    // If we have liberties and the new state hash is not in the history (we haven't seen this state before)
    return hasLiberties && m_zobristHistorySet.find(newHash) == m_zobristHistorySet.end();
}


std::array<int, 2> GoNode::computeScore() {
    std::array<bool, GO_BOARD_SIZE> visited;
    visited.fill(false);

    // In this algorithm, visited[i] == 1 implies m_board[i] == -1.

    std::array<int, 2> points = {0, 0};
    for (int i = 0; i < GO_BOARD_SIZE; i++) {
        if (m_board[i] == Piece::ZERO) {
            points[0]++;
            continue;
        }
        if (m_board[i] == Piece::ONE) {
            points[1]++;
            continue;
        }
        
        if (visited[i] != 0)
            continue;
        
        std::queue<int> q;
        q.push(i);
        visited[i] = 1;
        int count = 0;
        std::array<bool, 2> possible_territory = {true, true}; // {black, white}.
        while (!q.empty()) {
            int current = q.front();
            q.pop();
            count++;
            // Some assert statements just to verify the correctness of the algorithm
            assert (m_board[current] == Piece::NONE);
            assert (visited[current] == 1);
            for (int neighbor : neighbors(current)) {
                if (m_board[neighbor] == Piece::ZERO) {
                    // touches a black stone, hence it cannot be white territory
                    possible_territory[1] = false;
                }
                else if (m_board[neighbor] == Piece::ONE) {
                    // touches a white stone, hence it cannot be black territory
                    possible_territory[0] = false;
                } else {
                    // vacant, continue BFS
                    if (visited[neighbor] != 0) {
                        continue;
                    }
                    visited[neighbor] = 1;
                    q.push(neighbor);
                }
            }
        }
        if (possible_territory[0]) {
            if (possible_territory[1]) {
                // The only situation in which this occurs is if the board is completely empty.
            }else{
                points[0] += count;
            }
        } else {
            if (possible_territory[1]) {
                points[1] += count;
            } else {
                // Neither black nor white territory
            }
        }
    }
    return points;
}

GoNode::ActionDist GoNode::actionMask(const GoNode& state) const {
    GoNode::ActionDist mask;
    for (Coord i = 0; i < GO_BOARD_SIZE; i++) {
        mask[i] = checkLegalPlacement(i, state.m_player);
    }
    mask[GO_BOARD_SIZE] = true; // pass is always a valid action
}


void GoNode::setStartNode() {
    // TODO @will fix this i dont really know how initialization works
    m_parent = nullptr;
    m_action = 0;
}


std::unique_ptr<GoNode::GNode> GoNode::getNextNode(ActionIdx actionIdx) {

    // Copy the state.

    ActionDist newActionMask = m_actionMask;
    Board newBoard = m_board;
    std::unordered_set<ZobristHash> newZobristHistorySet = m_zobristHistorySet;
    DSU<Coord, GO_BOARD_SIZE> newDSU = m_dsu;
    std::array<LibertyCount, GO_BOARD_SIZE> newLiberties = m_liberties;
    std::array<ZobristHash, GO_BOARD_SIZE> newComponentZobristValues = m_componentZobristValues;

    GoNode copyNode {
        static_cast<GoNode*>(m_parent),
        m_action,
        std::move(newActionMask),
        m_player,
        m_winner,
        m_isTerminal,
        std::move(newBoard),
        m_hash,
        m_depth,
        std::move(newZobristHistorySet),
        std::move(newDSU),
        std::move(newLiberties),
        std::move(newComponentZobristValues)
    };

    if (actionIdx != GO_BOARD_SIZE) {
        // Handle a piece placement.
        assert(actionIdx >= 0 && actionIdx < GO_BOARD_SIZE);
        assert(checkLegalPlacement(actionIdx, m_player));

        copyNode.placePiece(actionIdx, m_player);
    }

    // @rowechen

    return std::make_unique<GoNode>(std::move(copyNode));
}

} // namespace SPRL