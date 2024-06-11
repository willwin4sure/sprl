#include "GoNode.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <queue>

namespace SPRL {

GoNode::LibertyCount GoNode::computeLiberties(Coord coord) const {
    Piece piece = m_board[coord];
    
    if (piece == Piece::NONE) {
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
        assert(m_board[current] == piece);

        for (Coord neighbor : neighbors(current)) {
            if (m_board[neighbor] == piece) {
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

void GoNode::clearComponent(Coord coord, Piece piece) {
    assert(m_board[coord] == piece);
    
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

        if (m_board[neighbor] == piece) {
            clearComponent(neighbor, piece);
            
        } else {
            Coord group = m_dsu.find(neighbor);

            if (std::find(oppNeighborGroups.begin(), oppNeighborGroups.end(),
                          group) != oppNeighborGroups.end()) {
                continue;  // Already counted.
            }
            oppNeighborGroups.push_back(group);
            
            ++liberties(group);
        }
    }
}

void GoNode::placePiece(Coord coord, Piece piece) {
    assert(m_board[coord] == Piece::NONE);

    m_board[coord] = piece;
    
    /*
     * Phase One: check for friendly neighbors, and merge all of them together.
     * In addition, we will compute the Zobrist hash of the new component.
    */

    // New hash for the component that this piece is joining.
    ZobristHash newComponentHash = getPieceHash(coord, piece);
    
    for (Coord neighbor : neighbors(coord)) {
        if (m_board[neighbor] == piece) {
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
    ZobristHash stateHashUpdate = getPieceHash(coord, piece);

    std::vector<Coord> oppNeighborGroups;
    oppNeighborGroups.reserve(4);

    for (Coord neighbor : neighbors(coord)) {
        if (m_board[neighbor] == otherPiece(piece)) {
            Coord group = m_dsu.find(neighbor);

            if (std::find(oppNeighborGroups.begin(), oppNeighborGroups.end(),
                          group) != oppNeighborGroups.end()) {
                continue;  // Already counted.
            }
            oppNeighborGroups.push_back(group);

            // Remove a liberty from this gruop.            
            --liberties(group);

            // Kill the component if necessary.
            if (getLiberties(group) == 0) {
                stateHashUpdate ^= getComponentZobristValue(group);
                clearComponent(group, otherPiece(piece));
            }
        }
    }

    // Update the Zobrist value.
    m_hash ^= stateHashUpdate;

    // For positional super-ko detection.
    assert(m_zobristHistorySet.find(m_hash) == m_zobristHistorySet.end());
    m_zobristHistorySet.insert(m_hash);
}

bool GoNode::checkLegalPlacement(Coord coordinate, Piece piece) const {
    assert(playerFromPiece(piece) == m_player);  // Correct player.

    if (m_board[coordinate] != Piece::NONE) {
        // Position is already occupied.
        return false;
    }

    ZobristHash newHash = m_hash ^ getPieceHash(coordinate, piece);
    bool hasLiberties = false;

    std::vector<Coord> oppNeighborGroups;
    oppNeighborGroups.reserve(4);

    for (Coord neighbor : neighbors(coordinate)) {
        if (m_board[neighbor] == Piece::NONE) {
            // Empty neighbor, must have liberties.
            hasLiberties = true;

        } else if (m_board[neighbor] == piece) {
            // Friendly neighbor, see if it keeps us alive.
            if (getLiberties(neighbor) > 1) {

                // We have a liberty because we're attached to a
                // friendly piece with more than one liberty (and we've
                // only consumed one of them, leaving at least one).

                hasLiberties = true;
            }

        } else {
            // Enemy neighbor, see if we capture its group.
            if (getLiberties(neighbor) == 1) {
                // We would capture the enemy piece, so we must have a liberty.
                hasLiberties = true;
                Coord group = m_dsu.find(neighbor);
                if (std::find(oppNeighborGroups.begin(), oppNeighborGroups.end(),
                              group) != oppNeighborGroups.end()) {
                    continue;  // Already counted.
                }
                oppNeighborGroups.push_back(group);

                // Hash update that would occur from capturing enemy group.
                newHash ^= getComponentZobristValue(group);
            }
        }
    }

    // If we have liberties and the new state hash is not in the history (PSK)
    return hasLiberties && m_zobristHistorySet.find(newHash) == m_zobristHistorySet.end();
}

std::array<int, 2> GoNode::countTerritory() const {
    std::array<bool, GO_BOARD_SIZE> visited;
    visited.fill(false);

    // In this algorithm, visited[i] == 1 implies m_board[i] == -1.

    std::array<int, 2> territory = { 0, 0 };
    for (int i = 0; i < GO_BOARD_SIZE; ++i) {
        if (m_board[i] == Piece::ZERO) {
            territory[0]++;
            continue;
        }

        if (m_board[i] == Piece::ONE) {
            territory[1]++;
            continue;
        }
        
        if (visited[i]) continue;

        // Run a BFS. 
        
        std::deque<int> q;
        visited[i] = true;
        q.push_back(i);

        int count = 0;
        std::array<bool, 2> possibleTerritory = { true, true };

        while (!q.empty()) {
            int current = q.front();
            q.pop_front();
            ++count;

            assert(m_board[current] == Piece::NONE);
            assert(visited[current] == true);

            for (int neighbor : neighbors(current)) {
                if (m_board[neighbor] == Piece::ZERO) {
                    // Cannot be the territory of player 1.
                    possibleTerritory[1] = false;

                } else if (m_board[neighbor] == Piece::ONE) {
                    // Cannot be the territory of player 0.
                    possibleTerritory[0] = false;

                } else {
                    // Vacant, continue BFS.
                    if (visited[neighbor]) continue;
                    visited[neighbor] = true;
                    q.push_back(neighbor);
                }
            }
        }

        if (possibleTerritory[0] && !possibleTerritory[1]) territory[0] += count;
        if (possibleTerritory[1] && !possibleTerritory[0]) territory[1] += count;
    }

    return territory;
}

GoNode::ActionDist GoNode::computeActionMask() const {
    GoNode::ActionDist mask;
    for (Coord i = 0; i < GO_BOARD_SIZE; ++i) {
        mask[i] = checkLegalPlacement(i, pieceFromPlayer(m_player));
    }

    mask[GO_BOARD_SIZE] = true;

    return mask;
}

void GoNode::setStartNodeImpl() {
    m_parent = nullptr;
    m_action = 0;
    m_actionMask.fill(1.0f);
    m_player = Player::ZERO;
    m_winner = Player::NONE;
    m_isTerminal = false;
    m_board.fill(Piece::NONE);
    m_hash = 0;
    m_depth = 0;
    m_zobristHistorySet.clear();
    m_dsu.clear();
    m_liberties.fill(0);
    m_componentZobristValues.fill(0);
}

std::unique_ptr<GoNode> GoNode::getNextNodeImpl(ActionIdx actionIdx) {

    // Copy the state.

    ActionDist newActionMask = m_actionMask;
    Board newBoard = m_board;
    std::unordered_set<ZobristHash> newZobristHistorySet = m_zobristHistorySet;
    DSU<Coord, GO_BOARD_SIZE> newDSU = m_dsu;
    std::array<LibertyCount, GO_BOARD_SIZE> newLiberties = m_liberties;
    std::array<ZobristHash, GO_BOARD_SIZE> newComponentZobristValues = m_componentZobristValues;

    auto copyNode = std::make_unique<GoNode>(
        m_parent,
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
    );

    if (actionIdx != GO_BOARD_SIZE) {
        // Handle a piece placement.
        assert(actionIdx >= 0 && actionIdx < GO_BOARD_SIZE);
        assert(copyNode->checkLegalPlacement(actionIdx, m_player));

        copyNode->placePiece(actionIdx, pieceFromPlayer(m_player));
    }

    copyNode->m_parent = this;
    copyNode->m_action = actionIdx;
    copyNode->m_player = otherPlayer(m_player);
    ++copyNode->m_depth;

    copyNode->m_isTerminal = m_action == GO_BOARD_SIZE && actionIdx == GO_BOARD_SIZE;
    copyNode->m_actionMask = !copyNode->m_isTerminal ? copyNode->computeActionMask()
                                                     : ActionDist {};

    // Update winner and terminal status.
    if (copyNode->m_isTerminal) {
        std::array<int, 2> territory = copyNode->countTerritory();
        std::array<float, 2> score = { static_cast<float>(territory[0]),
                                       static_cast<float>(territory[1]) };

        score[1] += GO_KOMI;

        if (score[0] > score[1] + 0.1) {
            copyNode->m_winner = Player::ZERO;
        } else if (score[1] > score[0] + 0.1) {
            copyNode->m_winner = Player::ONE;
        } else {
            copyNode->m_winner = Player::NONE;
        }
    }

    return copyNode;
}

GoNode::State GoNode::getGameStateImpl() const {
    std::array<Board, GO_HISTORY_SIZE> history;

    const GoNode* current = this;

    int t = 0;
    while (t < GO_HISTORY_SIZE && current != nullptr) {
        history[t] = current->m_board;
        current = current->m_parent;
        ++t;
    }

    return State { std::move(history), t, m_player };
}

std::array<Value, 2> GoNode::getRewardsImpl() const {
    switch (m_winner) {
    case Player::ZERO: return { 1.0f, -1.0f };
    case Player::ONE:  return { -1.0f, 1.0f };
    default:           return { 0.0f, 0.0f };
    }
}

std::string GoNode::toStringImpl() const {
    std::string str = "";

    str += "Player: " + std::to_string(static_cast<int>(m_player)) + "\n";
    str += "Winner: " + std::to_string(static_cast<int>(m_winner)) + "\n";
    str += "IsTerminal: " + std::to_string(m_isTerminal) + "\n";
    str += "Action: " + std::to_string(m_action) + "\n";
    str += "Depth: " + std::to_string(m_depth) + "\n";
    str += "Hash: " + std::to_string(m_hash) + "\n";


    str += "Board:\n";
    
    str += "  ";
    for (int col = 0; col < GO_BOARD_WIDTH; col++) {
        str += ('A' + col);
        str += " ";
    }
    str += "\n";
    
    for (int row = 0; row < GO_BOARD_WIDTH; row++) {
        str += std::to_string(row) + " ";
        for (int col = 0; col < GO_BOARD_WIDTH; col++) {
            switch (m_board[toCoord(row, col)]) {
            case Piece::NONE:
                str += "+ ";
                break;
                
            case Piece::ZERO:
                // O, colored red. If the last move, then bold it as well.
                if (m_action == toCoord(row, col)) {
                    str += "\x1b[31m\x1b[1mO\x1b[0m\033[0m ";
                } else {
                    str += "\x1b[31mO\033[0m ";
                }
                break;

            case Piece::ONE:
                // X, colored yellow. If the last move, then bold it as well.
                if (m_action == toCoord(row, col)) {
                    str += "\x1b[33m\x1b[1mX\x1b[0m\033[0m ";
                } else {
                    str += "\x1b[33mX\033[0m ";
                }
                break;

            default:
                assert(false);
            }
        }
        str += std::to_string(row);
        str += "\n";   
    }

    str += "  ";
    for (int col = 0; col < GO_BOARD_WIDTH; col++) {
        str += ('A' + col);
        str += " ";
    }
    str += "\n";


    str += "ActionMask:\n";

    str += "  ";
    for (int col = 0; col < GO_BOARD_WIDTH; col++) {
        str += ('A' + col);
        str += " ";
    }
    str += "\n";    

    for (int i = 0; i < GO_BOARD_WIDTH; ++i) {
        str += std::to_string(i) + " ";
        for (int j = 0; j < GO_BOARD_WIDTH; j++) {
            if(m_actionMask[toCoord(i, j)] == 1.0f) {
                str += "1 ";
            } else {
                str += "0 ";
            }
        }
        str += std::to_string(i);
        str += "\n";
    }
    
    str += "  ";
    for (int col = 0; col < GO_BOARD_WIDTH; col++) {
        str += ('A' + col);
        str += " ";
    }
    str += "\n";


    str += "Liberties:\n";

    str += "  ";
    for (int col = 0; col < GO_BOARD_WIDTH; col++) {
        str += ('A' + col);
        str += " ";
    }
    str += "\n";
    

    for (int i = 0; i < GO_BOARD_WIDTH; ++i) {
        str += std::to_string(i) + " ";
        for (int j = 0; j < GO_BOARD_WIDTH; j++) {
            str += std::to_string(getLiberties(toCoord(i, j))) + " ";
        }
        str += std::to_string(i);
        str += "\n";
    }

    str += "  ";
    for (int col = 0; col < GO_BOARD_WIDTH; col++) {
        str += ('A' + col);
        str += " ";
    }
    str += "\n";
    
    str += "  ";


    str += "Territories: " + std::to_string(countTerritory()[0])
                     + " " + std::to_string(countTerritory()[1]) + "\n";

    return str;
}

} // namespace SPRL