#include "Go.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <queue>

namespace SPRL {

ZobristHash GoNode::clearComponent(Coord coord, Player player) {
    assert (m_board[coord] == static_cast<Piece>(player));
    if (m_koCoord == NO_KO_COORD) {
        m_koCoord = coord;
        
    } else {
        m_koCoord = MORE_THAN_ONE_CAPTURE_KO_COORD; 
    }

    m_dsu[coord] = coord;
    m_board[coord] = Piece::NONE;

    m_liberties[coord] = 0;

    // check for stones which belong to the opposite player, adjacent to the cleared component.
    // each of these components should gain a liberty.

    std::vector<int> opp_neighbor_groups;

    ZobristHash hash = getPieceHash(coord, player);
    for (Coord neighbor : neighbors(coord)) {
        if (m_board[neighbor] == Piece::NONE) {
            continue;
        }
        if (m_board[neighbor] == static_cast<Piece>(player)) {
            hash ^= clearComponent(neighbor, player);
        } else {
            if (std::find(opp_neighbor_groups.begin(), opp_neighbor_groups.end(), parent(neighbor)) != opp_neighbor_groups.end()) {
                continue; // already counted
            }
            opp_neighbor_groups.push_back(parent(neighbor));
            int32_t new_opp_liberties = getLiberties(neighbor) + 1;
            setLiberties(neighbor, new_opp_liberties);
        }
    }
    
    return hash;
}

ZobristHash GoNode::placePiece(const Coord coordinate, const Player who){
    assert(m_board[coordinate] == Piece::NONE);
    m_board[coordinate] = static_cast<Piece>(who);
    
    LibertyCount new_liberties = 0;
    ZobristHash hash = 0;
    m_koCoord = NO_KO_COORD;

    /*
    Phase one: check for friendly neighbors, and merge all of them together.
    Add together all of the inherited liberties.
    */
    for (Coord neighbor : neighbors(coordinate)) {
        if (m_board[neighbor] == static_cast<Piece>(who)) {
            if (parent(neighbor) != parent(coordinate)) {
                // If two neighbors are part of the same component, then
                // the first guy gets evaluated and merged, and the
                // second guy is then detected as being part of the first guy's component
                new_liberties += getLiberties(neighbor) - 1;
                m_dsu[parent(coordinate)] = parent(neighbor);
            }
        }
    }

    /*
    Phase two: check for empty neighbors, and add a liberty for each one.
    */

    for (Coord neighbor : neighbors(coordinate)) {
        if (m_board[neighbor] == Piece::NONE) {
            // This time, we need to check all neighbors of this neighbor.
            // If *any* of them are part of the same group, then this is not a new liberty.
            // Else, it is.
            bool has_friendly_neighbor = false;
            for (Coord neighbor_neighbor : neighbors(neighbor)) {
                if (parent(neighbor_neighbor) == parent(coordinate) && neighbor_neighbor != coordinate) {
                    has_friendly_neighbor = true;
                    break;
                }
            }
            if (!has_friendly_neighbor) {
                new_liberties++;
            }
        }
    }

    /*
    Update the liberties of the new component. At this stage, the liberty count for coordinate is correct,
    assuming that no captured enemy stones have been removed yet.
    */

    setLiberties(coordinate, new_liberties);

    /*
    Phase three: check for enemy neighbors, and deduct a liberty from those components.
    If a component has no liberties left, it dies. In the process of removing the stones, we also:
     - update the hash 
     - update the ko coordinate
     - update the liberties of the adjacent groups of player.
    */


    std::vector<int> opp_neighbor_groups;
    for (Coord neighbor : neighbors(coordinate)) {
        if (m_board[neighbor] != static_cast<Piece>(who)) {
            // check if the parent of neighbor lies in opp_neighbor_groups
            if (std::find(opp_neighbor_groups.begin(), opp_neighbor_groups.end(), parent(neighbor)) != opp_neighbor_groups.end()) {
                continue; // already counted
            }
            opp_neighbor_groups.push_back(parent(neighbor));
            
            int32_t new_opp_liberties = getLiberties(neighbor) - 1;
            setLiberties(neighbor, new_opp_liberties);
            if(new_opp_liberties == 0){
                hash ^= clearComponent(neighbor, otherPlayer(who));
            }
        }
    }

    hash ^= getPieceHash(coordinate, who);
    // Right-shift the history, and append the new hash.
    for (int i = 0; i < GO_HISTORY_LENGTH - 1; i++) {
        m_zobristHistory[i + 1] = m_zobristHistory[i];
    }
    m_zobristHistory[0] ^= hash; // history[0] = history[1] ^ hash

    // zero out the passes
    m_passes = 0;

    if(m_koCoord == MORE_THAN_ONE_CAPTURE_KO_COORD){ // account for the special value
        m_koCoord = NO_KO_COORD;
    }
}

void GoNode::pass() {
    for (int i = 0; i < GO_HISTORY_LENGTH - 1; i++) {
        m_zobristHistory[i+1] = m_zobristHistory[i];
    }
    // history[0] is already state.history[1]
    m_passes++;
    m_player = otherPlayer(m_player);
}

bool GoNode::checkSuicide(const Coord coordinate, const Player player) const{
    assert (m_board[coordinate] == Piece::NONE); // empty square
    assert (player != Player::NONE); // player is either 0 or 1
    assert (m_koCoord != coordinate); // not an illegal move
    for (int32_t neighbor : neighbors(coordinate)) {
        if (neighbor == -1) {
            continue;
        }
        if (m_board[neighbor] == Piece::NONE) {
            return false; // Empty square
        }else if (m_board[neighbor] == static_cast<Piece>(player)) {
            if (getLiberties(neighbor) > 1) {
                return false;
                // We have a liberty because we're attached to a
                // friendly piece with more than one liberty
                // (and we've only consumed one of them, leaving at least one)
            }
        }else{
            if (getLiberties(neighbor) == 1) {
                return false; // We just captured a piece
            }
        }
    }
    // We have no liberties and we're not capturing anything
    return true;
}

/**
 * Returns the rewards for player 0 and 1, which are in (-1, 0, 1) for loss, draw, win.
 * 
 * The algorithm is a WYSIWYG implementation of Go scoring. All stones of a particular color belong to that player.
 * A blank cell belongs to a player if all its neighbors belong to that player, computed using a BFS.
*/
std::array<int32_t, 2> GoNode::computeScore(){
    using Board = std::array<int8_t, GO_BOARD_SIZE>;
    Board visited;
    visited.fill(0);

    // In this algorithm, visited[i] == 1 implies m_board[i] == -1.

    std::array<int32_t, 2> points = {0, 0};
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



std::unique_ptr<GameNode<GridState<GO_BOARD_SIZE>, GO_ACTION_SIZE>> GoNode::getNextNode(ActionIdx actionIdx) {
    using Board = std::array<int8_t, GO_BOARD_SIZE>;
    GoNode new_state = GoNode(state);
    if (actionIdx == GO_BOARD_SIZE) {
        // pass
        new_state.pass();
    }else{
        // place piece
        assert (actionIdx >= 0 && actionIdx < GO_BOARD_SIZE);
        assert (state.owners[actionIdx] == -1); // empty square

        new_state.placePiece(actionIdx, state.player);

    }
    return new_state;
}

GoGame::ActionDist GoNode::actionMask(const GoState& state) const {
    GoGame::ActionDist mask;
    for (int32_t i = 0; i < GO_BOARD_SIZE; i++) {
        mask[i] = state.m_board[i] == -1 && state.m_koCoord != i && !state.checkSuicide(i, state.player);
    }
    mask[GO_BOARD_SIZE] = true; // pass is always a valid action
}

} // namespace SPRL