#include "Go.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <queue>

namespace SPRL {

ZobristHash GoNode::clearComponent(Coord coord, Player player) {
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

    std::vector<int> opp_neighbor_groups;
    for (Coord neighbor : neighbors(coordinate)) {
        if (m_board[neighbor] == Piece::NONE) {
            // We need to determine if this is actually a new liberty. Postponed until we finish union-ing all the neighbors.
        } else if (m_board[neighbor] == static_cast<Piece>(who)) {
            if (parent(neighbor) != parent(coordinate)) {
                // If two neighbors are part of the same component, then
                // the first guy gets evaluated and merged, and the
                // second guy is then detected as being part of the first guy's component
                new_liberties += getLiberties(neighbor) - 1;
                m_dsu[parent(coordinate)] = parent(neighbor);
            }
        } else {
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

    setLiberties(coordinate, new_liberties);

    hash ^= getPieceHash(coordinate, who);
    // Right-shift the history, and append the new hash.
    for (int i = 0; i < GO_HISTORY_LENGTH - 1; i++) {
        m_history[i + 1] = m_history[i];
    }
    m_history[9] ^= hash; // history[0] = history[1] ^ hash

    // zero out the passes
    passes = 0;

    if(m_koCoord == MORE_THAN_ONE_CAPTURE_KO_COORD){ // account for the special value
        m_koCoord = NO_KO_COORD;
    }
}

void GoNode::pass() {
    for (int i = 0; i < GO_HISTORY_LENGTH - 1; i++) {
        m_history[i+1] = m_history[i];
    }
    // history[0] is already state.history[1]
    passes++;
    player = 1 - player;
}

bool GoNode::checkSuicide(const int32_t coordinate, const int8_t who) const{
    assert (m_board[coordinate] == -1); // empty square
    assert (who == 0 || who == 1); // player 0 or 1
    assert (m_koCoord != coordinate); // not an illegal move
    for (int32_t neighbor : neighbors(coordinate)) {
        if (neighbor == -1) {
            continue;
        }
        if (m_board[neighbor] == -1) {
            return false; // Empty square
        }else if (m_board[neighbor] == who) {
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
std::pair<int, int> GoNode::compute_score(){
    using Board = std::array<int8_t, GO_BOARD_SIZE>;
    Board visited;
    visited.fill(0);

    // In this algorithm, visited[i] == 1 implies m_board[i] == -1.

    std::pair<int, int> points = {0, 0};
    for (int i = 0; i < GO_BOARD_SIZE; i++) {
        if (m_board[i] == 0){
            points.first++;
            continue;
        }
        if (m_board[i] == 1){
            points.second++;
            continue;
        }
        
        if (visited[i] != 0)
            continue;
        
        std::queue<int> q;
        q.push(i);
        visited[i] = 1;
        int count = 0;
        std::pair<bool, bool> possible_territory = {true, true}; // {black, white}.
        while (!q.empty()) {
            int current = q.front();
            q.pop();
            count++;
            // Some assert statements just to verify the correctness of the algorithm
            assert (m_board[current] == -1);
            assert (visited[current] == 1);
            for (int neighbor : neighbors(current)) {
                if (m_board[neighbor] == 0) {
                    // touches a black stone, hence it cannot be white territory
                    possible_territory.second = false;
                }
                else if (m_board[neighbor] == 1) {
                    // touches a white stone, hence it cannot be black territory
                    possible_territory.second = false;
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
        if (possible_territory.first) {
            if (possible_territory.second) {
                // The only situation in which this occurs is if the board is completely empty.
            }else{
                points.first += count;
            }
        } else {
            if (possible_territory.second) {
                points.second += count;
            } else {
                // Neither black nor white territory
            }
        }
    }
    return points;
}





GoState GoNode::startState() const {
    return GoState {};
}

ActionIdx GoNode::actionToActionIdx(const GoGame::Action& action) const{
    if (action.pass) {
        assert (action.boardIdx == 0); // pass action has boardIdx = 0
        return GO_BOARD_SIZE;
    }
    assert (action.boardIdx >= 0 && action.boardIdx < GO_BOARD_SIZE);
    return action.boardIdx;
}

GoGame::Action GoNode::actionIdxToAction(const ActionIdx actionIdx) const {
    if (actionIdx == GO_BOARD_SIZE) {
        return {true, 0};
    }
    assert (actionIdx >= 0 && actionIdx < GO_BOARD_SIZE);
    return {false, static_cast<int8_t>(actionIdx)};
}

GoState GoNode::nextState(const GoState& state, const ActionIdx actionIdx) const {
    using Board = std::array<int8_t, GO_BOARD_SIZE>;
    GoState new_state = GoState(state);
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