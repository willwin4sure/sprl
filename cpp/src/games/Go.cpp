#include "Go.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <queue>

namespace SPRL {

void GoNode::clearComponent(Coord coord, Player player) {
    assert (m_board[coord] == static_cast<Piece>(player));
    if (m_koCoord == NO_KO_COORD) {
        m_koCoord = coord;
        
    } else {
        m_koCoord = MORE_THAN_ONE_CAPTURE_KO_COORD; 
    }

    m_dsu[coord] = coord;
    m_board[coord] = Piece::NONE;
    m_componentZobristValues[coord] = 0;

    m_liberties[coord] = 0;

    // Check for stones which belong to the opposite player
    // adjacent to the current stone in the removed component.
    // Each of these components should gain a liberty.

    std::vector<Coord> opp_neighbor_groups;
    opp_neighbor_groups.reserve(4);

    // ZobristHash state_hash_update = getPieceHash(coord, player);
    for (Coord neighbor : neighbors(coord)) {
        if (m_board[neighbor] == Piece::NONE) {
            continue;
        }

        if (m_board[neighbor] == static_cast<Piece>(player)) {
            clearComponent(neighbor, player);
            
        } else {
            Coord group = parent(neighbor);

            if (std::find(opp_neighbor_groups.begin(), opp_neighbor_groups.end(), group) != opp_neighbor_groups.end()) {
                continue;  // Already counted
            }
            opp_neighbor_groups.push_back(group);
            
            ++getLiberties(neighbor);
        }
    }
    
    // return state_hash_update;
}

ZobristHash GoNode::placePiece(const Coord coordinate, const Player player){
    assert(m_board[coordinate] == Piece::NONE);
    m_board[coordinate] = static_cast<Piece>(player);
    
    m_koCoord = NO_KO_COORD;


    /*
     * Phase one: check for friendly neighbors, and merge all of them together.
    */
    ZobristHash component_hash_update = getPieceHash(coordinate, player);
    
    for (Coord neighbor : neighbors(coordinate)) {
        if (parent(coordinate) == parent(neighbor)){
            // Have already merged this component
            continue;
        }
        
        if (m_board[neighbor] == static_cast<Piece>(player)) {
            component_hash_update ^= m_componentZobristValues[parent(neighbor)];
            m_dsu[parent(coordinate)] = parent(neighbor);
        }
    }
    
    m_componentZobristValues[parent(coordinate)] = component_hash_update;
    

    LibertyCount new_liberties = 0;

    // BFS to re-evaluate the liberties of the new component

    std::vector<bool> visited(GO_BOARD_SIZE, false);
    std::deque<Coord> q;
    visited[coordinate] = true;
    q.push_back(coordinate);
    
    while (!q.empty()) {
        Coord current = q.front();
        q.pop_front();
        
        assert(visited[current]);
        assert(m_board[current] == static_cast<Piece>(player));

        for (Coord neighbor : neighbors(current)) {
            if (m_board[neighbor] == static_cast<Piece>(player)) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    q.push_back(neighbor);
                }
                
            } else if (m_board[neighbor] == Piece::NONE) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;  // Don't double count liberties
                    ++new_liberties;
                }
            }
        }
    }

    /*
    Update the liberties of the new component. At this stage, the liberty count for coordinate is correct,
    assuming that no captured enemy stones have been removed yet.

    (The reason we don't add in the new liberties from the deleted components at this stage is because in the next
    stage we'll add them, plus possible other liberties to other unrelated friendly components).
    */

    getLiberties(coordinate) = new_liberties;

    /*
    Phase two: check for enemy neighbors, and deduct a liberty from those components.
    If a component has no liberties left, it dies. In the process of removing the stones, we also:
     - update the state_hash_update 
     - update the ko coordinate
     - update the liberties of the adjacent groups of player.
    */

    ZobristHash state_hash_update = getPieceHash(coordinate, player); // this is the state_hash_update update for the entire state.

    std::vector<int> opp_neighbor_groups;
    for (Coord neighbor : neighbors(coordinate)) {
        if (m_board[neighbor] != static_cast<Piece>(player)) {
            // check if the parent of neighbor lies in opp_neighbor_groups
            if (std::find(opp_neighbor_groups.begin(), opp_neighbor_groups.end(), parent(neighbor)) != opp_neighbor_groups.end()) {
                continue; // already counted
            }
            opp_neighbor_groups.push_back(parent(neighbor));
            
            --getLiberties(neighbor);
            if (getLiberties(neighbor) == 0){
                state_hash_update ^= m_componentZobristValues[parent(neighbor)];
                clearComponent(neighbor, otherPlayer(player));
            }
        }
    }

    
    // Right-shift the history, and append the new state_hash_update.
    for (int i = 0; i < GO_HISTORY_LENGTH - 1; i++) {
        m_zobristHistory[i + 1] = m_zobristHistory[i];
    }
    m_zobristHistory[0] ^= state_hash_update; // history[0] = history[1] ^ state_hash_update

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


GoNode::ActionDist GoNode::actionMask(const GoNode& state) const {
    GoNode::ActionDist mask;
    for (int32_t i = 0; i < GO_BOARD_SIZE; i++) {
        mask[i] = state.m_board[i] == Piece::NONE && !state.checkSuicide(i, state.m_player);
    }
    mask[m_koCoord] = false; // simple Ko rule
    mask[GO_BOARD_SIZE] = true; // pass is always a valid action
}

void GoNode::setStartNode() {
    m_parent = nullptr;
    m_action = 0;
    m_board.fill(Piece::NONE);
    m_dsu.fill(-1);
    m_liberties.fill(0);
    m_zobristHistory.fill(0);
    m_koCoord = NO_KO_COORD;
    m_passes = 0;
    m_player = Player::ZERO;
    m_actionMask = actionMask(*this);
}




std::unique_ptr<GameNode<GridState<GO_BOARD_SIZE>, GO_ACTION_SIZE>> GoNode::getNextNode(ActionIdx actionIdx) {
    using Board = std::array<int8_t, GO_BOARD_SIZE>;
    // GoNode(GoNode* parent, ActionIdx action, const ActionDist& actionMask,
    //        Player player, Player winner, bool isTerminal, const Board& board,
    //        Coord koCoord, const std::array<ZobristHash, GO_HISTORY_LENGTH>& zobristHistory,
    //        int8_t passes)
    GoNode new_state(
        this,
        actionIdx,
        m_actionMask,  // Needs to be updated
        otherPlayer(m_player),
        Player::NONE,
        false,
        m_board,  // Under here needs to be updated
        m_koCoord, 
        m_zobristHistory,
        m_passes
    );
    if (actionIdx == GO_BOARD_SIZE) {
        // pass
        new_state.pass();
    }else{
        // place piece
        assert (actionIdx >= 0 && actionIdx < GO_BOARD_SIZE);
        assert (state.owners[actionIdx] == Player::NONE);

        new_state.placePiece(actionIdx, state.player);

    }
    return new_state;
}

} // namespace SPRL