#include "Othello.hpp"

#include <cassert>
#include <iostream>

inline int toIndex(int row, int col) {
    return row * SPRL::OTH_SIZE + col;
}

inline bool inBounds(int row, int col) {
    return 0 <= row && row < SPRL::OTH_SIZE && 0 <= col && col < SPRL::OTH_SIZE;
}

namespace SPRL {

Othello::State Othello::startState() const {
    GameBoard<OTH_SIZE * OTH_SIZE> board{};
    board.fill(-1);
    board[toIndex(3, 3)] = 1;
    board[toIndex(3, 4)] = 0;
    board[toIndex(4, 3)] = 0;
    board[toIndex(4, 4)] = 1;
    return State {board, 0, -1, false};
}

const std::vector<int> Othello::captures(const State::Board& board, const int row, const int col, const Piece piece) const {
    std::vector<int> output {};

    constexpr std::array<int, 8> r_delta {1, 1, 0, -1, -1, -1, 0, 1};
    constexpr std::array<int, 8> c_delta {0, 1, 1, 1, 0, -1, -1, -1};

    const Piece opponentPiece = 1 - piece;

    for (int i = 0; i < 8; ++i) {
        int nextRow = row + r_delta[i];
        int nextCol = col + c_delta[i];

        while (inBounds(nextRow, nextCol) && board[toIndex(nextRow, nextCol)] == opponentPiece) {
            nextRow += r_delta[i];
            nextCol += c_delta[i];
        }

        if (inBounds(nextRow, nextCol) && board[toIndex(nextRow, nextCol)] == piece) {
            for (int r = row + r_delta[i], c = col + c_delta[i]; r != nextRow || c != nextCol; r += r_delta[i], c += c_delta[i]) {
                output.push_back(toIndex(r, c));
            }
        }
    }

    // std::cout << "Output size: " << output.size() << std::endl;
    // std::cout << "Outputs: " << std::endl;
    // for (int i = 0; i < output.size(); ++i) {
    //     std::cout << output[i] << " ";
    // }
    // std::cout << std::endl;

    return output;
}

bool Othello::canCapture(const State::Board& board, int row, int col, const Piece piece) const {
    constexpr std::array<int, 8> r_delta {1, 1, 0, -1, -1, -1, 0, 1};
    constexpr std::array<int, 8> c_delta {0, 1, 1, 1, 0, -1, -1, -1};

    const Piece opponentPiece = 1 - piece;

    for (int i = 0; i < 8; ++i) {
        int nextRow = row + r_delta[i];
        int nextCol = col + c_delta[i];

        bool oppExists = false;

        while (inBounds(nextRow, nextCol) && board[toIndex(nextRow, nextCol)] == opponentPiece) {
            nextRow += r_delta[i];
            nextCol += c_delta[i];

            oppExists = true;
        }

        if (oppExists && inBounds(nextRow, nextCol) && board[toIndex(nextRow, nextCol)] == piece) {
            return true;
        }
    }

    return false;
}

Othello::State Othello::nextState(const State& state, const ActionIdx action) const {
    assert(!state.isTerminal());
    assert(actionMask(state)[action] == 1.0f);

    State::Board newBoard = state.getBoard();  // The board that we return, a copy of the original

    const Player player = state.getPlayer();
    const Player newPlayer = 1 - player;

    const Piece piece = static_cast<Piece>(player);

    // Action index 64 is a pass
    if (action != OTH_SIZE * OTH_SIZE) {
        // Place the piece there
        newBoard[action] = piece;

        int row = action / OTH_SIZE;
        int col = action % OTH_SIZE;

        // Perform all the captures
        for (const int idx : captures(newBoard, row, col, piece)) {
            newBoard[idx] = piece;
        }
    }

    Player winner = -1;
    const bool terminal = isTerminal(newBoard);

    if (terminal) {
        int count0 = 0;
        int count1 = 0;
        
        for (int i = 0; i < OTH_SIZE * OTH_SIZE; ++i) {
            if (newBoard[i] == 0) {
                count0++;
            }
            else if (newBoard[i] == 1) {
                count1++;
            }
        }

        if (count0 > count1) winner = 0;
        if (count1 > count0) winner = 1;
    }

    return State { newBoard, newPlayer, winner, terminal };
}

Othello::ActionDist Othello::actionMask(const State::Board& board, const Player player) const {
    ActionDist mask {};
    mask.fill(0.0f);

    for (int row = 0; row < OTH_SIZE; ++row) {
        for (int col = 0; col < OTH_SIZE; ++col) {
            if (board[toIndex(row, col)] != -1) continue;
            // mask[toIndex(row, col)] = (captures(board, row, col, static_cast<Piece>(player)).size() > 0) ? 1.0f : 0.0f;
            mask[toIndex(row, col)] = canCapture(board, row, col, static_cast<Piece>(player)) ? 1.0f : 0.0f;
        }
    }
    
    bool canPass = true;
    for (int i = 0; i < OTH_SIZE * OTH_SIZE; ++i) {
        if (mask[i] > 0) canPass = false;
    }

    mask[OTH_SIZE * OTH_SIZE] = canPass ? 1.0f : 0.0f;

    return mask;
}

Othello::ActionDist Othello::actionMask(const State& state) const {
    return actionMask(state.getBoard(), state.getPlayer());
}

std::pair<Value, Value> Othello::rewards(const State& state) const {
    const Player winner = state.getWinner();
    switch (winner) {
    case 0:
        return { 1.0f, -1.0f };
    case 1:
        return { -1.0f, 1.0f };
    default:
        return { 0.0f, 0.0f };
    }
}

int Othello::numSymmetries() const {
    // Symmetry group is dihedral D_4: we have four rotations and four reflected rotations
    //
    // Mapping:
    //     0: identity
    //     1: single 90 deg cw rotation
    //     2: full 180 deg rotation
    //     3: single 90 deg ccw rotation
    //     4: reflection across vertical axis
    //     5: reflection across vertical axis followed by single 90 deg cw rotation
    //     6: reflection across vertical axis followed by full 180 deg rotation
    //     7: reflection across vertical axis followed by single 90 deg ccw rotation

    return 8;
}

Symmetry Othello::inverseSymmetry(const Symmetry& symmetry) const {
    assert(symmetry >= 0);
    assert(symmetry < 8);

    static constexpr std::array<Symmetry, 8> inverseSymmetries = { 0, 3, 2, 1, 4, 5, 6, 7 };
    return inverseSymmetries[symmetry];
}

// Copied from Pentago
Othello::State Othello::symmetrizeSingleState(const State& state, Symmetry symmetry) const {
    switch (symmetry) {
    case 0: {
        // identity
        return state;
    }
    
    case 1: {
        // single 90 deg cw rotation
        State::Board board = state.getBoard(); // Copy of the board
        for (int row = 0; row < OTH_SIZE / 2; ++row) {
            for (int col = 0; col < OTH_SIZE / 2; ++col) {
                Piece temp = board[row * OTH_SIZE + col];
                board[row * OTH_SIZE + col] = board[(OTH_SIZE - 1 - col) * OTH_SIZE + row];
                board[(OTH_SIZE - 1 - col) * OTH_SIZE + row] = board[(OTH_SIZE - 1 - row) * OTH_SIZE + (OTH_SIZE - 1 - col)];
                board[(OTH_SIZE - 1 - row) * OTH_SIZE + (OTH_SIZE - 1 - col)] = board[col * OTH_SIZE + (OTH_SIZE - 1 - row)];
                board[col * OTH_SIZE + (OTH_SIZE - 1 - row)] = temp;
            }
        }
        return State { board, state.getPlayer(), state.getWinner(), state.isTerminal() };
    }

    case 2: {
        // full 180 deg rotation
        State::Board board = state.getBoard(); // Copy of the board
        for (int row = 0; row < OTH_SIZE / 2; ++row) {
            for (int col = 0; col < OTH_SIZE; ++col) {
                std::swap(board[row * OTH_SIZE + col],
                            board[(OTH_SIZE - 1 - row) * OTH_SIZE + (OTH_SIZE - 1 - col)]);
            }
        }
        return State { board, state.getPlayer(), state.getWinner(), state.isTerminal() };
    }

    case 3: {
        // single 90 deg ccw rotation
        State::Board board = state.getBoard(); // Copy of the board
        for (int row = 0; row < OTH_SIZE / 2; ++row) {
            for (int col = 0; col < OTH_SIZE / 2; ++col) {
                Piece temp = board[row * OTH_SIZE + col];
                board[row * OTH_SIZE + col] = board[col * OTH_SIZE + (OTH_SIZE - 1 - row)];
                board[col * OTH_SIZE + (OTH_SIZE - 1 - row)] = board[(OTH_SIZE - 1 - row) * OTH_SIZE + (OTH_SIZE - 1 - col)];
                board[(OTH_SIZE - 1 - row) * OTH_SIZE + (OTH_SIZE - 1 - col)] = board[(OTH_SIZE - 1 - col) * OTH_SIZE + row];
                board[(OTH_SIZE - 1 - col) * OTH_SIZE + row] = temp;
            }
        }
        return State { board, state.getPlayer(), state.getWinner(), state.isTerminal() };
    }

    case 4: case 5: case 6: case 7: {
        // first perform reflection across vertical axis
        State::Board board = state.getBoard(); // Copy of the board
        for (int row = 0; row < OTH_SIZE; ++row) {
            for (int col = 0; col < OTH_SIZE / 2; ++col) {
                std::swap(board[row * OTH_SIZE + col], board[row * OTH_SIZE + OTH_SIZE - 1 - col]);
            }
        }

        // then perform rotation by reusing the code above
        return symmetrizeSingleState(State { board, state.getPlayer(), state.getWinner(), state.isTerminal() },
                                     static_cast<Symmetry>(symmetry - 4));
    }

    default:
        assert(false);
    }

    // Should not reach here.
    return state;
}

std::vector<Othello::State> Othello::symmetrizeState(const State& state, const std::vector<Symmetry>& symmetries) const {
    std::vector<State> symmetrizedStates;
    symmetrizedStates.reserve(symmetries.size());

    for (const Symmetry& symmetry : symmetries) {
        symmetrizedStates.push_back(symmetrizeSingleState(state, symmetry));
    }

    return symmetrizedStates;
}

Othello::ActionDist Othello::symmetrizeSingleActionDist(const ActionDist& actionSpace, Symmetry symmetry) const {
    switch (symmetry) {
    case 0: {
        // identity
        return actionSpace;
    }
    
    case 1: {
        // single 90 deg cw rotation
        ActionDist newSpace = actionSpace; // Copy of the board
        for (int row = 0; row < OTH_SIZE / 2; ++row) {
            for (int col = 0; col < OTH_SIZE / 2; ++col) {
                float temp = newSpace[row * OTH_SIZE + col];
                newSpace[row * OTH_SIZE + col] = newSpace[(OTH_SIZE - 1 - col) * OTH_SIZE + row];
                newSpace[(OTH_SIZE - 1 - col) * OTH_SIZE + row] = newSpace[(OTH_SIZE - 1 - row) * OTH_SIZE + (OTH_SIZE - 1 - col)];
                newSpace[(OTH_SIZE - 1 - row) * OTH_SIZE + (OTH_SIZE - 1 - col)] = newSpace[col * OTH_SIZE + (OTH_SIZE - 1 - row)];
                newSpace[col * OTH_SIZE + (OTH_SIZE - 1 - row)] = temp;
            }
        }
        return newSpace;
    }

    case 2: {
        // full 180 deg rotation
        ActionDist newSpace = actionSpace; // Copy of the board
        for (int row = 0; row < OTH_SIZE / 2; ++row) {
            for (int col = 0; col < OTH_SIZE; ++col) {
                std::swap(newSpace[row * OTH_SIZE + col],
                            newSpace[(OTH_SIZE - 1 - row) * OTH_SIZE + (OTH_SIZE - 1 - col)]);
            }
        }
        return newSpace;
    }

    case 3: {
        // single 90 deg ccw rotation
        ActionDist newSpace = actionSpace; // Copy of the board
        for (int row = 0; row < OTH_SIZE / 2; ++row) {
            for (int col = 0; col < OTH_SIZE / 2; ++col) {
                float temp = newSpace[row * OTH_SIZE + col];
                newSpace[row * OTH_SIZE + col] = newSpace[col * OTH_SIZE + (OTH_SIZE - 1 - row)];
                newSpace[col * OTH_SIZE + (OTH_SIZE - 1 - row)] = newSpace[(OTH_SIZE - 1 - row) * OTH_SIZE + (OTH_SIZE - 1 - col)];
                newSpace[(OTH_SIZE - 1 - row) * OTH_SIZE + (OTH_SIZE - 1 - col)] = newSpace[(OTH_SIZE - 1 - col) * OTH_SIZE + row];
                newSpace[(OTH_SIZE - 1 - col) * OTH_SIZE + row] = temp;
            }
        }
        return newSpace;
    }

    case 4: case 5: case 6: case 7: {
        // first perform reflection across vertical axis
        ActionDist newSpace = actionSpace; // Copy of the board
        for (int row = 0; row < OTH_SIZE; ++row) {
            for (int col = 0; col < OTH_SIZE / 2; ++col) {
                std::swap(newSpace[row * OTH_SIZE + col], newSpace[row * OTH_SIZE + OTH_SIZE - 1 - col]);
            }
        }

        // then perform rotation by reusing the code above
        return symmetrizeSingleActionDist(newSpace, static_cast<Symmetry>(symmetry - 4));
    }

    default:
        assert(false);
    }

    // Should not reach here.
    return actionSpace;
}

std::vector<Othello::ActionDist> Othello::symmetrizeActionDist(const ActionDist& actionDist, const std::vector<Symmetry>& symmetries) const {
    std::vector<ActionDist> symmetrizedActionDists;
    symmetrizedActionDists.reserve(symmetries.size());

    for (const Symmetry& symmetry : symmetries) {
        symmetrizedActionDists.push_back(symmetrizeSingleActionDist(actionDist, symmetry));
    }

    return symmetrizedActionDists;
}


std::string Othello::stateToString(const State& state) const {
    std::string str = "";

    const State::Board& board = state.getBoard();
    for (int row = 0; row < OTH_SIZE; row++) {
        str += std::to_string(row) + " ";
        for (int col = 0; col < OTH_SIZE; col++) {
            switch (board[toIndex(row, col)]) {
            case -1:
                str += ". ";
                break;
            case 0:
                // colored red
                str += "\033[31mO\033[0m ";
                break;
            case 1:
                // colored yellow
                str += "\033[33mX\033[0m ";
                break;
            default:
                assert(false);
            }
        }
        str += "\n";   
    }

    str += "  ";
    for (int col = 0; col < OTH_SIZE; col++) {
        str += ('A' + col);
        str += " ";
    }

    return str;
}

bool Othello::isTerminal(const State::Board& board) const {
    bool output = true;

    ActionDist mask0 = actionMask(board, 0);

    // if you cannot pass, then you can move
    if (mask0[OTH_SIZE * OTH_SIZE] == 0.0f) return false;

    ActionDist mask1 = actionMask(board, 1);

    // returns true iff both players can only pass
    return mask1[OTH_SIZE * OTH_SIZE] > 0.0f;
}

} // namespace SPRL

