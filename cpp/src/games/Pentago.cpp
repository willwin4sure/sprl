#include "Pentago.hpp"

#include <cassert>

namespace SPRL {

Pentago::State Pentago::startState() const {
    return State {};
}

Pentago::Action Pentago::actionIdxToAction(const ActionIdx actionIdx) const {
    const Pentago::Action action =  {
        actionIdx / (PENTAGO_BOARD_SIZE * PENTAGO_NUM_QUADRANTS),
        (actionIdx / PENTAGO_BOARD_SIZE) % 4,
        actionIdx % PENTAGO_BOARD_SIZE
    };
    return action;
}

ActionIdx Pentago::actionToActionIdx(const Pentago::Action& action) const {
    return action.rotDirection * PENTAGO_NUM_ACTIONS / PENTAGO_NUM_ROT_DIRECTIONS
            + action.rotQuadrant * PENTAGO_NUM_ACTIONS / (PENTAGO_NUM_ROT_DIRECTIONS * PENTAGO_NUM_QUADRANTS)
            + action.boardIdx;
}

Pentago::State Pentago::nextState(const State& state, const ActionIdx actionIdx) const {
    assert(!isTerminal(state));
    assert(actionMask(state)[actionIdx] == 1.0f);

    State::Board newBoard = state.getBoard();  // The board that we return, a copy of the original

    const Player player = state.getPlayer();
    const Player newPlayer = 1 - player;

    const Piece piece = static_cast<Piece>(player);

    Action action = actionIdxToAction(actionIdx);

    // Place the piece there
    newBoard[action.boardIdx] = piece;
    // Rotate the board
    /*
    0  1  2     3  4  5
    6  7  8     9  10 11
    12 13 14    15 16 17

    18 19 20    21 22 23
    24 25 26    27 28 29
    30 31 32    33 34 35
    */

   // Some of the most heinous code you've ever seen
   int subboardTopLeft = -1;
   switch(action.rotQuadrant)
   {
    case 0:
        subboardTopLeft = 0;
        break;
    case 1:
        subboardTopLeft = 3;
        break;
    case 2:
        subboardTopLeft = 18;
        break;
    case 3:
        subboardTopLeft = 21;
        break;
    default:
        assert(false);
   }

   switch(action.rotDirection)
   {
    case 0: //clockwise
        Piece temp = newBoard[subboardTopLeft];
        newBoard[subboardTopLeft] = newBoard[subboardTopLeft + 12];
        newBoard[subboardTopLeft + 12] = newBoard[subboardTopLeft + 14];
        newBoard[subboardTopLeft + 14] = newBoard[subboardTopLeft + 2];
        newBoard[subboardTopLeft + 2] = temp;

        temp = newBoard[subboardTopLeft + 1];
        newBoard[subboardTopLeft + 1] = newBoard[subboardTopLeft + 6];
        newBoard[subboardTopLeft + 6] = newBoard[subboardTopLeft + 13];
        newBoard[subboardTopLeft + 13] = newBoard[subboardTopLeft + 8];
        newBoard[subboardTopLeft + 8] = temp;
    case 1: // counterclockwise
        Piece temp = newBoard[subboardTopLeft];
        newBoard[subboardTopLeft] = newBoard[subboardTopLeft + 2];
        newBoard[subboardTopLeft + 2] = newBoard[subboardTopLeft + 14];
        newBoard[subboardTopLeft + 14] = newBoard[subboardTopLeft + 12];
        newBoard[subboardTopLeft + 12] = temp;

        temp = newBoard[subboardTopLeft + 1];
        newBoard[subboardTopLeft + 1] = newBoard[subboardTopLeft + 8];
        newBoard[subboardTopLeft + 8] = newBoard[subboardTopLeft + 13];
        newBoard[subboardTopLeft + 13] = newBoard[subboardTopLeft + 6];
        newBoard[subboardTopLeft + 6] = temp;
    default:
        assert(false);
   }

    // Check if the move wins the game
    Player winner = checkWin(newBoard) ? player : -1;

    return State { newBoard, newPlayer, winner };
}

bool Pentago::isTerminal(const State& state) const {
    if (state.getWinner() != -1) {
        return true;
    }

    // terminal if no empty squares
    const State::Board& board = state.getBoard();
    return std::find(std::begin(board), std::end(board), emptySquare) == std::end(board);
}

Pentago::ActionDist Pentago::actionMask(const State& state) const {
    ActionDist mask {};
    mask.fill(0.0f);

    const State::Board& board = state.getBoard();
    for (int boardIdx = 0; boardIdx < PENTAGO_BOARD_SIZE; boardIdx++) {
        // Check if the top row has a piece for each column
        if (board[boardIdx] == -1) {
            for (int i = 0; i < PENTAGO_NUM_ACTIONS / PENTAGO_BOARD_SIZE; ++i) {
                mask[boardIdx + i * PENTAGO_BOARD_SIZE] = 1.0f;
            }
        }
    }

    return mask;
}

std::pair<Value, Value> Pentago::rewards(const State& state) const {
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

int Pentago::numSymmetries() const {
    // TODO
    return 1;
}

Symmetry Pentago::inverseSymmetry(const Symmetry& symmetry) const {
    // TODO
    return symmetry;
}

std::vector<Pentago::State> Pentago::symmetrizeState(const State& state, const std::vector<Symmetry>& symmetries) const {
    // TODO
    std::vector<State> symmetrizedStates;
    symmetrizedStates.reserve(symmetries.size());

    for (const Symmetry& symmetry : symmetries) {
        switch (symmetry) {
        case 0:
            symmetrizedStates.push_back(state);
            break;

        default:
            assert(false);
        }
    }

    return symmetrizedStates;
}

std::vector<Pentago::ActionDist> Pentago::symmetrizeActionDist(const ActionDist& actionDist, const std::vector<Symmetry>& symmetries) const {
    // TODO
    std::vector<ActionDist> symmetrizedActionDists;
    symmetrizedActionDists.reserve(symmetries.size());

    for (const Symmetry& symmetry : symmetries) {
        ActionDist symmetrizedActionDist {};
        switch (symmetry) {
        case 0:
            symmetrizedActionDist = actionDist;
            symmetrizedActionDists.push_back(symmetrizedActionDist);
            break;

        default:
            assert(false);
        }
    }

    return symmetrizedActionDists;
}


std::string Pentago::stateToString(const State& state) const {
    std::string str = "";

    const State::Board& board = state.getBoard();
    for (int row = 0; row < PENTAGO_BOARD_WIDTH; row++) {
        for (int col = 0; col < PENTAGO_BOARD_WIDTH; col++) {
            switch (board[row * PENTAGO_BOARD_WIDTH + col]) {
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

    for (int col = 0; col < PENTAGO_BOARD_WIDTH; col++) {
        str += std::to_string(col) + " ";
    }

    return str;
}


bool Pentago::checkWin(const State::Board& board) const {
    // horizontal win
    for (int row = 0; row < PENTAGO_BOARD_WIDTH; ++row) {
        for (int col = 0; col <= 1; ++col) {
            Piece piece = board[row * PENTAGO_BOARD_WIDTH + col];
            if (piece == emptySquare) continue;

            bool isRow = true;
            for (int i = 1; i < 5; ++i) {
                if (piece != board[row * PENTAGO_BOARD_WIDTH + col + i]) {
                    isRow = false;
                }
            }
            if (isRow) return true;
        }
    }

    // vertical win
    for (int row = 0; row <= 1; ++row) {
        for (int col = 0; col < PENTAGO_BOARD_WIDTH; ++col) {
            Piece piece = board[row * PENTAGO_BOARD_WIDTH + col];
            if (piece == emptySquare) continue;

            bool isCol = true;
            for (int i = 1; i < 5; ++i) {
                if (piece != board[(row+i) * PENTAGO_BOARD_WIDTH + col]) {
                    isCol = false;
                }
            }
            if (isCol) return true;
        }
    }

    // diagonal win
    for (int row = 0; row <= 1; ++row) {
        for (int col = 0; col <=1; ++col) {
            Piece piece = board[row * PENTAGO_BOARD_WIDTH + col];
            if (piece == emptySquare) continue;

            bool isDiag = true;
            for (int i = 1; i < 5; ++i) {
                if (piece != board[(row+i) * PENTAGO_BOARD_WIDTH + (col+i)]) {
                    isDiag = false;
                }
            }
            if (isDiag) return true;
        }
    }

    // anti-diagonal win
    for (int row = 0; row <= 1; ++row) {
        for (int col = PENTAGO_BOARD_WIDTH - 2; col < PENTAGO_BOARD_WIDTH; ++col) {
            Piece piece = board[row * PENTAGO_BOARD_WIDTH + col];
            if (piece == emptySquare) continue;

            bool isDiag = true;
            for (int i = 1; i < 5; ++i) {
                if (piece != board[(row+i) * PENTAGO_BOARD_WIDTH + (col-i)]) {
                    isDiag = false;
                }
            }
            if (isDiag) return true;
        }
    }
    return false;
}

} // namespace SPRL

