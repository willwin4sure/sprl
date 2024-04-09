#ifndef CONNECT_FOUR_HPP
#define CONNECT_FOUR_HPP

#include "Game.hpp"

namespace SPRL {

constexpr int C4_NUM_ROWS = 6;
constexpr int C4_NUM_COLS = 7;

/**
 * Implementation of the classic Connect Four game.
*/
class ConnectFour : public Game<C4_NUM_ROWS * C4_NUM_COLS, C4_NUM_COLS> {
public:
    State startState() const override;
    State nextState(const State& state, const ActionIdx action) const override;
    bool isTerminal(const State& state) const override;
    ActionSpace actionMask(const State& state) const override;
    std::pair<float, float> rewards(const State& state) const override;
    int numSymmetries() const override;
    Symmetry inverseSymmetry(const Symmetry& symmetry) const override;
    std::vector<State> symmetrizeState(const State& state, const std::vector<Symmetry>& symmetries) const override;
    std::vector<ActionSpace> symmetrizeActionSpace(const ActionSpace& actionSpace, const std::vector<Symmetry>& symmetries) const override;

    std::string stateToString(const State& state) const override;

private:
    bool checkWin(const std::array<Piece, C4_NUM_ROWS * C4_NUM_COLS>& board, int row, int col, const Piece piece) const;
};

} // namespace SPRL

#endif