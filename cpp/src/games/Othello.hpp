#ifndef OTHELLO_HPP
#define OTHELLO_HPP

#include "Game.hpp"

namespace SPRL {

constexpr int OTH_SIZE = 8;

/**
 * Implementation of Othello/Reversi
*/
class Othello : public Game<OTH_SIZE * OTH_SIZE, OTH_SIZE * OTH_SIZE + 1> {
public:
    State startState() const override;
    State nextState(const State& state, const ActionIdx action) const override;
    ActionDist actionMask(const State& state) const override;
    std::pair<Value, Value> rewards(const State& state) const override;
    int numSymmetries() const override;
    Symmetry inverseSymmetry(const Symmetry& symmetry) const override;
    std::vector<State> symmetrizeState(const State& state, const std::vector<Symmetry>& symmetries) const override;
    std::vector<ActionDist> symmetrizeActionDist(const ActionDist& actionSpace, const std::vector<Symmetry>& symmetries) const override;

    std::string stateToString(const State& state) const override;

private:
    bool isTerminal(const State::Board& board) const;
    ActionDist actionMask(const State::Board& state, const Player player) const;
    const std::vector<int> captures(const State::Board& board, const int row, const int col, const Piece piece) const;
    bool canCapture(const State::Board& board, int row, int col, const Piece piece) const;

    State symmetrizeSingleState(const State& state, Symmetry symmetry) const;
    ActionDist symmetrizeSingleActionDist(const ActionDist& actionSpace, Symmetry symmetry) const;

    friend class OthelloHeuristic;
};

} // namespace SPRL

#endif