#ifndef PENTAGO_HPP
#define PENTAGO_HPP

#include "Game.hpp"

namespace SPRL {

constexpr int PENTAGO_BOARD_WIDTH = 6;
constexpr int PENTAGO_QUADRANT_WIDTH = PENTAGO_BOARD_WIDTH / 2;
constexpr int PENTAGO_NUM_QUADRANTS = 4;
constexpr int PENTAGO_NUM_ROT_DIRECTIONS = 2;
constexpr int PENTAGO_BOARD_SIZE = PENTAGO_BOARD_WIDTH * PENTAGO_BOARD_WIDTH;
constexpr int PENTAGO_NUM_ACTIONS = PENTAGO_BOARD_SIZE * PENTAGO_NUM_QUADRANTS * PENTAGO_NUM_ROT_DIRECTIONS;

/**
 * Implementation of the game Pentago.
*/
class Pentago : public Game<PENTAGO_BOARD_SIZE, PENTAGO_NUM_ACTIONS> {
public:
    State startState() const override;
    State nextState(const State& state, const ActionIdx actionIdx) const override;
    bool isTerminal(const State& state) const override;
    ActionDist actionMask(const State& state) const override;
    std::pair<Value, Value> rewards(const State& state) const override;
    int numSymmetries() const override;
    Symmetry inverseSymmetry(const Symmetry& symmetry) const override;
    std::vector<State> symmetrizeState(const State& state, const std::vector<Symmetry>& symmetries) const override;
    std::vector<ActionDist> symmetrizeActionDist(const ActionDist& actionSpace, const std::vector<Symmetry>& symmetries) const override;

    std::string stateToString(const State& state) const override;

private:
    bool checkWin(const State::Board& board) const;

    static constexpr Piece emptySquare = -1;
    struct Action {
        int8_t rotDirection; // 0 cw, 1 ccw
        int8_t rotQuadrant; // 0 top left, 1 top right, 2 bot left, 3 bot right
        int8_t boardIdx; // index of piece on board
    };
    Action actionIdxToAction(const ActionIdx actionIdx) const;
    ActionIdx actionToActionIdx(const Action& action) const;
};

} // namespace SPRL

#endif