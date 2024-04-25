#ifndef PENTAGO_HPP
#define PENTAGO_HPP

#include "Game.hpp"

namespace SPRL {

constexpr int PTG_BOARD_WIDTH = 6;  // code does not generalize beyond this case
constexpr int PTG_NUM_QUADRS = 4;
constexpr int PTG_NUM_ROT_DIRS = 2;
constexpr int PTG_BOARD_SIZE = PTG_BOARD_WIDTH * PTG_BOARD_WIDTH;
constexpr int PTG_NUM_ACTIONS = PTG_BOARD_SIZE * PTG_NUM_QUADRS * PTG_NUM_ROT_DIRS;

/**
 * Implementation of the game Pentago.
 * 
 * See https://en.m.wikipedia.org/wiki/Pentago for details.
*/
class Pentago : public Game<PTG_BOARD_SIZE, PTG_NUM_ACTIONS> {
public:
    State startState() const override;
    State nextState(const State& state, const ActionIdx actionIdx) const override;
    ActionDist actionMask(const State& state) const override;
    std::pair<Value, Value> rewards(const State& state) const override;
    int numSymmetries() const override;
    Symmetry inverseSymmetry(const Symmetry& symmetry) const override;
    std::vector<State> symmetrizeState(const State& state, const std::vector<Symmetry>& symmetries) const override;
    std::vector<ActionDist> symmetrizeActionDist(const ActionDist& actionSpace, const std::vector<Symmetry>& symmetries) const override;

    std::string stateToString(const State& state) const override;

private:
    enum class RotationDirection : int8_t {
        clockwise = 0,
        counterClockwise = 1
    };

    enum class RotationQuadrant : int8_t {
        topLeft = 0,
        topRight = 1,
        bottomLeft = 2,
        bottomRight = 3
    };

    static constexpr Piece emptySquare = -1;

    struct Action {
        RotationDirection rotDirection;
        RotationQuadrant rotQuadrant;
        int8_t boardIdx; // index of piece on board, in interval [0, PTG_BOARD_SIZE)
    };

    Action actionIdxToAction(const ActionIdx actionIdx) const;
    ActionIdx actionToActionIdx(const Action& action) const;

    bool checkPlacementWin(const State::Board& board, int row, int col, const Piece piece) const;
    std::pair<bool, bool> checkWin(const State::Board& board) const;

    State symmetrizeSingleState(const State& state, Symmetry symmetry) const;
    Action symmetrizeSingleAction(const Action& action, Symmetry symmetry) const;
    ActionDist symmetrizeSingleActionDist(const ActionDist& actionSpace, Symmetry symmetry) const;
};

} // namespace SPRL

#endif