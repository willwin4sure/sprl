#ifndef SPRL_CONNECT_FOUR_SYMMETRIZER_HPP
#define SPRL_CONNECT_FOUR_SYMMETRIZER_HPP

#include "ISymmetrizer.hpp"

#include "../games/ConnectFourNode.hpp"

namespace SPRL {

/**
 * Symmetrizer for the Connect Four game.
*/
class ConnectFourSymmetrizer : public ISymmetrizer<GridState<C4_BOARD_SIZE, C4_HISTORY_SIZE>, C4_ACTION_SIZE> {
public:
    using Board = ConnectFourNode::Board;
    using State = ConnectFourNode::State;
    using ActionDist = ConnectFourNode::ActionDist;

    int numSymmetries() const override;
    SymmetryIdx inverseSymmetry(SymmetryIdx symmetry) const override;

    std::vector<State> symmetrizeState(const State& state,
                                       const std::vector<SymmetryIdx>& symmetries) const override;

    std::vector<ActionDist> symmetrizeActionDist(const ActionDist& actionDist,
                                                 const std::vector<SymmetryIdx>& symmetries) const override;
};

} // namespace SPRL

#endif