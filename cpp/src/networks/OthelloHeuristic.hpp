#ifndef SPRL_OTHELLO_HEURISTIC_HPP
#define SPRL_OTHELLO_HEURISTIC_HPP

#include "../games/OthelloNode.hpp"

#include "INetwork.hpp"

namespace SPRL {

/**
 * A basic heuristic for Othello that returns a uniform policy and a value
 * equal to your number of legal moves minus the opponent's number of legal moves,
 * divided by the number of empty spaces on the board so it is in the range `[-1, 1]`.
 */
class OthelloHeuristic : public INetwork<GridState<OTH_BOARD_SIZE, OTH_HISTORY_SIZE>, OTH_ACTION_SIZE> {
public:
    using ActionDist = GameActionDist<OTH_ACTION_SIZE>;
    using State = GridState<OTH_BOARD_SIZE, OTH_HISTORY_SIZE>;

    OthelloHeuristic() = default;

    std::vector<std::pair<ActionDist, Value>> evaluate(
        const std::vector<State>& states,
        const std::vector<ActionDist>& masks) override;

    int getNumEvals() override {
        return m_numEvals;
    }    

private:
    int m_numEvals { 0 };
};

} // namespace SPRL

#endif
