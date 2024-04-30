#ifndef PENTAGO_HEURISTIC_HPP
#define PENTAGO_HEURISTIC_HPP

#include "../games/Pentago.hpp"

#include "Network.hpp"

namespace SPRL {

class PentagoHeuristic : public Network<PTG_BOARD_SIZE, PTG_NUM_ACTIONS> {
public:
    PentagoHeuristic() = default;

    std::vector<std::pair<GameActionDist<PTG_NUM_ACTIONS>, Value>> evaluate(
        const std::vector<GameState<PTG_BOARD_SIZE>>& states,
        const std::vector<GameActionDist<PTG_NUM_ACTIONS>>& masks) override;

    int getNumEvals() override {
        return m_numEvals;
    }    

private:
    int m_numEvals { 0 };
};

} // namespace SPRL

#endif
