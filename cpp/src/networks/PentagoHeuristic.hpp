#ifndef PENTAGO_HEURISTIC_HPP
#define PENTAGO_HEURISTIC_HPP

#include "../games/Pentago.hpp"

#include "Network.hpp"

namespace SPRL {

class PentagoHeuristic : public SPRL::Network<PTG_BOARD_SIZE, PTG_NUM_ACTIONS> {
public:
    PentagoHeuristic() = default;

    std::vector<std::pair<SPRL::GameActionDist<PTG_NUM_ACTIONS>, SPRL::Value>> evaluate(
        SPRL::Game<PTG_BOARD_SIZE, PTG_NUM_ACTIONS>* game,
        const std::vector<SPRL::GameState<PTG_BOARD_SIZE>>& states) override;

    int getNumEvals() override {
        return m_numEvals;
    }    

private:
    int m_numEvals { 0 };
};

} // namespace SPRL

#endif
