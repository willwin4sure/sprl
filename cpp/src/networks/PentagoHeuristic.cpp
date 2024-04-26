#include "PentagoHeuristic.hpp"

namespace SPRL {


std::vector<std::pair<SPRL::GameActionDist<PTG_NUM_ACTIONS>, SPRL::Value>> PentagoHeuristic::evaluate(
    SPRL::Game<PTG_BOARD_SIZE, PTG_NUM_ACTIONS>* game,
    const std::vector<SPRL::GameState<PTG_BOARD_SIZE>>& states) {

    int numStates = states.size();
    m_numEvals += numStates;
    
    std::vector<std::pair<SPRL::GameActionDist<PTG_NUM_ACTIONS>, SPRL::Value>> results;
    results.reserve(numStates);

    SPRL::GameActionDist<PTG_NUM_ACTIONS> uniformDist;
    for (int i = 0; i < PTG_NUM_ACTIONS; ++i) {
        uniformDist[i] = 1.0f / PTG_NUM_ACTIONS;
    }

    for (const SPRL::GameState<PTG_BOARD_SIZE>& state : states) {
        static constexpr std::array<float, 36> squareValues = {
            0.01, 0.02, 0.02, 0.02, 0.02, 0.01,
            0.02, 0.04, 0.03, 0.03, 0.04, 0.02,
            0.02, 0.03, 0.03, 0.03, 0.03, 0.02,
            0.02, 0.03, 0.03, 0.03, 0.03, 0.02,
            0.02, 0.04, 0.03, 0.03, 0.04, 0.02,
            0.01, 0.02, 0.02, 0.02, 0.02, 0.01,
        };

        // get the sum of the square values of your pieces minus opponent pieces
        float value = 0.0f;
        for (int i = 0; i < PTG_BOARD_SIZE; ++i) {
            Piece piece = state.getBoard()[i];
            if (piece == -1) {
                continue;
            } else if (state.getBoard()[i] == state.getPlayer()) {
                value += squareValues[i];
            } else {
                value -= squareValues[i];
            }
        }

        results.push_back({ uniformDist, value });
    }

    return results;
}

} // namespace SPRL
