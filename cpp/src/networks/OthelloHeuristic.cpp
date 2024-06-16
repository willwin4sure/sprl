#include "OthelloHeuristic.hpp"

namespace SPRL {

std::vector<std::pair<OthelloHeuristic::ActionDist, Value>> OthelloHeuristic::evaluate(
    const std::vector<State>& states,
    const std::vector<ActionDist>& masks) {

    int numStates = states.size();
    m_numEvals += numStates;
    
    std::vector<std::pair<ActionDist, Value>> results;
    results.reserve(numStates);

    for (int b = 0; b < numStates; ++b) {
        int numLegal = 0;
        for (int i = 0; i < OTH_ACTION_SIZE; ++i) {
            if (masks[b][i] > 0.0f) ++numLegal;
        }

        float uniform = 1.0f / numLegal;

        ActionDist uniformDist;
        for (int i = 0; i < OTH_ACTION_SIZE; ++i) {
            uniformDist[i] = (masks[b][i] > 0.0f) ? uniform : 0.0f;
        }

        const State& state = states[b];

        // Count the number of empty squares
        int numEmpty = 0;
        for (int i = 0; i < OTH_BOARD_SIZE; ++i) {
            if (state.getHistory()[0][i] == Piece::NONE) {
                ++numEmpty;
            }
        }

        const Player opponent = otherPlayer(state.getPlayer());

        const ActionDist& oppMask = OthelloNode::actionMask(state.getHistory()[0], opponent);

        // Count number of legal moves per player
        int numOppLegal = 0;

        for (int i = 0; i < OTH_BOARD_SIZE; ++i) {
            if (oppMask[i] > 0.0f) ++numOppLegal;
        }

        results.push_back({ uniformDist, static_cast<float>(numLegal - numOppLegal) / numEmpty });
    }

    return results;
}

} // namespace SPRL
