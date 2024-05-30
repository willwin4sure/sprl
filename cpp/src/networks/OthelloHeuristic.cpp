// #include "OthelloHeuristic.hpp"

// namespace SPRL {

// std::vector<std::pair<GameActionDist<OTH_SIZE * OTH_SIZE + 1>, Value>> OthelloHeuristic::evaluate(
//     const std::vector<GameState<OTH_SIZE * OTH_SIZE>>& states,
//     const std::vector<GameActionDist<OTH_SIZE * OTH_SIZE + 1>>& masks) {

//     static Othello othello {};

//     int numStates = states.size();
//     m_numEvals += numStates;
    
//     std::vector<std::pair<GameActionDist<OTH_SIZE * OTH_SIZE + 1>, Value>> results;
//     results.reserve(numStates);

//     GameActionDist<OTH_SIZE * OTH_SIZE + 1> uniformDist;
//     for (int i = 0; i < OTH_SIZE * OTH_SIZE + 1; ++i) {
//         uniformDist[i] = 1.0f / OTH_SIZE * OTH_SIZE + 1;
//     }

//     for (int i = 0; i < numStates; ++i) {
//         const GameState<OTH_SIZE * OTH_SIZE>& state = states[i];

//         // Count the number of empty squares
//         int numEmpty = 0;
//         for (int i = 0; i < OTH_SIZE * OTH_SIZE; ++i) {
//             if (state.getBoard()[i] == -1) {
//                 numEmpty++;
//             }
//         }

//         const Player opponent = 1 - state.getPlayer();

//         const GameActionDist<OTH_SIZE * OTH_SIZE + 1>& mask = masks[i];
//         const GameActionDist<OTH_SIZE * OTH_SIZE + 1>& oppMask = othello.actionMask(state.getBoard(), opponent);

//         // Count number of legal moves per player
//         int numLegal = 0;
//         int numOppLegal = 0;

//         for (int i = 0; i < OTH_SIZE * OTH_SIZE; ++i) {
//             if (mask[i] > 0.0f) {
//                 numLegal++;
//             }
//             if (oppMask[i] > 0.0f) {
//                 numOppLegal++;
//             }
//         }

//         results.push_back({ uniformDist, static_cast<float>(numLegal - numOppLegal) / numEmpty });
//     }

//     return results;
// }

// } // namespace SPRL
