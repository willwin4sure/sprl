// #ifndef RANDOM_NETWORK_HPP
// #define RANDOM_NETWORK_HPP

// #include "Network.hpp"

// namespace SPRL {

// template <int BOARD_SIZE, int ACTION_SIZE> 
// class RandomNetwork : public Network<BOARD_SIZE, ACTION_SIZE> {
// public:
//     RandomNetwork() {}

//     std::vector<std::pair<GameActionDist<ACTION_SIZE>, Value>> evaluate(
//         const std::vector<GameState<BOARD_SIZE>>& states,
//         const std::vector<GameActionDist<ACTION_SIZE>>& masks) override {

//         int numStates = states.size();

//         m_numEvals += numStates;

//         // Return a uniform distribution and a value of 0 for everything
//         std::vector<std::pair<GameActionDist<ACTION_SIZE>, Value>> results;
//         results.reserve(numStates);

//         for (int b = 0; b < numStates; ++b) {
//             int numLegal = 0;
//             for (int i = 0; i < ACTION_SIZE; ++i) {
//                 if (masks[0][i] == 1.0f) {
//                     ++numLegal;
//                 }
//             }

//             float uniform = 1.0f / numLegal;

//             GameActionDist<ACTION_SIZE> uniformDist;
//             for (int i = 0; i < ACTION_SIZE; ++i) {
//                 if (masks[0][i] == 1.0f) {
//                     uniformDist[i] = uniform;
//                 } else {
//                     uniformDist[i] = 0.0f;
//                 }
//             }

//             results.push_back({ uniformDist, 0.0f });
//         }

//         return results;
//     }

//     int getNumEvals() override {
//         return m_numEvals;
//     }

// private:
//     int m_numEvals { 0 };
// };

// } // namespace SPRL

// #endif
