// #ifndef OTHELLO_HEURISTIC_HPP
// #define OTHELLO_HEURISTIC_HPP

// #include "../games/Othello.hpp"

// #include "Network.hpp"

// namespace SPRL {

// class OthelloHeuristic : public Network<OTH_SIZE * OTH_SIZE, OTH_SIZE * OTH_SIZE + 1> {
// public:
//     OthelloHeuristic() = default;

//     std::vector<std::pair<GameActionDist<OTH_SIZE * OTH_SIZE + 1>, Value>> evaluate(
//         const std::vector<GameState<OTH_SIZE * OTH_SIZE>>& states,
//         const std::vector<GameActionDist<OTH_SIZE * OTH_SIZE + 1>>& masks) override;

//     int getNumEvals() override {
//         return m_numEvals;
//     }    

// private:
//     int m_numEvals { 0 };
// };

// } // namespace SPRL

// #endif
