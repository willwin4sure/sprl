// #ifndef PENTAGO_NETWORK_HPP
// #define PENTAGO_NETWORK_HPP

// #include <torch/torch.h>
// #include <torch/script.h>

// #include "Network.hpp"

// namespace SPRL {

// class PentagoNetwork : public Network<36, 288> {
// public:
//     PentagoNetwork(std::string path);

//     /**
//      * Implementation of evaluate for Connect Four, including
//      * the proper embedding of the game state.
//     */
//     std::vector<std::pair<GameActionDist<288>, Value>> evaluate(
//         const std::vector<GameState<36>>& states,
//         const std::vector<GameActionDist<288>>& masks) override;

//     int getNumEvals() override {
//         return m_numEvals;
//     }    

// private:
//     int m_numEvals { 0 };

//     torch::Device m_device { torch::kCPU };
//     std::shared_ptr<torch::jit::script::Module> m_model;
// };

// } // namespace SPRL

// #endif
