#ifndef TRACED_PYTORCH_NETWORK_HPP
#define TRACED_PYTORCH_NETWORK_HPP

#include <torch/torch.h>
#include <torch/script.h>

#include "Network.hpp"

namespace SPRL {

class ConnectFourNetwork : public SPRL::Network<42, 7> {
public:
    ConnectFourNetwork(std::string path);

    /**
     * Implementation of evaluate for Connect Four, including
     * the proper embedding of the game state.
    */
    std::vector<std::pair<SPRL::GameActionDist<7>, SPRL::Value>> evaluate(
        SPRL::Game<42, 7>* game,
        const std::vector<SPRL::GameState<42>>& states) override;

    /// Tracks the number of evaluations made by the network, summed over batches.
    int m_numEvals { 0 };

private:
    torch::Device m_device { torch::kCPU };
    std::shared_ptr<torch::jit::script::Module> m_model;
};

} // namespace SPRL

#endif