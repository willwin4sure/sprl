#ifndef PENTAGO_NETWORK_HPP
#define PENTAGO_NETWORK_HPP

#include <torch/torch.h>
#include <torch/script.h>

#include "Network.hpp"

namespace SPRL {

class PentagoNetwork : public SPRL::Network<36, 288> {
public:
    PentagoNetwork(std::string path);

    /**
     * Implementation of evaluate for Connect Four, including
     * the proper embedding of the game state.
    */
    std::vector<std::pair<SPRL::GameActionDist<288>, SPRL::Value>> evaluate(
        SPRL::Game<36, 288>* game,
        const std::vector<SPRL::GameState<36>>& states) override;

    int getNumEvals() override {
        return m_numEvals;
    }    

private:
    int m_numEvals { 0 };

    torch::Device m_device { torch::kCPU };
    std::shared_ptr<torch::jit::script::Module> m_model;
};

} // namespace SPRL

#endif
