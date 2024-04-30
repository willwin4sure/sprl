#ifndef TRACED_PYTORCH_NETWORK_HPP
#define TRACED_PYTORCH_NETWORK_HPP

#include <torch/torch.h>
#include <torch/script.h>

#include "Network.hpp"

namespace SPRL {

class ConnectFourNetwork : public Network<42, 7> {
public:
    ConnectFourNetwork(std::string path);

    /**
     * Implementation of evaluate for Connect Four, including
     * the proper embedding of the game state.
    */
    std::vector<std::pair<GameActionDist<7>, Value>> evaluate(
        const std::vector<GameState<42>>& states,
        const std::vector<GameActionDist<7>>& masks) override;

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
