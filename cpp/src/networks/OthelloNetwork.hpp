#ifndef OTHELLO_NETWORK_HPP
#define OTHELLO_NETWORK_HPP

#include <torch/torch.h>
#include <torch/script.h>

#include "Network.hpp"

namespace SPRL {

class OthelloNetwork : public Network<64, 65> {
public:
    OthelloNetwork(std::string path);

    /**
     * Implementation of evaluate for Othello, including
     * the proper embedding of the game state.
    */
    std::vector<std::pair<GameActionDist<65>, Value>> evaluate(
        const std::vector<GameState<64>>& states,
        const std::vector<GameActionDist<65>>& masks) override;

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
