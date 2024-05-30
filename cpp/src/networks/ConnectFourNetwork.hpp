#ifndef SPRL_CONNECT_FOUR_NETWORK_HPP
#define SPRL_CONNECT_FOUR_NETWORK_HPP

#include <torch/torch.h>
#include <torch/script.h>

#include "Network.hpp"
#include "../games/ConnectFourNode.hpp"

namespace SPRL {

class ConnectFourNetwork : public Network<GridState<C4_BS>, C4_AS> {
public:
    using State = GridState<C4_BS>;

    ConnectFourNetwork(std::string path);

    /**
     * Implementation of evaluate for Connect Four, including
     * the proper embedding of the game state.
    */
    std::vector<std::pair<ActionDist, Value>> evaluate(
        const std::vector<State>& states,
        const std::vector<ActionDist>& masks) override;

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
