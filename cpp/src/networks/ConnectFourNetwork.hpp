#ifndef SPRL_CONNECT_FOUR_NETWORK_HPP
#define SPRL_CONNECT_FOUR_NETWORK_HPP

#include <torch/torch.h>
#include <torch/script.h>

#include "Network.hpp"
#include "../games/ConnectFourNode.hpp"

namespace SPRL {

/**
 * A network that evaluates Connect Four game states.
*/
class ConnectFourNetwork : public Network<GridState<C4_BOARD_SIZE, C4_HISTORY_SIZE>, C4_ACTION_SIZE> {
public:
    using State = GridState<C4_BOARD_SIZE, C4_HISTORY_SIZE>;

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
