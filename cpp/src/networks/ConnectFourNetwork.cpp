#include "ConnectFourNetwork.hpp"

namespace SPRL {

ConnectFourNetwork::ConnectFourNetwork(std::string path) {
    try {
        auto model = std::make_shared<torch::jit::Module>(torch::jit::load(path));
        model->to(m_device);
        m_model = model;

        std::cout << "Connect Four Network loaded successfully." << std::endl;

    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
    }
}

std::vector<std::pair<SPRL::GameActionDist<7>, SPRL::Value>> ConnectFourNetwork::evaluate(
    SPRL::Game<42, 7>* game,
    const std::vector<SPRL::GameState<42>>& states) {
    
    torch::NoGradGuard no_grad;
    m_model->eval();

    int numStates = states.size();

    m_numEvals += numStates;

    auto input = torch::zeros({numStates, 2, 6, 7}).to(m_device);

    for (int b = 0; b < numStates; ++b) {
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 7; ++j) {
                if (states[b].getBoard()[i * 7 + j] == states[b].getPlayer()) {
                    input[b][0][i][j] = 1.0f;
                } else if (states[b].getBoard()[i * 7 + j] == 1 - states[b].getPlayer()) {
                    input[b][1][i][j] = 1.0f;
                }
            }
        }
    }

    auto output = m_model->forward({input}).toTuple();

    auto policyOutput = output->elements()[0].toTensor();
    auto valueOutput = output->elements()[1].toTensor();

    std::vector<std::pair<SPRL::GameActionDist<7>, SPRL::Value>> results;
    results.reserve(numStates);

    for (int b = 0; b < numStates; ++b) {
        std::array<float, 7> policy;
        for (int i = 0; i < 7; ++i) {
            policy[i] = policyOutput[b][i].item<float>();
        }

        // Exponentiate the policy
        for (int i = 0; i < 7; ++i) {
            policy[i] = std::exp(policy[i]);
        }

        // Mask out illegal actions
        auto actionMask = game->actionMask(states[b]);
        for (int i = 0; i < 7; ++i) {
            if (actionMask[i] == 0.0f) {
                policy[i] = 0.0f;
            }
        }

        float sum = 0.0f;
        for (int i = 0; i < 7; ++i) {
            sum += policy[i];
        }
        float norm = 1.0f / sum;

        if (sum == 0.0f) {
            for (int i = 0; i < 7; ++i) {
                policy[i] = 1.0f / 7.0f;
            }
        } else {
            for (int i = 0; i < 7; ++i) {
                policy[i] = policy[i] * norm;
            }
        }

        results.push_back({ policy, valueOutput[b].item<float>() });
    }

    return results;
}

} // namespace SPRL