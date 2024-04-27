#include "PentagoNetwork.hpp"

namespace SPRL {

PentagoNetwork::PentagoNetwork(std::string path) {
    if (path == "random") {
        std::cout << "Random network requested, not loading model." << std::endl;
        return;
    }

    try {
        auto model = std::make_shared<torch::jit::Module>(torch::jit::load(path));
        model->to(m_device);
        m_model = model;

        std::cout << "Pentago Network loaded successfully." << std::endl;

    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
    }
}

std::vector<std::pair<SPRL::GameActionDist<288>, SPRL::Value>> PentagoNetwork::evaluate(
    SPRL::Game<36, 288>* game,
    const std::vector<SPRL::GameState<36>>& states) {
    
    torch::NoGradGuard no_grad;
    m_model->eval();

    int numStates = states.size();

    m_numEvals += numStates;

    auto input = torch::zeros({numStates, 3, 6, 6}).to(m_device);

    for (int b = 0; b < numStates; ++b) {
        // Stone bitmask channels for current player and opponent player
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                if (states[b].getBoard()[i * 6 + j] == states[b].getPlayer()) {
                    input[b][0][i][j] = 1.0f;
                } else if (states[b].getBoard()[i * 6 + j] == 1 - states[b].getPlayer()) {
                    input[b][1][i][j] = 1.0f;
                }
            }
        }

        // Color channel for which player you are
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                input[b][2][i][j] = ((states[b].getPlayer() == 0) ? 1.0f : 0.0f);
            }
        }
    }

    auto output = m_model->forward({ input }).toTuple();

    auto policyOutput = output->elements()[0].toTensor();
    auto valueOutput = output->elements()[1].toTensor();

    std::vector<std::pair<SPRL::GameActionDist<288>, SPRL::Value>> results;
    results.reserve(numStates);

    for (int b = 0; b < numStates; ++b) {
        std::array<float, 288> policy;
        for (int i = 0; i < 288; ++i) {
            policy[i] = policyOutput[b][i].item<float>();
        }

        // Exponentiate the policy
        for (int i = 0; i < 288; ++i) {
            policy[i] = std::exp(policy[i]);
        }

        // Mask out illegal actions
        auto actionMask = game->actionMask(states[b]);
        for (int i = 0; i < 288; ++i) {
            if (actionMask[i] == 0.0f) {
                policy[i] = 0.0f;
            }
        }

        float sum = 0.0f;
        for (int i = 0; i < 288; ++i) {
            sum += policy[i];
        }

        if (sum == 0.0f) {
            float uniform = 1.0f / 288.0f;
            for (int i = 0; i < 288; ++i) {
                policy[i] = uniform;
            }
        } else {
            float norm = 1.0f / sum;
            for (int i = 0; i < 288; ++i) {
                policy[i] = policy[i] * norm;
            }
        }

        results.push_back({ policy, valueOutput[b].item<float>() });
    }

    return results;
}

} // namespace SPRL