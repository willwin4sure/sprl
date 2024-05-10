#include "OthelloNetwork.hpp"

namespace SPRL {

OthelloNetwork::OthelloNetwork(std::string path) {
    if (path == "random") {
        // Requested random network instead, not going to load anything.
        return;
    }

    try {
        auto model = std::make_shared<torch::jit::Module>(torch::jit::load(path));
        model->to(m_device);
        m_model = model;

    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
    }
}

std::vector<std::pair<GameActionDist<65>, Value>> OthelloNetwork::evaluate(
    const std::vector<GameState<64>>& states,
    const std::vector<GameActionDist<65>>& masks) {
    
    torch::NoGradGuard no_grad;
    m_model->eval();

    int numStates = states.size();

    m_numEvals += numStates;

    auto input = torch::zeros({numStates, 3, 8, 8}).to(m_device);

    for (int b = 0; b < numStates; ++b) {
        // Stone bitmask channels for current player and opponent player
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                if (states[b].getBoard()[i * 8 + j] == states[b].getPlayer()) {
                    input[b][0][i][j] = 1.0f;
                } else if (states[b].getBoard()[i * 8 + j] == 1 - states[b].getPlayer()) {
                    input[b][1][i][j] = 1.0f;
                }
            }
        }

        // Color channel for which player you are
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                input[b][2][i][j] = ((states[b].getPlayer() == 0) ? 1.0f : 0.0f);
            }
        }
    }

    auto output = m_model->forward({ input }).toTuple();

    auto policyOutput = output->elements()[0].toTensor();
    auto valueOutput = output->elements()[1].toTensor();

    std::vector<std::pair<GameActionDist<65>, Value>> results;
    results.reserve(numStates);

    for (int b = 0; b < numStates; ++b) {
        std::array<float, 65> policy;
        for (int i = 0; i < 65; ++i) {
            policy[i] = policyOutput[b][i].item<float>();
        }

        // Exponentiate the policy
        for (int i = 0; i < 65; ++i) {
            policy[i] = std::exp(policy[i]);
        }

        // Mask out illegal actions
        int numLegal = 0;
        for (int i = 0; i < 65; ++i) {
            if (masks[b][i] == 0.0f) {
                policy[i] = 0.0f;
            } else {
                ++numLegal;
            }
        }

        // Compute the new sum
        float sum = 0.0f;
        for (int i = 0; i < 65; ++i) {
            sum += policy[i];
        }

        if (sum == 0.0f) {
            // If somehow the sum is zero, uniform over legal actions
            float uniform = 1.0f / numLegal;
            for (int i = 0; i < 65; ++i) {
                if (masks[b][i] == 1.0f) {
                    policy[i] = uniform;
                } else {
                    policy[i] = 0.0f;
                }
            }

        } else {
            // Otherwise, normalize the policy
            float norm = 1.0f / sum;
            for (int i = 0; i < 65; ++i) {
                policy[i] = policy[i] * norm;
            }
        }

        // Append the policy and value to the results
        results.push_back({ policy, valueOutput[b].item<float>() });
    }

    return results;
}

} // namespace SPRL
