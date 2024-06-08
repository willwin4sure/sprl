#include "ConnectFourNetwork.hpp"

namespace SPRL {

ConnectFourNetwork::ConnectFourNetwork(std::string path) {
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

std::vector<std::pair<ConnectFourNetwork::ActionDist, Value>> ConnectFourNetwork::evaluate(
    const std::vector<State>& states,
    const std::vector<ActionDist>& masks) {
    
    torch::NoGradGuard no_grad;
    m_model->eval();

    int numStates = states.size();

    m_numEvals += numStates;

    auto input = torch::zeros({ numStates, 3, C4_NUM_ROWS, C4_NUM_COLS }).to(m_device);

    for (int b = 0; b < numStates; ++b) {
        Player player = states[b].getPlayer();
        Piece piece = pieceFromPlayer(player);

        GridBoard<C4_BOARD_SIZE> board = states[b].getHistory()[0];

        // Stone bitmask channels for current player and opponent player
        for (int i = 0; i < C4_NUM_ROWS; ++i) {
            for (int j = 0; j < C4_NUM_COLS; ++j) {
                if (board[ConnectFourNode::toIndex(i, j)] == piece) {
                    input[b][0][i][j] = 1.0f;

                } else if (board[ConnectFourNode::toIndex(i, j)] == otherPiece(piece)) {
                    input[b][1][i][j] = 1.0f;
                }
            }
        }

        // Color channel for which player you are
        for (int i = 0; i < C4_NUM_ROWS; ++i) {
            for (int j = 0; j < C4_NUM_COLS; ++j) {
                input[b][2][i][j] = ((player == Player::ZERO) ? 1.0f : 0.0f);
            }
        }
    }

    auto output = m_model->forward({ input }).toTuple();

    auto policyOutput = output->elements()[0].toTensor();
    auto valueOutput = output->elements()[1].toTensor();

    std::vector<std::pair<ActionDist, Value>> results;
    results.reserve(numStates);

    for (int b = 0; b < numStates; ++b) {
        ActionDist policy;
        for (int i = 0; i < C4_NUM_COLS; ++i) {
            policy[i] = policyOutput[b][i].item<float>();
        }

        // Network returns logits, we need to exponentiate them
        policy = policy.exp();

        // Mask out illegal actions
        int numLegal = 0;
        for (int i = 0; i < C4_NUM_COLS; ++i) {
            if (masks[b][i] == 0.0f) {
                policy[i] = 0.0f;
            } else {
                ++numLegal;
            }
        }

        // Compute the new sum
        float sum = policy.sum();

        if (sum == 0.0f) {
            // If somehow the sum is zero, uniform over legal actions
            float uniform = 1.0f / numLegal;
            for (int i = 0; i < C4_NUM_COLS; ++i) {
                if (masks[b][i] == 0.0f) {
                    policy[i] = 0.0f;
                } else {
                    policy[i] = uniform;
                }
            }

        } else {
            // Otherwise normalize the policy
            float norm = 1.0f / sum;
            policy = policy * norm;
        }

        // Append the policy and value to the results
        results.push_back({ policy, valueOutput[b].item<float>() });
    }

    return results;
}

} // namespace SPRL
