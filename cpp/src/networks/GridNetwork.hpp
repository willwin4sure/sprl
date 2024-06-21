#ifndef SPRL_GRID_NETWORK_HPP
#define SPRL_GRID_NETWORK_HPP

#include "games/GridState.hpp"

#include "INetwork.hpp"

#include <torch/torch.h>
#include <torch/script.h>

namespace SPRL {

/**
 * A network that evaluates grid game states using a standard embedding.
 * 
 * The state is embedded into `2 * HISTORY_SIZE + 1` channels, where
 * the first `HISTORY_SIZE` pairs of channels are bitmasks of the
 * current player's stones and the opponent's stones, and the last
 * channel is a color channel for which player you are.
 * 
 * @tparam NUM_ROWS The number of rows in the grid.
 * @tparam NUM_COLS The number of columns in the grid.
 * @tparam HISTORY_SIZE The number of previous states to include in the state.
 * @tparam ACTION_SIZE The number of actions in the action space.
 */
template <int NUM_ROWS, int NUM_COLS, int HISTORY_SIZE, int ACTION_SIZE>
class GridNetwork : public INetwork<GridState<NUM_ROWS * NUM_COLS, HISTORY_SIZE>, ACTION_SIZE> {
public:
    using ActionDist = GameActionDist<ACTION_SIZE>;
    using State = GridState<NUM_ROWS * NUM_COLS, HISTORY_SIZE>;

    /**
     * Constructs a GridNetwork from a model file.
     * 
     * @param path The path to the model file, or "random" to do nothing.
     */
    GridNetwork(std::string path) {
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

    /**
     * Implementation of evaluate for Go, including
     * the proper embedding of the game state.
     * 
     * @param states The states to evaluate.
     * @param masks The action masks for the states.
     * 
     * @return A vector of (policy, value) pairs for each state.
    */
    std::vector<std::pair<ActionDist, Value>> evaluate(
        const std::vector<State>& states,
        const std::vector<ActionDist>& masks) override {

        torch::NoGradGuard no_grad;
        m_model->eval();

        int numStates = states.size();
        m_numEvals += numStates;

        auto input = torch::zeros({numStates, 2 * HISTORY_SIZE + 1, NUM_ROWS, NUM_COLS}).to(m_device);

        for (int b = 0; b < numStates; ++b) {
            Piece ourPiece = pieceFromPlayer(states[b].getPlayer());

            // Stone bitmask channels for current player and opponent player
            for (int t = 0; t < states[b].size(); ++t) {
                for (int i = 0; i < NUM_ROWS; ++i) {
                    for (int j = 0; j < NUM_COLS; ++j) {
                        if (states[b].getHistory()[t][i * NUM_COLS + j] == ourPiece) {
                            input[b][2 * t][i][j] = 1.0f;

                        } else if (states[b].getHistory()[t][i * NUM_COLS + j] == otherPiece(ourPiece)) {
                            input[b][2 * t + 1][i][j] = 1.0f;
                        }
                    }
                }
            }

            // Color channel for which player you are
            for (int i = 0; i < NUM_ROWS; ++i) {
                for (int j = 0; j < NUM_COLS; ++j) {
                    input[b][2 * HISTORY_SIZE][i][j] = ((states[b].getPlayer() == Player::ZERO) ? 1.0f : 0.0f);
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
            for (int i = 0; i < ACTION_SIZE; ++i) {
                policy[i] = policyOutput[b][i].item<float>();
            }

            // Policy is returned as logits, so exponentiate.
            policy = policy.exp();

            // Mask out illegal actions
            int numLegal = 0;
            for (int i = 0; i < ACTION_SIZE; ++i) {
                if (masks[b][i] == 0.0f) {
                    policy[i] = 0.0f;

                } else {
                    ++numLegal;
                }
            }

            float sum = policy.sum();
            if (sum == 0.0f) {
                // If sum is zero, uniform over legal actions.
                float uniform = 1.0f / numLegal;
                for (int i = 0; i < ACTION_SIZE; ++i) {
                    policy[i] = (masks[b][i] == 0.0f) ? 0.0f : uniform;
                }

            } else {
                // Normalize the policy
                policy = policy / sum;
            }

            // Append the policy and value to the results.
            results.emplace_back(policy, valueOutput[b].item<float>());
        }

        return results;
    }

    /**
     * @returns The number of evaluations made by the network, summed over batches.
    */
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