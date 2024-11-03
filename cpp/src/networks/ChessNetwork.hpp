#ifndef SPRL_CHESS_NETWORK_HPP
#define SPRL_CHESS_NETWORK_HPP

#include "games/ChessState.hpp"

#include "INetwork.hpp"

#include <torch/cuda.h>
#include <torch/script.h>
#include <torch/torch.h>

namespace SPRL {

using EmbeddingFlags = uint64_t;

enum ChessEmbeddingFlags : EmbeddingFlags {
    BASE = 1ULL << 0,
    LEGAL_MOVES = 1ULL << 1,
    LEGAL_CAPTURES = 1ULL << 2,
    YOUR_ATTACKING = 1ULL << 3,
    OPP_ATTACKING = 1ULL << 4,
    NUM_DEFENDERS = 1ULL << 5,
    NUM_ATTACKERS = 1ULL << 6,
};

/**
 * Network for chess.
 */
class ChessNetwork : public INetwork<ChessState, CHESS_ACTION_SIZE> {
public:
    using ActionDist = GameActionDist<CHESS_ACTION_SIZE>;
    using State = ChessState;

    /**
     * Constructs a GridNetwork from a model file.
     * 
     * @param path The path to the model file, or "random" to do nothing.
     */
    ChessNetwork(std::string path);

    /**
     * @returns Whether the network is alive and can be used.
     */
    bool isAlive() override { return m_alive; }

    /**
     * Embeds the given states into the network's input format.
     */
    std::vector<at::Tensor> embed(
        const std::vector<State>& states,
        EmbeddingFlags flags = ChessEmbeddingFlags::BASE);
    
    /**
     * Implementation of evaluate for Chess, including
     * the proper embedding of the game state.
     * 
     * @param states The states to evaluate.
     * @param masks The action masks for the states.
     * 
     * @return A vector of (policy, value) pairs for each state.
    */
    std::vector<std::pair<ActionDist, Value>> evaluate(
        const std::vector<State>& states,
        const std::vector<ActionDist>& masks) override;

    /**
     * @returns The number of evaluations made by the network, summed over batches.
    */
    int getNumEvals() override { return m_numEvals; }

private:
    bool m_alive { false };
    int m_numEvals { 0 };

    torch::Device m_device { torch::kCPU };
    std::shared_ptr<torch::jit::script::Module> m_model;
};

} // namespace SPRL

#endif