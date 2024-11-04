#include "ChessNetwork.hpp"

namespace SPRL {

ChessNetwork::ChessNetwork(std::string path) {
    m_alive = false;
    if (path == "random") {
        // Requested random network instead, not going to load anything.
        m_alive = true;
        return;
    }

    if (torch::cuda::is_available()) {
        std::cerr << "CUDA is available, using GPU." << std::endl;
        m_device = torch::kCUDA;

    } else {
        std::cerr << "CUDA is not available, using CPU." << std::endl;
        m_device = torch::kCPU;
    }
    
    while (!m_alive) {
        try {
            auto model = std::make_shared<torch::jit::Module>(
                torch::jit::load(path));

            model->to(m_device);
            m_model = model;
            m_alive = true;
        } catch (const c10::Error& e) {
            std::cerr << "Error loading the model: " << e.what() << std::endl;
        }
    }
}

static void addBasePlanes(
    std::vector<at::Tensor>& planes, const chess::Board& board) {

    // Add the planes for the pieces.
    chess::Color toMove = board.sideToMove();
    for (chess::Color color : { toMove, ~toMove }) {
        for (auto pieceType : { chess::PieceType::PAWN,
                                chess::PieceType::KNIGHT,
                                chess::PieceType::BISHOP,
                                chess::PieceType::ROOK,
                                chess::PieceType::QUEEN,
                                chess::PieceType::KING }) {

            at::Tensor plane = torch::zeros(
                { CHESS_BOARD_WIDTH, CHESS_BOARD_WIDTH }, torch::kFloat32);

            for (int i = 0; i < CHESS_BOARD_WIDTH; ++i) {
                for (int j = 0; j < CHESS_BOARD_WIDTH; ++j) {
                    // Flip the rank for Black.
                    int rank = (toMove == chess::Color::WHITE) ? CHESS_BOARD_WIDTH - 1 - i : i;
                    int file = j;

                    chess::Piece piece = board.at(
                        chess::Square { chess::Rank { rank }, chess::File { file } });

                    if (piece.color() == color && piece.type() == pieceType) {
                        plane[i][j] = 1.0f;
                    }
                }
            }
            planes.push_back(plane);
        }
    }

    // Add a plane for your color.
    if (toMove == chess::Color::WHITE) {
        planes.push_back(torch::ones(
            { CHESS_BOARD_WIDTH, CHESS_BOARD_WIDTH }, torch::kFloat32));

    } else {
        planes.push_back(torch::zeros(
            { CHESS_BOARD_WIDTH, CHESS_BOARD_WIDTH }, torch::kFloat32));
    }

    // Add planes for castling rights.
    for (chess::Color color : { toMove, ~toMove }) {
        for (auto side : { chess::Board::CastlingRights::Side::KING_SIDE,
                           chess::Board::CastlingRights::Side::QUEEN_SIDE }) {

            if (board.castlingRights().has(color, side)) {
                planes.push_back(torch::ones(
                    { CHESS_BOARD_WIDTH, CHESS_BOARD_WIDTH }, torch::kFloat32));
            } else {
                planes.push_back(torch::zeros(
                    { CHESS_BOARD_WIDTH, CHESS_BOARD_WIDTH }, torch::kFloat32));
            }
        }
    }
}

std::vector<at::Tensor> ChessNetwork::embed(
    const std::vector<ChessNetwork::State>& states,
    EmbeddingFlags flags) {

    std::vector<at::Tensor> embeddedStates;
    embeddedStates.reserve(states.size());

    for (const auto& state : states) {
        chess::Board board = chess::Board::Compact::decode(state.m_history[0]);

        std::vector<at::Tensor> planes;

        // When encoding these, remember to flip the ranks for Black!
        if (flags & ChessEmbeddingFlags::BASE) {
            addBasePlanes(planes, board);
        }

        at::Tensor embeddedState = torch::cat(planes, 0);
    }
}

std::vector<std::pair<ChessNetwork::ActionDist, Value>> ChessNetwork::evaluate(
    const std::vector<ChessNetwork::State>& states,
    const std::vector<ChessNetwork::ActionDist>& masks) {

    torch::NoGradGuard no_grad;
    m_model->eval();

    int numStates = states.size();
    m_numEvals += numStates;

    std::vector<at::Tensor> embeddedStates = embed(states);

    auto input = torch::stack(embeddedStates, 0).to(m_device);
    auto output = m_model->forward({ input }).toTuple();

    auto policyOutput = output->elements()[0].toTensor();  // [B, 73, 8, 8]
    auto valueOutput = output->elements()[1].toTensor();

    std::vector<std::pair<ActionDist, Value>> results;
    results.reserve(numStates);

    for (int b = 0; b < numStates; ++b) {
        ActionDist policy;
        for (int c = 0; c < 73; ++c) {
            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j) {
                    ActionIdx idx = convPolicyMapping[64 * c + 8 * i + j];
                    if (idx != -1) {
                        policy[idx] = policyOutput[b][c][i][j].item<float>();
                    }
                }
            }
        }

        // Policy is returned as logits, so exponentiate.
        policy = policy.exp();

        // Mask out illegal actions
        int numLegal = 0;
        for (int i = 0; i < CHESS_ACTION_SIZE; ++i) {
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
            for (int i = 0; i < CHESS_ACTION_SIZE; ++i) {
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

} // namespace SPRL

