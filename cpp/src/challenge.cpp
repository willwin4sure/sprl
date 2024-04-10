#include <torch/script.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>

#include "games/GameState.hpp"
#include "games/Game.hpp"
#include "games/ConnectFour.hpp"

#include "uct/UCTNode.hpp"
#include "uct/UCTTree.hpp"


class ConnectFourNetwork : public SPRL::Network<42, 7> {
public:
    ConnectFourNetwork() {
        try {
            std::string path = "./data/models/dragon/traced_dragon_iteration_80.pt";

            auto model = std::make_shared<torch::jit::script::Module>(torch::jit::load(path));
            model->to(m_device);

            std::cout << "Connect Four Network loaded successfully." << std::endl;

            m_model = model;

        } catch (const c10::Error& e) {
            std::cerr << "Error loading the model: " << e.what() << std::endl;
        }
    }

    std::pair<std::array<float, 7>, float> evaluate(SPRL::Game<42, 7>* game, const SPRL::GameState<42>& state) override {
        ++numEvals;

        torch::NoGradGuard no_grad;
        m_model->eval();

        auto input = torch::zeros({1, 2, 6, 7}).to(m_device);

        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 7; ++j) {
                if (state.getBoard()[i * 7 + j] == state.getPlayer()) {
                    input[0][0][i][j] = 1.0f;
                } else if (state.getBoard()[i * 7 + j] == 1 - state.getPlayer()) {
                    input[0][1][i][j] = 1.0f;
                }
            }
        }

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        auto output = m_model->forward(inputs).toTuple();

        auto policy_output = output->elements()[0].toTensor();
        auto value_output = output->elements()[1].toTensor();

        std::array<float, 7> policy;
        for (int i = 0; i < 7; ++i) {
            policy[i] = policy_output[0][i].item<float>();
        }

        // softmax the policy
        for (int i = 0; i < 7; ++i) {
            policy[i] = std::exp(policy[i]);
        }

        // mask out illegal actions
        auto actionMask = game->actionMask(state);
        for (int i = 0; i < 7; ++i) {
            if (actionMask[i] == 0.0f) {
                policy[i] = 0.0f;
            }
        }

        float sum = 0.0f;
        for (int i = 0; i < 7; ++i) {
            sum += policy[i];
        }

        for (int i = 0; i < 7; ++i) {
            policy[i] = policy[i] / sum;
        }

        return { policy, value_output.item<float>() };

        // std::array<float, 7> policy;
        // policy.fill(1.0f / 7.0f);
        // return { policy, 0.0f };
    }

    int numEvals { 0 };

private:
    torch::Device m_device { torch::kCPU };
    std::shared_ptr<torch::jit::script::Module> m_model;
};


template <int ACTION_SIZE>
int getHumanAction(const SPRL::GameActionDist<ACTION_SIZE>& actionSpace) {
    int action = -1;
    while (action < 0 || action >= actionSpace.size() || actionSpace[action] != 1.0f) {
        std::cout << "Enter a valid action: ";
        std::cin >> action;
    }

    return action;
}

// https://www.learncpp.com/cpp-tutorial/timing-your-code/
class Timer {
private:
    using Clock = std::chrono::steady_clock;
    using Second = std::chrono::duration<double, std::ratio<1>>;

    std::chrono::time_point<Clock> m_beg { Clock::now() };

public:
    void reset() {
        m_beg = Clock::now();
    }

    double elapsed() const {
        return std::chrono::duration_cast<Second>(Clock::now() - m_beg).count();
    }
};


template <int BOARD_SIZE, int ACTION_SIZE>
void play(SPRL::Game<BOARD_SIZE, ACTION_SIZE>* game, int numIters) {
    using State = SPRL::GameState<BOARD_SIZE>;
    using ActionDist = SPRL::GameActionDist<ACTION_SIZE>;
    
    ConnectFourNetwork network;

    State state = game->startState();
    SPRL::UCTTree<BOARD_SIZE, ACTION_SIZE> tree { game, state };

    Timer t{};

    float totalTime = 0.0f;

    int moves = 0;
    while (!game->isTerminal(state)) {
        std::cout << game->stateToString(state) << std::endl;

        ActionDist actionMask = game->actionMask(state);
        std::cout << "Action mask: ";
        for (auto& action : actionMask) {
            std::cout << action << ' ';
        }
        std::cout << '\n';

        int action;
        // if (moves % 2 == 1) {
        //     action = getHumanAction(actionMask);
        // } else {
            // SPRL::UCTTree<BOARD_SIZE, ACTION_SIZE> tree { game, state };
            t.reset();
            for (int i = 0; i < numIters; ++i) {
                tree.searchIteration(&network);
            }
            std::cout << "Time taken: " << t.elapsed() << "s" << std::endl;
            totalTime += t.elapsed();
            std::cout << "Total number of evaluations: " << network.numEvals << std::endl;
            
            auto priors = tree.getRoot()->getEdgeStatistics()->m_childPriors;
            auto values = tree.getRoot()->getEdgeStatistics()->m_totalValues;
            auto visits = tree.getRoot()->getEdgeStatistics()->m_numberVisits;

            std::cout << "Priors: ";
            for (auto& prior : priors) {
                std::cout << prior << ' ';
            }

            std::cout << "\nValues: ";
            for (auto& value : values) {
                std::cout << value << ' ';
            }

            std::cout << "\nVisits: ";
            for (auto& visit : visits) {
                std::cout << visit << ' ';
            }

            std::cout << '\n';

            // sample action with most visits
            action = std::distance(visits.begin(), std::max_element(visits.begin(), visits.end()));
        // }

        tree.rerootTree(action);
        state = game->nextState(state, action);

        ++moves;
    }

    std::cout << "Game over!" << '\n';
    std::cout << game->stateToString(state) << std::endl;

    std::cout << "The winner is Player " << static_cast<int>(state.getWinner()) << '\n';
    std::cout << "The rewards are " << game->rewards(state).first << " and " << game->rewards(state).second << '\n';
    std::cout << "Total time taken: " << totalTime << "s" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: ./challenge.exe <numIters>" << std::endl;
        return 1;
    }

    int numIters = std::stoi(argv[1]);

    auto game = std::make_unique<SPRL::ConnectFour>();
    play(game.get(), numIters);

    return 0;
}
