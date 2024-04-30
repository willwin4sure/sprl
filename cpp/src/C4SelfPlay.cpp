#include "games/GameState.hpp"
#include "games/Game.hpp"
#include "games/ConnectFour.hpp"

#include "interface/npy.hpp"

#include "networks/Network.hpp"
#include "networks/RandomNetwork.hpp"
#include "networks/ConnectFourNetwork.hpp"

#include "selfplay/SelfPlay.hpp"

#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
    if (argc != 7) {
        std::cerr << "Usage: ./C4SelfPlay.exe <model_path> <save_path> <numGames> <numIters> <maxTraversals> <maxQueueSize>" << std::endl;
        return 1;
    }

    std::string modelPath { argv[1] };
    std::string savePath { argv[2] };
    int numGames = std::stoi(argv[3]);
    int numIters = std::stoi(argv[4]);
    int maxTraversals = std::stoi(argv[5]);
    int maxQueueSize = std::stoi(argv[6]);

    SPRL::ConnectFour game;

    SPRL::Network<42, 7>* network;

    SPRL::RandomNetwork<42, 7> randomNetwork {};
    SPRL::ConnectFourNetwork neuralNetwork { modelPath };

    if (modelPath == "random") {
        // std::cout << "Using random network..." << std::endl;
        network = &randomNetwork;
    } else {
        // std::cout << "Using traced PyTorch network..." << std::endl;
        network = &neuralNetwork;
    }

    auto [states, distributions, outcomes] = SPRL::runIteration(&game, network, numGames, numIters, maxTraversals, maxQueueSize);

    std::vector<float> embeddedStates;
    for (const SPRL::GameState<42>& state : states) {
        int player = state.getPlayer();

        for (int row = 0; row < 6; ++row) {
            for (int col = 0; col < 7; ++col) {
                if (state.getBoard()[row * 7 + col] == player) {
                    embeddedStates.push_back(1.0f);
                } else {
                    embeddedStates.push_back(0.0f);
                }
            }
        }

        for (int row = 0; row < 6; ++row) {
            for (int col = 0; col < 7; ++col) {
                if (state.getBoard()[row * 7 + col] == (1 - player)) {
                    embeddedStates.push_back(1.0f);
                } else {
                    embeddedStates.push_back(0.0f);
                }
            }
        }

        for (int row = 0; row < 6; ++row) {
            for (int col = 0; col < 7; ++col) {
                embeddedStates.push_back((state.getPlayer() == 0) ? 1.0f : 0.0f);
            }
        }
    }

    npy::npy_data_ptr<float> stateData {};
    stateData.data_ptr = embeddedStates.data();
    stateData.shape = { static_cast<unsigned long>(states.size()), 3, 6, 7 };

    npy::write_npy(savePath + "_states.npy", stateData);

    std::vector<float> embeddedDistributions;
    for (const SPRL::GameActionDist<7>& dist : distributions) {
        for (int i = 0; i < 7; ++i) {
            embeddedDistributions.push_back(dist[i]);
        }
    }

    npy::npy_data_ptr<float> distData {};
    distData.data_ptr = embeddedDistributions.data();
    distData.shape = { static_cast<unsigned long>(distributions.size()), 7 };

    npy::write_npy(savePath + "_distributions.npy", distData);

    npy::npy_data_ptr<float> outcomeData {};
    outcomeData.data_ptr = outcomes.data();
    outcomeData.shape = { static_cast<unsigned long>(outcomes.size()) };

    npy::write_npy(savePath + "_outcomes.npy", outcomeData);

    return 0;
}   
