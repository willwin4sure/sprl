// #include "agents/UCTNetworkAgent.hpp"

// #include "games/GameState.hpp"
// #include "games/Game.hpp"
// #include "games/Pentago.hpp"

// #include "networks/PentagoNetwork.hpp"

// #include "uct/UCTTree.hpp"

// #include <iostream>

// int main(int argc, char* argv[]) {
//     if (argc != 6) {
//         std::cerr << "Usage: ./Peek.exe <modelPath> <numTraversals> <maxBatchSize> <maxQueueSize> <boardString>" << std::endl;
//         return 1;
//     }

//     std::string modelPath = argv[1];
//     int numTraversals = std::stoi(argv[2]);
//     int maxBatchSize = std::stoi(argv[3]);
//     int maxQueueSize = std::stoi(argv[4]);
//     std::string boardString = argv[5];

//     SPRL::Pentago game {};

//     SPRL::PentagoNetwork network { modelPath };

//     SPRL::GameState<36> state = game.stringToState(boardString);

//     std::cout << game.stateToString(state) << std::endl;

//     SPRL::UCTTree<36, 288> tree { &game, state, false };

//     SPRL::UCTNetworkAgent<36, 288> networkAgent { &network, &tree, numTraversals, maxBatchSize, maxQueueSize };
    
//     SPRL::ActionIdx action = networkAgent.act(&game, state, game.actionMask(state), true);

//     state = game.nextState(state, action);

//     std::cout << game.stateToString(state) << std::endl;

//     return 0;
// }