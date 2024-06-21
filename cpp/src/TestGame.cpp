#include "agents/HumanAgent.hpp"
#include "agents/HumanGridAgent.hpp"

#include "evaluate/play.hpp"

#include "games/GameNode.hpp"
#include "games/ConnectFourNode.hpp"
#include "games/OthelloNode.hpp"
#include "games/GoNode.hpp"

#include <array>
#include <iostream>

int main(int argc, char* argv[]) {
    SPRL::HumanGridAgent<SPRL::OthelloNode, SPRL::OTH_BOARD_WIDTH, SPRL::OTH_BOARD_WIDTH, SPRL::OTH_HISTORY_SIZE> humanAgent {};
    std::array<SPRL::IAgent<SPRL::OthelloNode, SPRL::GridState<SPRL::OTH_BOARD_SIZE, SPRL::OTH_HISTORY_SIZE>, SPRL::OTH_ACTION_SIZE>*, 2> agents = { &humanAgent, &humanAgent };
    
    SPRL::OthelloNode rootNode {};

    SPRL::playGame(&rootNode, agents, true);

    return 0;
}
