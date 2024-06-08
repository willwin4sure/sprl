#include "agents/HumanAgent.hpp"
#include "agents/HumanGoAgent.hpp"

#include "evaluate/play.hpp"

#include "games/GameNode.hpp"
#include "games/ConnectFourNode.hpp"
#include "games/GoNode.hpp"

#include <array>
#include <iostream>

int main(int argc, char* argv[]) {
    SPRL::HumanGoAgent humanAgent {};
    std::array<SPRL::Agent<SPRL::GoNode, SPRL::GridState<SPRL::GO_BOARD_SIZE>, SPRL::GO_ACTION_SIZE>*, 2> agents = { &humanAgent, &humanAgent };
    
    SPRL::GoNode rootNode {};

    SPRL::playGame(&rootNode, agents, true);

    return 0;
}
