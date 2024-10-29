#include "agents/HumanAgent.hpp"
#include "agents/HumanChessAgent.hpp"
#include "agents/HumanGridAgent.hpp"

#include "evaluate/play.hpp"

#include "games/GameNode.hpp"
#include "games/ConnectFourNode.hpp"
#include "games/ChessNode.hpp"
#include "games/OthelloNode.hpp"
#include "games/GoNode.hpp"

#include <array>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#endif

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    SPRL::HumanChessAgent humanAgent {};

    std::array<
        SPRL::IAgent<
            SPRL::ChessNode,
            SPRL::ChessState,
            SPRL::CHESS_ACTION_SIZE
        >*, 2
    > agents = { &humanAgent, &humanAgent };

    SPRL::ChessNode rootNode {};

    SPRL::playGame(&rootNode, agents, true);

    return 0;
}
