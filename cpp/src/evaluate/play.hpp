#ifndef SPRL_PLAY_HPP
#define SPRL_PLAY_HPP

#include "../agents/IAgent.hpp"
#include "../games/GameNode.hpp"

#include "../symmetry/D4GridSymmetrizer.hpp"
#include "../utils/timer.hpp"

namespace SPRL {

/**
 * Plays a game between two agents and returns the winner.
 * 
 * @tparam ImplNode The implementation of the game node, e.g. `GoNode`.
 * @tparam State The state of the game, e.g. `GridState`.
 * @tparam ACTION_SIZE The number of possible actions in the game.
 * 
 * @param rootNode The root node of the game to start playing from.
 * @param agents The pair of agents that will play the game.
 * @param verbose Whether to print debug information.
*/
template <typename ImplNode, typename State, int ACTION_SIZE>
Player playGame(GameNode<ImplNode, State, ACTION_SIZE>* rootNode,
                std::array<IAgent<ImplNode, State, ACTION_SIZE>*, 2> agents,
                bool verbose = false) {

    using ActionDist = SPRL::GameActionDist<ACTION_SIZE>;

    float totalTime = 0.0f;
    Timer t {};

    GameNode<ImplNode, State, ACTION_SIZE>* curNode = rootNode;

    while (!curNode->isTerminal()) {
        if (verbose) std::cout << curNode->toString() << '\n';

        ActionIdx action;

        t.reset();

        int playerIdx = static_cast<int>(curNode->getPlayer());

        action = agents[playerIdx]->act(curNode, verbose);
        totalTime += t.elapsed();

        agents[1 - playerIdx]->opponentAct(action);

        if (verbose) {
            std::cout << "Player " << playerIdx << " chose action " << action << '\n';
            std::cout << "Time taken: " << t.elapsed() << "s\n";
        }

        curNode = curNode->getAddChild(action);
    }

    Player winner = curNode->getWinner();

    if (verbose) {
        std::cout << "Game over!\n";
        std::cout << curNode->toString() << '\n';

        std::cout << "The winner is Player " << static_cast<int>(winner) << '\n';
        std::cout << "The rewards are " << curNode->getRewards()[0] << " and " << curNode->getRewards()[1] << '\n';
        std::cout << "Total time taken: " << totalTime << "s\n";
    }

    return winner;
}

} // namespace SPRL

#endif
