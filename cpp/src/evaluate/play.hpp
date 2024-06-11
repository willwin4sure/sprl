#ifndef SPRL_PLAY_HPP
#define SPRL_PLAY_HPP

#include "../agents/Agent.hpp"
#include "../games/GameNode.hpp"

#include "../symmetry/D4GridSymmetrizer.hpp"

#include <chrono>

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

namespace SPRL {

/**
 * Plays a game with two agents.
*/
template <typename ImplNode, typename State, int ACTION_SIZE>
Player playGame(GameNode<ImplNode, State, ACTION_SIZE>* rootNode,
                std::array<Agent<ImplNode, State, ACTION_SIZE>*, 2> agents,
                bool verbose = false) {

    using ActionDist = SPRL::GameActionDist<ACTION_SIZE>;

    float totalTime = 0.0f;
    Timer t {};

    GameNode<ImplNode, State, ACTION_SIZE>* curNode = rootNode;

    while (!curNode->isTerminal()) {
        if (verbose) {
            std::cout << curNode->toString() << '\n';
        }

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
