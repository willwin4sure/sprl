#ifndef PLAY_HPP
#define PLAY_HPP

#include "../agents/Agent.hpp"
#include "../games/GameState.hpp"
#include "../games/Game.hpp"

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
 * 
 * @param game The game to play.
 * @param agent0 The first agent.
 * @param agent1 The second agent.
 * @param verbose Whether to print information.
 * @return The winner of the game.
*/
template <int BOARD_SIZE, int ACTION_SIZE>
int playGame(Game<BOARD_SIZE, ACTION_SIZE>* game,
             GameState<BOARD_SIZE> initialState,
             std::array<Agent<BOARD_SIZE, ACTION_SIZE>*, 2> agents,
             bool verbose = false) {

    using State = SPRL::GameState<BOARD_SIZE>;
    using ActionDist = SPRL::GameActionDist<ACTION_SIZE>;

    float totalTime = 0.0f;
    Timer t {};

    GameState<BOARD_SIZE> state = initialState;

    while (!state.isTerminal()) {
        ActionDist actionMask = game->actionMask(state);

        if (verbose) {
            std::cout << game->stateToString(state) << '\n';
            std::cout << "Action mask: ";
            for (int i = 0; i < ACTION_SIZE; ++i) {
                std::cout << actionMask[i] << ' ';
            }
            std::cout << '\n';
        }

        ActionIdx action;

        t.reset();
        action = agents[state.getPlayer()]->act(game, state, actionMask, verbose);
        totalTime += t.elapsed();

        agents[1 - state.getPlayer()]->opponentAct(action);

        if (verbose) {
            std::cout << "Player " << static_cast<int>(state.getPlayer()) << " chose action " << action << '\n';
            std::cout << "Time taken: " << t.elapsed() << "s\n";
        }

        state = game->nextState(state, action);
    }

    if (verbose) {
        std::cout << "Game over!\n";
        std::cout << game->stateToString(state) << '\n';

        std::cout << "The winner is Player " << static_cast<int>(state.getWinner()) << '\n';
        std::cout << "The rewards are " << game->rewards(state).first << " and " << game->rewards(state).second << '\n';
        std::cout << "Total time taken: " << totalTime << "s\n";
    }

    return state.getWinner();
}

} // namespace SPRL

#endif
