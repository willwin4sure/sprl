#ifndef RANDOM_NETWORK_HPP
#define RANDOM_NETWORK_HPP

#include "Network.hpp"

namespace SPRL {

template <int BOARD_SIZE, int ACTION_SIZE> 
class RandomNetwork : public SPRL::Network<BOARD_SIZE, ACTION_SIZE> {
public:
    RandomNetwork() {}

    std::vector<std::pair<SPRL::GameActionDist<ACTION_SIZE>, SPRL::Value>> evaluate(
        SPRL::Game<BOARD_SIZE, ACTION_SIZE>* game,
        const std::vector<SPRL::GameState<BOARD_SIZE>>& states) override {

        m_numEvals += states.size();

        // Return a uniform distribution and a value of 0 for everything
        std::vector<std::pair<SPRL::GameActionDist<ACTION_SIZE>, SPRL::Value>> results;
        results.reserve(states.size());

        SPRL::GameActionDist<ACTION_SIZE> uniformDist;
        for (int i = 0; i < ACTION_SIZE; ++i) {
            uniformDist[i] = 1.0f / ACTION_SIZE;
        }

        for (const SPRL::GameState<BOARD_SIZE>& state : states) {
            results.push_back({ uniformDist, 0.0f });
        }

        return results;
    }

    int getNumEvals() override {
        return m_numEvals;
    }

private:
    int m_numEvals { 0 };
};

} // namespace SPRL

#endif
