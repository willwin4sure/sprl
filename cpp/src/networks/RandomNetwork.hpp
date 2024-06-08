#ifndef SPRL_RANDOM_NETWORK_HPP
#define SPRL_RANDOM_NETWORK_HPP

#include "Network.hpp"

namespace SPRL {

/**
 * A network that returns a uniform distribution and a value of 0 for every state.
 * 
 * @tparam State The state of the game.
 * @tparam ACTION_SIZE The size of the action space.
*/
template <typename State, int ACTION_SIZE>
class RandomNetwork : public Network<State, ACTION_SIZE> {
public:
    using ActionDist = GameActionDist<ACTION_SIZE>;

    RandomNetwork() {}

    std::vector<std::pair<ActionDist, Value>> evaluate(
        const std::vector<State>& states,
        const std::vector<ActionDist>& masks) override {

        int numStates = states.size();

        m_numEvals += numStates;

        // Return a uniform distribution and a value of 0 for everything
        std::vector<std::pair<ActionDist, Value>> results;
        results.reserve(numStates);

        for (int b = 0; b < numStates; ++b) {
            int numLegal = 0;
            for (int i = 0; i < ACTION_SIZE; ++i) {
                if (masks[0][i] == 1.0f) {
                    ++numLegal;
                }
            }

            float uniform = 1.0f / numLegal;

            ActionDist uniformDist;
            for (int i = 0; i < ACTION_SIZE; ++i) {
                if (masks[0][i] == 1.0f) {
                    uniformDist[i] = uniform;
                } else {
                    uniformDist[i] = 0.0f;
                }
            }

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
