#ifndef SPRL_CONNECT_FOUR_HPP
#define SPRL_CONNECT_FOUR_HPP

#include "GameNode.hpp"

namespace SPRL {

constexpr int C4_NUM_ROWS = 6;
constexpr int C4_NUM_COLS = 7;

constexpr int C4_BS = C4_NUM_ROWS * C4_NUM_COLS;
constexpr int C4_AS = C4_NUM_COLS;

/**
 * Implementation of the classic Connect Four game.
 * 
 * See https://en.wikipedia.org/wiki/Connect_Four for details.
*/
class ConnectFourNode : public GameNode<C4_BS, C4_AS> {
public:
    void setStartNode() override;
    std::unique_ptr<GameNode<C4_BS, C4_AS>> getNextNode(ActionIdx action) const override;
    State getGameState() const override;
    std::array<Value, 2> getRewards() const override;

    std::string toString() const override;

private:
};

} // namespace SPRL

#endif