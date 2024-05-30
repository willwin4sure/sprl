#include "../src/games/ConnectFourNode.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE( "Handles a basic horizontal victory" ) {
    SPRL::ConnectFourNode rootNode;

    SPRL::GameNode<SPRL::ConnectFourNode::State, SPRL::C4_AS>* curNode = &rootNode;

    std::vector<SPRL::ActionIdx> actions { 3, 3, 4, 4, 2, 3, 1 };

    for (SPRL::ActionIdx action : actions) {
        auto nextNode = curNode->getAddChild(action);

        REQUIRE ( !curNode->isTerminal() );
        REQUIRE ( curNode->getWinner() == SPRL::Player::NONE );

        REQUIRE ( nextNode->getParent() == curNode );
        REQUIRE ( nextNode->getPlayer() == otherPlayer(curNode->getPlayer()) );

        curNode = nextNode;
    }

    REQUIRE( curNode->isTerminal() == true );
    REQUIRE( curNode->getRewards() == std::array<SPRL::Value, 2>{ 1.0f, -1.0f } );
}