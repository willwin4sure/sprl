#ifndef SPRL_GTP_BOT_HPP
#define SPRL_GTP_BOT_HPP

#include "../games/GoNode.hpp"

#include "networks/GridNetwork.hpp"
#include "symmetry/D4GridSymmetrizer.hpp"
#include "uct/UCTTree.hpp"

#include "commands.hpp"
#include "responses.hpp"
#include "ts_deque.hpp"

#include <chrono>
#include <tuple>

namespace SPRL {

// Letter labels Go. Excludes `I`.
const std::string GO_LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ";

class GTPBot {
public:
    using ActionDist = GameActionDist<GO_ACTION_SIZE>;
    using State = GridState<GO_BOARD_SIZE, GO_HISTORY_SIZE>;

    GTPBot(std::string modelPath,
           std::string optionsPath,
           int numTraversals,
           int numPonderTraversals,
           int maxPonderTraversals,
           int maxBatchSize,
           int maxQueueSize)

        : m_gameRunning { false },
          m_network { nullptr },
          m_symmetrizer { nullptr },
          m_tree { nullptr },
          m_numTraversals { numTraversals },
          m_numPonderTraversals { numPonderTraversals },
          m_maxPonderTraversals { maxPonderTraversals },
          m_maxBatchSize { maxBatchSize },
          m_maxQueueSize { maxQueueSize } {
        
        // Setup the network and symmetrizer.
        m_network = std::make_unique<
            GridNetwork<GO_BOARD_WIDTH, GO_BOARD_WIDTH, GO_HISTORY_SIZE, GO_ACTION_SIZE>>(modelPath);
        m_symmetrizer = std::make_unique<D4GridSymmetrizer<GO_BOARD_WIDTH, GO_HISTORY_SIZE>>();

        // Parse the tree options from the path.
        UCTOptionsParser uctParser {};
        uctParser.parse(optionsPath, m_treeOptions);

        m_lastContact = std::chrono::steady_clock::now();
    }

    /**
     * Resets the state of the game.
    */
    void setupGame();

    /**
     * Plays a move on the board.
     * 
     * Returns whether it is legal. If not, the move is not played.
    */
    bool play(std::string color, std::string vertex);

    /**
     * Generates a move for the given color.
     * 
     * Also returns whether the move is legal. If not, no move is returned.
    */
    std::tuple<bool, std::string> genmove(std::string color);

    /**
     * Spends some time pondering, growing the tree.
    */
    void ponder();

    /**
     * Thread function that handles commands and sends responses.
    */
    void botThreadFunc(gtp::ts_deque<gtp::Command>& commandQueue);


private:
    bool m_gameRunning;

    std::unique_ptr<GridNetwork<GO_BOARD_WIDTH, GO_BOARD_WIDTH, GO_HISTORY_SIZE, GO_ACTION_SIZE>> m_network;
    std::unique_ptr<D4GridSymmetrizer<GO_BOARD_WIDTH, GO_HISTORY_SIZE>> m_symmetrizer;
    std::unique_ptr<UCTTree<GoNode, State, GO_ACTION_SIZE>> m_tree;
    TreeOptions m_treeOptions;

    int m_numTraversals;
    int m_numPonderTraversals;
    int m_maxPonderTraversals;
    int m_maxBatchSize;
    int m_maxQueueSize;

    int m_ponderTraversals { 0 };

    std::chrono::time_point<std::chrono::steady_clock> m_lastContact;
    int m_timeout = 3600;  // 60 minutes.
};


} // namespace SPRL

#endif