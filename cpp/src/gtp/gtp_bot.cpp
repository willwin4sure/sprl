#include "gtp_bot.hpp"

namespace SPRL {

void GTPBot::setupGame() {
    m_gameRunning = true;
    m_tree = std::make_unique<UCTTree<GoNode, State, GO_ACTION_SIZE>>(m_treeOptions, m_symmetrizer.get());
}

bool GTPBot::play(std::string color, std::string vertex) {
    // The `color` string should be either "black" or "white".
    Player player = color == "black" ? Player::ZERO : Player::ONE;
    if (player != m_tree->getDecisionNode()->getPlayer()) {
        // We only support moves from the current player to act.
        std::cerr << "Wrong color: " << color << std::endl;
        return false;
    }

    ActionIdx action;

    // Vertex has format e.g. `B13`, `j11`, `pass`.
    if (vertex == "pass") {
        action = GO_BOARD_SIZE;  // Pass action.

    } else {
        int row = GO_BOARD_WIDTH - std::stoi(vertex.substr(1));
        int col = GO_LETTERS.find(std::toupper(vertex[0]));

        if (row < 0 || row >= GO_BOARD_WIDTH || col < 0 || col >= GO_BOARD_WIDTH) {
            // The action is out of bounds.
            std::cerr << "Out of bounds move: " << vertex << std::endl;
            return false;
        }

        // Compute the action and verify that it is valid.
        action = row * GO_BOARD_WIDTH + col;
        if (m_tree->getDecisionNode()->getActionMask()[action] == 0.0f) {
            // The action is illegal.
            std::cerr << "Illegal move: " << vertex << std::endl;
            return false;
        }
    }

    // Advance the decision node.
    m_tree->advanceDecision(action, false);  // Don't clear edge statistics.
    m_ponderTraversals = 0;  // Reset pondering limit.
    m_lastPonderPrintout = 0;

    return true;
}

std::tuple<bool, std::string> GTPBot::genmove(std::string color) {
    // The `color` string should be either "black" or "white".
    Player player = color == "black" ? Player::ZERO : Player::ONE;
    if (player != m_tree->getDecisionNode()->getPlayer()) {
        // We only support moves from the current player to act.
        std::cerr << "Wrong color: " << color << std::endl;
        return { false, "" };
    }

    // Perform the traversals.
    int traversals = 0;
    while (traversals < m_numTraversals) {
        // Greedily search and collect leaves, expanding the tree iteratively.
        auto [leaves, trav] = m_tree->searchAndGetLeaves(
            m_maxBatchSize, m_maxQueueSize, false, m_network.get());

        // If we have leaves, evaluate them using the NN and backpropagate.
        if (leaves.size() > 0) {
            m_tree->evaluateAndBackpropLeaves(leaves, m_network.get());
        }

        traversals += trav;
    }

    auto visits = m_tree->getDecisionNode()->getEdgeStatistics()->m_numVisits;
    ActionIdx action = std::distance(visits.begin(), std::max_element(visits.begin(), visits.end()));

    // Advance the decision node.
    m_tree->advanceDecision(action, false);  // Don't clear edge statistics.
    m_ponderTraversals = 0;  // Reset pondering limit.
    m_lastPonderPrintout = 0;

    // Convert the action to a vertex string.
    int row = action / GO_BOARD_WIDTH;
    int col = action % GO_BOARD_WIDTH;
    std::string vertex = GO_LETTERS[col] + std::to_string(GO_BOARD_WIDTH - row);

    return { true, vertex };
}

void GTPBot::ponder() {
    int traversals = 0;
    while (traversals < m_numPonderTraversals && m_ponderTraversals < m_maxPonderTraversals) {
        // Greedily search and collect leaves, expanding the tree iteratively.
        auto [leaves, trav] = m_tree->searchAndGetLeaves(
            m_maxBatchSize, m_maxQueueSize, false, m_network.get());

        // If we have leaves, evaluate them using the NN and backpropagate.
        if (leaves.size() > 0) {
            m_tree->evaluateAndBackpropLeaves(leaves, m_network.get());
        }

        traversals += trav;
        m_ponderTraversals += trav;
    }
}

void GTPBot::botThreadFunc(gtp::ts_deque<gtp::Command>& commandQueue) {
    // Loop until told to quit.
    while (true) {
        // If no game is ongoing, wait for input.
        if (!m_gameRunning || m_ponderTraversals == m_maxPonderTraversals) {
            std::cerr << "No game running or hit pondering limit. Waiting for commands." << std::endl;
            commandQueue.wait();
        }

        bool quit = false;

        // Process a command from the queue.
        if (!commandQueue.empty()) {
            m_lastContact = std::chrono::steady_clock::now();

            gtp::Command command = commandQueue.pop_front();

            std::cerr << "Processing command: " << command.m_command_name << std::endl;
            std::cerr << "Args: ";
            for (const std::string& arg : command.m_args) {
                std::cerr << arg << " ";
            }
            std::cerr << std::endl;

            gtp::Response res;
            
            if (command.m_command_name == gtp::UNKNOWN_COMMAND_NAME) {
                res = gtp::Response { command.m_id, true, "unknown command" };


            } else if (command.m_command_name == "protocol_version") {
                res = gtp::Response { command.m_id, false, "2" };


            } else if (command.m_command_name == "name") {
                res = gtp::Response { command.m_id, false, "SPRL Bot" };


            } else if (command.m_command_name == "version") {
                res = gtp::Response { command.m_id, false, "0.1" };


            } else if (command.m_command_name == "known_command") {
                std::string commandName = command.m_args[0];
                std::string known = gtp::isKnownCommandName(commandName) ? "true" : "false";
                res = gtp::Response { command.m_id, false, known };


            } else if (command.m_command_name == "list_commands") {
                std::string commands;
                for (const std::string& commandName : gtp::KNOWN_COMMAND_NAMES) {
                    commands += commandName + "\n";
                }
                commands.pop_back();  // Remove the trailing newline.
                res = gtp::Response { command.m_id, false, commands };


            } else if (command.m_command_name == "quit") {
                res = gtp::Response { command.m_id, false, "" };
                quit = true;


            } else if (command.m_command_name == "boardsize") {
                int size = std::stoi(command.m_args[0]);
                if (size != GO_BOARD_WIDTH) {
                    res = gtp::Response { command.m_id, true, "unacceptable size" };

                } else {
                    setupGame();
                    res = gtp::Response { command.m_id, false, "" };
                }


            } else if (command.m_command_name == "clear_board") {
                setupGame();
                res = gtp::Response { command.m_id, false, "" };


            } else if (command.m_command_name == "komi") {
                float komi = std::stof(command.m_args[0]);
                if (komi != GO_KOMI) {
                    res = gtp::Response { command.m_id, true, "unacceptable komi" };

                } else {
                    res = gtp::Response { command.m_id, false, "" };
                }


            } else if (command.m_command_name == "play") {
                if (!m_gameRunning) {
                    res = gtp::Response { command.m_id, true, "no game running" };

                } else {
                    std::string color = command.m_args[0];
                    std::string vertex = command.m_args[1];

                    bool legal = play(color, vertex);
                    if (!legal) {
                        res = gtp::Response { command.m_id, true, "illegal move" };

                    } else {
                        res = gtp::Response { command.m_id, false, "" };
                    }
                }


            } else if (command.m_command_name == "genmove") {
                if (!m_gameRunning) {
                    res = gtp::Response { command.m_id, true, "no game running" };

                } else {
                    std::string color = command.m_args[0];
                    auto [legal, move] = genmove(color);
                    if (!legal) {
                        res = gtp::Response { command.m_id, true, "wrong color" };

                    } else {
                        res = gtp::Response { command.m_id, false, move };
                    }
                }
            }

            // Responses end with two newlines.
            std::cout << res.getResponseString() << "\n" << std::endl;

            // Check if we should terminate the main loop.
            if (quit) break;

        } else {
            // Check if it has been too long since last contact.
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - m_lastContact);
            if (elapsed.count() > m_timeout) {
                // If it has been too long, quit the game.
                std::cerr << "Timeout. Quitting game." << std::endl;
                m_gameRunning = false;

            } else {
                // The game is running and no commands to process. Ponder!
                ponder();
                if (m_ponderTraversals - m_lastPonderPrintout >= 1000) {
                    std::cerr << m_ponderTraversals << " ponder traversals." << std::endl;
                    m_lastPonderPrintout = m_ponderTraversals;
                }
            }
        }
    }
}

} // namespace SPRL