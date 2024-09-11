/**
 * @file GTP_Challenge.cpp
 * 
 * Main file for the GTP Challenge executable.
 * Implements the Go Text Protocol:
 * https://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html.
*/

#include "gtp/gtp_bot.hpp"

#include "gtp/commands.hpp"
#include "gtp/ts_deque.hpp"

#include <algorithm>
#include <iostream>
#include <string>
#include <thread>


/**
 * Thread function that handles inputs.
 * 
 * @param commandQueue The queue to push input onto.
*/
void inputThreadFunc(gtp::ts_deque<gtp::Command>& commandQueue) {
    while (true) {
        gtp::Command command = gtp::getNextCommand();
        bool shutdown = (command.m_command_name == "quit");

        // Push input onto the queue.
        commandQueue.push_back(std::move(command));

        // If `quit` was entered, break out of the loop.
        if (shutdown) break;
    }
}

constexpr int NUM_PONDER_TRAVERSALS = 128;

int main(int argc, char* argv[]) {
    if (argc != 7) {
        std::cerr << "Usage: ./GTP.exe <modelPath> <optionsPath> <numTraversals> <maxPonderTraversals> <maxBatchSize> <maxQueueSize>" << std::endl;
        return 1;
    }

    std::string modelPath = argv[1];
    std::string optionsPath = argv[2];
    int numTraversals = std::stoi(argv[3]);
    int maxPonderTraversals = std::stoi(argv[4]);
    int maxBatchSize = std::stoi(argv[5]);
    int maxQueueSize = std::stoi(argv[6]);

    gtp::ts_deque<gtp::Command> commandQueue;
    std::thread inputThread { inputThreadFunc, std::ref(commandQueue) };

    SPRL::GTPBot bot { modelPath, optionsPath, numTraversals, NUM_PONDER_TRAVERSALS, maxPonderTraversals, maxBatchSize, maxQueueSize };
    std::thread botThread { &SPRL::GTPBot::botThreadFunc, &bot, std::ref(commandQueue) };

    inputThread.join();
    botThread.join();
}