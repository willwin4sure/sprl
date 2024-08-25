/**
 * @file GTP_Challenge.cpp
 * 
 * Main file for the GTP Challenge executable.
 * Implements the Go Text Protocol:
 * https://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html.
*/

#include "gtp/commands.hpp"
#include "gtp/responses.hpp"
#include "gtp/ts_deque.hpp"

#include <algorithm>
#include <iostream>
#include <string>
#include <thread>


/**
 * Thread function that handles inputs.
 * 
 * @param inputQueue The queue to push input onto.
*/
void inputThreadFunc(gtp::ts_deque<gtp::Command>& inputQueue) {
    while (true) {
        gtp::Command command = gtp::getNextCommand();
        bool shutdown = (command.m_command_name == "quit");

        // Push input onto the queue.
        inputQueue.push_back(std::move(command));

        // If `quit` was entered, break out of the loop.
        if (shutdown) break;
    }
}


/**
 * Thread function that handles outputs.
 * 
 * @param outputQueue The queue to pop output from.
*/
void outputThreadFunc(gtp::ts_deque<gtp::Response>& outputQueue) {
    while (true) {
        // Wait until there is output to process.
        outputQueue.wait();

        // Pop output from the queue.
        gtp::Response response = outputQueue.pop_front();

        // If `quit` was entered, break out of the loop.
        if (response.m_shutdown) break;

        // Print the response with two newlines.
        std::cout << response.getResponseString() << std::endl << std::endl;
    }
}


/**
 * Thread function that runs the Go bot.
 * 
 * Each input should produce exactly one output, and be handled in order.
 * 
 * @param inputQueue The queue to pop input from.
 * @param outputQueue The queue to push output onto.
 */
void botThreadFunc(gtp::ts_deque<gtp::Command>& inputQueue, gtp::ts_deque<gtp::Response>& outputQueue) {
    while (true) {
        // Wait until there is input to process.
        inputQueue.wait();

        // Pop input from the queue.
        gtp::Command command = inputQueue.pop_front();
        bool shutdown = (command.m_command_name == "quit");

        // Process the command.
        gtp::Response response { command.m_id, "unknown command" };
        response.m_shutdown = shutdown;
        outputQueue.push_back(std::move(response));

        // If `quit` was entered, break out of the loop. Send a shutdown signal first.
        if (shutdown) break;
    }
}


int main() {
    gtp::ts_deque<gtp::Command> inputQueue;
    gtp::ts_deque<gtp::Response> outputQueue;

    std::thread inputThread(inputThreadFunc, std::ref(inputQueue));
    std::thread outputThread(outputThreadFunc, std::ref(outputQueue));
    std::thread botThread(botThreadFunc, std::ref(inputQueue), std::ref(outputQueue));

    inputThread.join();
    outputThread.join();
    botThread.join();
}