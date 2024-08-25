#include "commands.hpp"

namespace gtp {

bool isKnownCommandName(std::string command_name) {
    return std::find(std::begin(KNOWN_COMMAND_NAMES), std::end(KNOWN_COMMAND_NAMES), command_name) != std::end(KNOWN_COMMAND_NAMES);
}

Command::Command(std::string command_line) {
    // Split the string by spaces.
    std::vector<std::string> tokens;
    size_t start = 0;
    size_t end = command_line.find(' ');
    while (end != std::string::npos) {
        tokens.push_back(command_line.substr(start, end - start));
        start = end + 1;
        end = command_line.find(' ', start);
    }
    tokens.push_back(command_line.substr(start));

    // Check if the first token is an integer.
    if (std::all_of(tokens[0].begin(), tokens[0].end(), ::isdigit)) {
        m_id = std::stoi(tokens[0]);
        m_command_name = isKnownCommandName(tokens[1]) ? tokens[1] : UNKNOWN_COMMAND_NAME;
        m_args = std::vector<std::string>(tokens.begin() + 2, tokens.end());

    } else {
        m_id = NO_COMMAND_ID;
        m_command_name = isKnownCommandName(tokens[0]) ? tokens[0] : UNKNOWN_COMMAND_NAME;
        m_args = std::vector<std::string>(tokens.begin() + 1, tokens.end());
    }
}

Command getNextCommand() {
    while (true) {
        std::string input;
        std::getline(std::cin, input);

        // Remove all occurences of CR and other control characters except for HT and LF.
        input.erase(std::remove_if(input.begin(), input.end(), [](char c) {
            return (c <= 0x1F && c != 0x09 && c != 0x0A);
        }), input.end());

        // For each line with a hash sign (#), remove all text following and including this character.
        size_t hashPos = input.find('#');
        if (hashPos != std::string::npos) {
            input.erase(hashPos);
        }

        // Convert all occurences of HT to SPACE.
        std::replace(input.begin(), input.end(), 0x09, 0x20);

        // Discard any empty or white-space only lines.
        if (input.empty() || std::all_of(input.begin(), input.end(), isspace)) {
            continue;
        }

        return Command { input };
    }
}

} // namespace gtp