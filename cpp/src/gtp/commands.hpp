#ifndef GTP_COMMANDS_HPP
#define GTP_COMMANDS_HPP

/**
 * @file commands.hpp
 * 
 * Implements the Go Text Protocol commands.
*/

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace gtp {

using CommandID = uint32_t;
constexpr CommandID NO_COMMAND_ID = -1;

// Constant declaration of valid command names.
const std::string KNOWN_COMMAND_NAMES[] = {
    "protocol_version",
    "name",
    "version",
    "known_command",
    "list_commands",
    "quit",
    "boardsize",
    "clear_board",
    "komi",
    "play",
    "genmove",
};

const std::string UNKNOWN_COMMAND_NAME = "unknown_command";

/**
 * Checks if a command name is known.
*/
bool isKnownCommandName(std::string command_name);

struct Command {
    /**
     * Constructs a Command from an incoming command string.
     * 
     * A command is of the form `[int] command_name [arguments]`.
     * Note that `[int]` may be optional, and if it is present, it is the command ID.
     * All following arguments are space-separated.
     */
    Command(std::string command_line);

    CommandID m_id { NO_COMMAND_ID };
    std::string m_command_name;
    std::vector<std::string> m_args;
};

/**
 * Gets the next command from the command stream.
 * 
 * Also handles initial input preprocessing:
 *  1. Remove all occurences of CR and other control characters except for HT and LF.
 *  2. For each line with a hash sign (#), remove all text following and including this character.
 *  3. Convert all occurences of HT to SPACE.
 *  4. Discard any empty or white-space only lines.
 */
Command getNextCommand();

} // namespace gtp

#endif
