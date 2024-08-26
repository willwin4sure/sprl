#ifndef GTP_RESPONSES_HPP
#define GTP_RESPONSES_HPP

/**
 * @file responses.hpp
 * 
 * Implements the Go Text Protocol responses.
*/

#include "commands.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

namespace gtp {

struct Response {
    Response() = default;
    Response(CommandID id, bool is_error, std::string result_or_error_msg)
        : m_id { id }, m_is_error { is_error }, m_result_or_error_msg { std::move(result_or_error_msg) } {
        
    }

    /**
     * Gives a response string based on the response.
     * 
     * Handles both successes and failures.
     */
    std::string getResponseString() const;

    CommandID m_id { NO_COMMAND_ID };
    bool m_is_error { false };
    std::string m_result_or_error_msg { "" };
};

} // namespace gtp

#endif