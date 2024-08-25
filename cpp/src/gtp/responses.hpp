#ifndef GTP_RESPONSES_HPP
#define GTP_RESPONSES_HPP

/**
 * @file responses.hpp
 * 
 * Implements the Go Text Protocol responses.
*/

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

namespace gtp {

using ResponseID = uint32_t;
constexpr ResponseID NO_RESPONSE_ID = -1;
constexpr auto UNKNOWN_COMMAND_MSG = "unknown command";

struct Response {
    Response(ResponseID id, std::string result_or_error_msg)
        : m_id { id }, m_result_or_error_msg { result_or_error_msg } {

    }

    /**
     * Gives a response string based on the response.
     * 
     * Handles both successes and failures.
     */
    std::string getResponseString() const {
        std::string response;
        response += (m_is_error ? "?" : "=");
        if (m_id != NO_RESPONSE_ID) {
            response += std::to_string(m_id);
            response += " ";
        }
        response += m_result_or_error_msg;
        return response;
    }

    ResponseID m_id { NO_RESPONSE_ID };
    bool m_is_error { false };
    std::string m_result_or_error_msg;

    // For quit messages, to signal the output thread to stop.
    bool m_shutdown { false };
};

} // namespace gtp

#endif
