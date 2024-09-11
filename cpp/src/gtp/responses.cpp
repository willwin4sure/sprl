#include "responses.hpp"

namespace gtp {

std::string Response::getResponseString() const {
    std::string response;
    response += (m_is_error ? "?" : "=");
    if (m_id != NO_COMMAND_ID) {
        response += std::to_string(m_id);
    }
    response += " ";
    response += m_result_or_error_msg;
    return response;
}

} // namespace gtp