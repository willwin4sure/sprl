#ifndef SPRL_UCT_OPTIONS_HPP
#define SPRL_UCT_OPTIONS_HPP

#include "utils/struct_mapping/struct_mapping.h"

#include <fstream>
#include <sstream>
#include <string>

namespace sm = struct_mapping;

namespace SPRL {

/**
 * Supported methods for initializing the `Q` values of
 * the UCT nodes.
*/
enum class InitQ {
    ZERO,            // Initialize to zero.
    PARENT_NN_EVAL,  // Initialize to the network output of the parent, if available.
    PARENT_LIVE_Q    // Initialize to the running ("live") `Q` value of the parent.
};

/**
 * Options for the behavior of a UCT node.
*/
struct NodeOptions {
    float dirEps;       // Mixing ratio for Dirichlet noise.
    float dirAlpha;     // Concentration parameter for Dirichlet noise.
    float uWeight;      // Weight for the `U` value against the `Q` value.
    InitQ initQMethod;  // Method for initializing the `Q` value.
    bool takeTrueQAvg;  // Whether to divide the `Q` value by `N` (instead of `N + 1`).
};

/**
 * Options for the behavior of the UCT tree.
*/
struct TreeOptions {
    bool addNoise;         // Whether to add Dirichlet noise.
    bool symmetrizeState;  // Whether to symmetrize the state randomly for network evaluation.

    NodeOptions nodeOptions;
};

/**
 * Parses UCT tree options from a JSON file.
*/
class UCTOptionsParser {
public:
    UCTOptionsParser() {
        // Enumerations are saved as strings, so must be mapped.
        sm::MemberString<InitQ>::set(
            [](const std::string& value) {
                if (value == "ZERO") return InitQ::ZERO;
                if (value == "PARENT_NN_EVAL") return InitQ::PARENT_NN_EVAL;
                if (value == "PARENT_LIVE_Q") return InitQ::PARENT_LIVE_Q;
                throw sm::StructMappingException("Invalid InitQ string: " + value);
            },
            [](InitQ value) {
                switch (value) {
                case InitQ::ZERO: return "ZERO";
                case InitQ::PARENT_NN_EVAL: return "PARENT_NN_EVAL";
                case InitQ::PARENT_LIVE_Q: return "PARENT_LIVE_Q";
                default: throw sm::StructMappingException("Missing InitQ conversion.");
                }
            }
        );

        sm::reg(&NodeOptions::dirEps, "dirEps", sm::Default { 0.25f });
        sm::reg(&NodeOptions::dirAlpha, "dirAlpha", sm::Default { 0.2f });
        sm::reg(&NodeOptions::uWeight, "uWeight", sm::Default { 1.1f });
        sm::reg(&NodeOptions::initQMethod, "initQMethod", sm::Default { InitQ::ZERO });
        sm::reg(&NodeOptions::takeTrueQAvg, "takeTrueQAvg", sm::Default { false });

        sm::reg(&TreeOptions::addNoise, "addNoise", sm::Default { true });
        sm::reg(&TreeOptions::symmetrizeState, "symmetrizeState", sm::Default { true });
        sm::reg(&TreeOptions::nodeOptions, "nodeOptions", sm::Required {});
    }

    /**
     * Parses UCT tree options from a JSON file.
     * 
     * @param path The path to the JSON file to parse.
     * @param options The UCT tree options to populate.
    */
    void parse(const std::string& path, TreeOptions& options) {
        auto stream = std::ifstream(path);
        sm::map_json_to_struct(options, stream);
    }

    /**
     * Converts UCT tree options to a JSON string.
     * 
     * @param options The UCT tree options to convert.
     * @returns The JSON string representation of the UCT tree options.
     * 
     * @note The input parameter cannot be const.
    */
    std::string toString(TreeOptions& options) {
        std::stringstream outJsonData;
        sm::map_struct_to_json(options, outJsonData, "    ");
        return outJsonData.str();
    }
};

} // namespace SPRL

#endif