#include "selfplay/SelfplayOptions.hpp"
#include "uct/UCTOptions.hpp"

int main() {
    SPRL::TreeOptions treeOptions;
    SPRL::UCTOptionsParser uctOptionsParser;

    uctOptionsParser.parse("./config_uct.json", treeOptions);
    std::cout << uctOptionsParser.toString(treeOptions) << std::endl;

    SPRL::WorkerOptions workerOptions;
    SPRL::SelfPlayOptionsParser selfPlayOptionsParser;

    selfPlayOptionsParser.parse("./config_selfplay.json", workerOptions);
    std::cout << selfPlayOptionsParser.toString(workerOptions) << std::endl;
    
    return 0;   
}