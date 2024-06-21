#ifndef SPRL_SGF_HPP
#define SPRL_SGF_HPP

/**
 * @file SGF.hpp
 * 
 * Functionality for the SGF (Smart Game Format) file format.
*/

#include "../constants.hpp"
#include "../games/GoNode.hpp"

#include <array>
#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>


namespace SPRL {
/**
 * Utility class for reading and writing SGF files.
*/
class SGF {
    static std::string content;
    static int ptr;
    static GoNode leaf; // the current leaf node
public:
    /**
     * @brief Reads an SGF file and returns the sequence of moves.
     * 
     * @param filePath The path to the SGF file.
     * 
     * @returns The final GameNode state after the sequence of moves.
     * This GameNode will likely not be terminal, because the majority of
     * games end with resignation.
    */
    static SPRL::GoNode read(const std::string& filePath) {
        parse(filePath);

        // remove everything before first `(` and after last `)`
        // (These should be the first and last char in the file, but just in case.)
        size_t lptr = content.find_first_of('(');
        size_t rptr = content.find_last_of(')');

        // split the remaining contents by `;`
        std::vector<std::string> moves;
        size_t pos = 0;
        while(consume(";")){
            if(consume("B")){
                consumeUntil("[");
                std::string move = consumeUntil("]");
                moves.push_back(move);
            }else if(consume("W")){
                consumeUntil("[");
                std::string move = consumeUntil("]");
                moves.push_back(move);
            }else{
                ptr++;
            }

        }
    }

    /**
     * @brief Converts a 0 or two-letter move string to a coordinate index.
     * 
     * @param move The two-letter move string.
     * 
     * @returns The coordinate index of the move.
     */
    static int moveStringToCoord(const std::string& move) {
        // if the string has length 0, it is a pass
        if (move.size() == 0) {
            return GO_BOARD_SIZE;
        }
        // if the string has length 2, it is a normal move
        if (move.size() != 2) {
            // raise error
            std::cerr << "Error: invalid move string" << std::endl;
            return -1;
        }
        int row = move[1] - 'a';
        int col = move[0] - 'a';
        if ( row >= GO_BOARD_WIDTH || col >= GO_BOARD_WIDTH) {
            // indicates a pass, encoded as GO_BOARD_SIZE
            return GO_BOARD_SIZE;
        }
        return row * GO_BOARD_WIDTH + col;
    }

    /**
     * @brief Parses an entire sgf file and updates the contents.
     * 
     * @param filePath The path to the SGF file.
    */
    static void parse(const std::string& filePath) {
        std::ifstream file(filePath);   
        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
        file.close();
        ptr = 0;
    }

    static bool consume(const std::string& token) {
        if (content.substr(ptr, token.size()) == token) {
            ptr += token.size();
            return true;
        }
        return false;
    }
    
    static std::string consumeUntil(const std::string& token) {
        std::string result = "";
        while (content.substr(ptr, token.size()) != token) {
            if (ptr >= content.size()) {
                // raise error
                std::cerr << "Error: token not found" << std::endl;
                return "";

            }
            result += content[ptr];
            ptr++;
        }
        return result;
    }
};

} // namespace SPRL

#endif