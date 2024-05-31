#ifndef SPRL_ZOBRIST_HPP
#define SPRL_ZOBRIST_HPP

/**
 * @file Zobrist.hpp
 * 
 * Functionality for the Zobrist hashing scheme, described here:
 * https://en.wikipedia.org/wiki/Zobrist_hashing.
*/

#include "Random.hpp"
#include "../constants.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace SPRL {

/**
 * @brief A class that generates and holds Zobrist values for atomic elements.
 * 
 * @tparam NUM_ATOMS The number of atomic elements you want to hash.
 * 
 * @note In Go, each pair (Coord, Piece) is an atomic element that is
 * assigned a random 64-bit unsigned integer. The Zobrist hash of a board
 * is the XOR of the Zobrist values of all the atomic elements on it.
 * This is used for efficient positional superko detection:
 * https://en.wikipedia.org/wiki/Rules_of_Go#Ko.
*/
template <int NUM_ATOMS>
class Zobrist {
public:
    /// The type of a Zobrist hash value.
    using Hash = uint64_t;

    /**
     * Constructs a Zobrist object and initializes the hash values.
    */
    Zobrist() {
        for (int i = 0; i < NUM_ATOMS; i++) {
            m_zobrist_values[i] = GetRandom()
                .UniformUint64(0, std::numeric_limits<uint64_t>::max());
        }
    }

    /**
     * @param atomIdx The index of the atomic element to hash.
     * 
     * @returns The Zobrist value of the atomic element at the given index.
    */
    Hash operator[](int atomIdx) const {
        return m_zobrist_values[atomIdx];
    }
    
private:
    /// Zobrist values for each atomic element you want to hash.
    std::array<Hash, NUM_ATOMS> m_zobrist_values;
};

} // namespace SPRL

#endif