/*
A Zobrist hash is a 64-bit hash value that is used to represent the state of a game.
It works well when mutations of the game involve only a few commutative
operations at a time. Then, each atomic element of the game state is assigned a random
hash value, and the hash of the game state is the XOR of the hash values of all the
elements. This way, the hash of the game state can be updated efficiently when a
mutation occurs.

For example, in the game of Go, the game state is represented by a board of size 19x19.
Each cell of the board can be empty, black, or white. We can assign a random 64-bit
hash value to each of these three states. (For simplicity, the empty cell is assigned
a value of 0). Then, the hash of the game state is the XOR of the hash values of all
the cells on the board.

Our Zobrist implementation will be a templated class, which takes in a parameter
NUMBER_ATOMIC_ELEMENTS. Each query will be a modification which either adds or
removes an element in the [0, NUMBER_ATOMIC_ELEMENTS) range. The class will have a method
to get the hash value of the current state, and a method to update the hash value when a
modification occurs.

In particular, for determinism, the Zobrist hash values should be a static class variable
(not re-generated for each instance!)
*/

#include <array>
#include <cstdint>
#include <vector>

#include "../constants.hpp"

namespace SPRL {

using ZobristHash = uint64_t;

template <int NUMBER_ATOMIC_ELEMENTS>
class Zobrist {
public:
    // Static Zobrist hash values
    static std::array<ZobristHash, NUMBER_ATOMIC_ELEMENTS> zobrist_values;

    // Initialize the Zobrist hash values
    // We do not use the GetRandom() function from Random, and instead instantiate a new seeded instance of Random, to ensure
    // determinism even if other objects have used GetRandom() before we initialize.
    static void InitializeZobristValues() {
        for (int i = 0; i < NUMBER_ATOMIC_ELEMENTS; i++) {
            zobrist_values[i] = Random(ZOBRIST_SEED, Random::kUniqueStream).UniformUint64(0, std::numeric_limits<uint64_t>::max());
        }
    }

    // Get the Zobrist hash value of a provided state. O(NUMBER_ATOMIC_ELEMENTS), not recommended for performance.
    ZobristHash GetHashValue(const std::vector<int>& state) const {
        ZobristHash hash_value = 0;
        for (int i = 0; i < state.size(); i++) {
            hash_value ^= zobrist_values[i] * state[i];
        }
        return hash_value;
    }
    
    // Modify the Zobrist hash value given the current hash value and the index of the element to modify. O(1).
    ZobristHash ModifyHashValue(ZobristHash hash_value, int index) const {
        return hash_value ^ (zobrist_values[index]);
    }
};

// GetZobrist() function which returns reference to a static Zobrist object
Zobrist<MAX_ZOBRIST>& GetZobrist() {
    static Zobrist<MAX_ZOBRIST> zobrist;
    return zobrist;
}

} // namespace SPRL