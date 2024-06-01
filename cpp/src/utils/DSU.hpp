#ifndef SPRL_DSU_HPP
#define SPRL_DSU_HPP

/**
 * @file DSU.hpp
 * 
 * Provides an efficient Union-Find data structure, described here:
 * https://en.wikipedia.org/wiki/Disjoint-set_data_structure.
*/

#include <array>
#include <type_traits>

namespace SPRL {

/**
 * Disjoint set union with path compression.
 * 
 * @tparam T The type of the indices in the DSU, e.g. `uint8_t`, etc.
 *           Should be able to hold the range `[0, N]`.
 * 
 * @tparam N The number of elements in the DSU.
*/
template <typename T, int N>
class DSU {
public:
    static_assert(N > 0, "DSU size must be greater than 0.");
    static_assert(std::is_integral_v<T>, "DSU index type must be integral.");

    /**
     * Constructs a DSU object and initializes the parent and size arrays.
    */
    DSU() {
        for (T i = 0; i < static_cast<T>(N); ++i) {
            m_parent[i] = i;
        }
    }

    /**
     * @returns The representative of the set containing `x`.
    */
    T find(T x) const {
        if (m_parent[x] != x) {
            // Path compression
            m_parent[x] = find(m_parent[x]);
        }
        return m_parent[x];
    }

    /**
     * Unites the sets containing `x` into the set containing `y`.
    */
    void unite(T x, T y) {
        T root_x = find(x);
        T root_y = find(y);

        if (root_x == root_y) return;

        m_parent[root_x] = root_y;
    }

    /**
     * @returns Whether `x` and `y` are in the same set.
    */
    bool sameSet(T x, T y) const {
        return find(x) == find(y);
    }

private:
    mutable std::array<T, N> m_parent;  // The parent of each element.
};

} // namespace SPRL

#endif