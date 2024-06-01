#ifndef SPRL_DSU_HPP
#define SPRL_DSU_HPP

/**
 * @file DSU.hpp
 * 
 * Provides an efficient Union-Find data structure, described here:
 * https://en.wikipedia.org/wiki/Disjoint-set_data_structure.
*/

#include <array>

namespace SPRL {

/**
 * Disjoint set union with path compression and union by size.
 * 
 * @tparam N The number of elements in the DSU.
*/
template <int N>
class DSU {
public:
    /**
     * Constructs a DSU object and initializes the parent and size arrays.
    */
    DSU() {
        for (int i = 0; i < N; ++i) {
            m_parent[i] = i;
        }

        m_size.fill(1);
    }

    /**
     * @returns The representative of the set containing `x`.
    */
    int find(int x) const {
        if (m_parent[x] != x) {
            // Path compression
            m_parent[x] = find(m_parent[x]);
        }
        return m_parent[x];
    }

    /**
     * Unites the sets containing `x` and `y`.
    */
    void unite(int x, int y) {
        int root_x = find(x);
        int root_y = find(y);

        if (root_x == root_y) return;

        // Union by size
        if (m_size[root_x] < m_size[root_y]) {
            m_parent[root_x] = root_y;
            m_size[root_y] += m_size[root_x];

        } else {
            m_parent[root_y] = root_x;
            m_size[root_x] += m_size[root_y];
        }
    }

    /**
     * @returns Whether `x` and `y` are in the same set.
    */
    bool sameSet(int x, int y) const {
        return find(x) == find(y);
    }

    /**
     * @returns The size of the set containing `x`.
    */
    int getSize(int x) const {
        return m_size[find(x)];
    }

private:
    mutable std::array<int, N> m_parent;  // The parent of each element.
    std::array<int, N> m_size;  // The size of each set, indexed by representative.
};

} // namespace SPRL

#endif