#ifndef SPRL_GAME_ACTION_DIST_HPP
#define SPRL_GAME_ACTION_DIST_HPP

#include <array>
#include <cmath>

namespace SPRL {

/**
 * Represents any array of floats with length `ACTION_SIZE`,
 * e.g. could correspond to a probability distribution over actions.
 * 
 * @tparam ACTION_SIZE The size of the action space.
 * 
 * @note Equipped with operations to manipulate
 * action distributions with pointwise operations.
*/
template <int ACTION_SIZE>
class GameActionDist {
public:
    /**
     * Constructs a new action distribution with all zeros.
    */
    GameActionDist() {
        m_data.fill(0.0f);
    }

    /**
     * @returns The value at the given index.
    */
    float operator[](int idx) const {
        return m_data[idx];
    }

    /**
     * @returns A reference to the value at the given index.
    */
    float& operator[](int idx) {
        return m_data[idx];
    }

    /**
     * Fills the action distribution with the given value.
    */
    void fill(float value) {
        m_data.fill(value);
    }

    /**
     * @returns The size of the action distribution.
    */
    int size() const {
        return ACTION_SIZE;
    }

    /**
     * @returns An iterator to the beginning of the action distribution.
    */
    auto begin() {
        return m_data.begin();
    }

    /**
     * @returns A const iterator to the beginning of the action distribution.
    */
    auto begin() const {
        return m_data.begin();
    }

    /**
     * @returns An iterator to the end of the action distribution.
    */
    auto end() {
        return m_data.end();
    }

    /**
     * @returns A const iterator to the end of the action distribution.
    */
    auto end() const {
        return m_data.end();
    }

    /**
     * @returns The sum of the action distribution.
    */
    float sum() const {
        float result = 0.0;

        for (int i = 0; i < ACTION_SIZE; ++i) {
            result += m_data[i];
        }

        return result;
    }

    /**
     * @returns The action distribution exponentiated, element-wise.
    */
    GameActionDist<ACTION_SIZE> exp() const {
        GameActionDist<ACTION_SIZE> result {};

        for (int i = 0; i < ACTION_SIZE; ++i) {
            result[i] = std::exp(m_data[i]);
        }

        return result;
    }

    /**
     * @returns The action distribution exponentiated by the constant, element-wise.
    */
    GameActionDist<ACTION_SIZE> pow(float rhs) const {
        GameActionDist<ACTION_SIZE> result {};

        for (int i = 0; i < ACTION_SIZE; ++i) {
            result[i] = std::pow(m_data[i], rhs);
        }

        return result;
    }

    /**
     * @returns The running cumulative sum of the action distribution.
    */
    GameActionDist<ACTION_SIZE> cumsum() const {
        GameActionDist<ACTION_SIZE> result {};

        result[0] = m_data[0];
        for (int i = 1; i < ACTION_SIZE; ++i) {
            result[i] = result[i - 1] + m_data[i];
        }

        return result;
    }

private:
    std::array<float, ACTION_SIZE> m_data {};
};


/**
 * @tparam ACTION_SIZE The size of the action space.
 * 
 * @param lhs The left-hand side action distribution.
 * @param rhs The right-hand side action distribution.
 * 
 * @returns The sum of the two action distributions element-wise.
*/
template <int ACTION_SIZE>
GameActionDist<ACTION_SIZE> operator+(const GameActionDist<ACTION_SIZE>& lhs, const GameActionDist<ACTION_SIZE>& rhs) {
    GameActionDist<ACTION_SIZE> result {};

    for (int i = 0; i < ACTION_SIZE; ++i) {
        result[i] = lhs[i] + rhs[i];
    }

    return result;
}

/**
 * @tparam ACTION_SIZE The size of the action space.
 * 
 * @param lhs The left-hand side action distribution.
 * @param rhs The right-hand side action distribution.
 * 
 * @returns The difference of the two action distributions element-wise.
*/
template <int ACTION_SIZE>
GameActionDist<ACTION_SIZE> operator-(const GameActionDist<ACTION_SIZE>& lhs, const GameActionDist<ACTION_SIZE>& rhs) {
    GameActionDist<ACTION_SIZE> result {};

    for (int i = 0; i < ACTION_SIZE; ++i) {
        result[i] = lhs[i] - rhs[i];
    }

    return result;
}

/**
 * @tparam ACTION_SIZE The size of the action space.
 * 
 * @param lhs The left-hand side action distribution.
 * @param rhs The right-hand side action distribution.
 * 
 * @returns The product of the two action distributions element-wise.
*/
template <int ACTION_SIZE>
GameActionDist<ACTION_SIZE> operator*(const GameActionDist<ACTION_SIZE>& lhs, const GameActionDist<ACTION_SIZE>& rhs) {
    GameActionDist<ACTION_SIZE> result {};

    for (int i = 0; i < ACTION_SIZE; ++i) {
        result[i] = lhs[i] * rhs[i];
    }

    return result;
}

/**
 * @tparam ACTION_SIZE The size of the action space.
 * 
 * @param lhs The left-hand side action distribution.
 * @param rhs The right-hand side action distribution.
 * 
 * @returns The quotient of the two action distributions element-wise.
*/
template <int ACTION_SIZE>
GameActionDist<ACTION_SIZE> operator/(const GameActionDist<ACTION_SIZE>& lhs, const GameActionDist<ACTION_SIZE>& rhs) {
    GameActionDist<ACTION_SIZE> result {};

    for (int i = 0; i < ACTION_SIZE; ++i) {
        result[i] = lhs[i] / rhs[i];
    }

    return result;
}

/**
 * @tparam ACTION_SIZE The size of the action space.
 * 
 * @param lhs The action distribution.
 * @param rhs The constant to add.
 * 
 * @returns The sum of the action distribution and the constant.
*/
template <int ACTION_SIZE>
GameActionDist<ACTION_SIZE> operator+(const GameActionDist<ACTION_SIZE>& lhs, float rhs) {
    GameActionDist<ACTION_SIZE> result {};

    for (int i = 0; i < ACTION_SIZE; ++i) {
        result[i] = lhs[i] + rhs;
    }

    return result;
}

/**
 * @tparam ACTION_SIZE The size of the action space.
 * 
 * @param lhs The action distribution.
 * @param rhs The constant to subtract.
 * 
 * @returns The difference of the action distribution and the constant.
*/
template <int ACTION_SIZE>
GameActionDist<ACTION_SIZE> operator-(const GameActionDist<ACTION_SIZE>& lhs, float rhs) {
    GameActionDist<ACTION_SIZE> result {};

    for (int i = 0; i < ACTION_SIZE; ++i) {
        result[i] = lhs[i] - rhs;
    }

    return result;
}

/**
 * @tparam ACTION_SIZE The size of the action space.
 * 
 * @param lhs The action distribution.
 * @param rhs The constant to multiply by.
 * 
 * @returns The product of the action distribution and the constant.
*/
template <int ACTION_SIZE>
GameActionDist<ACTION_SIZE> operator*(const GameActionDist<ACTION_SIZE>& lhs, float rhs) {
    GameActionDist<ACTION_SIZE> result {};

    for (int i = 0; i < ACTION_SIZE; ++i) {
        result[i] = lhs[i] * rhs;
    }

    return result;
}

/**
 * @tparam ACTION_SIZE The size of the action space.
 * 
 * @param lhs The action distribution.
 * @param rhs The constant to divide by.
 * 
 * @returns The quotient of the action distribution and the constant.
*/
template <int ACTION_SIZE>
GameActionDist<ACTION_SIZE> operator/(const GameActionDist<ACTION_SIZE>& lhs, float rhs) {
    GameActionDist<ACTION_SIZE> result {};

    float invRhs = 1.0f / rhs;
    for (int i = 0; i < ACTION_SIZE; ++i) {
        result[i] = lhs[i] * invRhs;
    }

    return result;
}

} // namespace SPRL

#endif