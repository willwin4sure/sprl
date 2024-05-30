#ifndef SPRL_GAME_ACTION_DIST_HPP
#define SPRL_GAME_ACTION_DIST_HPP

#include <array>
#include <cmath>

namespace SPRL {

/**
 * Represents any array of floats with length `AS`,
 * e.g. could correspond to a probability distribution over actions.
 * 
 * @tparam AS The size of the action space.
 * 
 * @note Equipped with operations to manipulate
 * action distributions with pointwise operations.
*/
template <int AS>
class GameActionDist {
public:
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
        return AS;
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

        for (int i = 0; i < AS; ++i) {
            result += m_data[i];
        }

        return result;
    }

    /**
     * @returns The action distribution exponentiated, element-wise.
    */
    GameActionDist<AS> exp() const {
        GameActionDist<AS> result {};

        for (int i = 0; i < AS; ++i) {
            result[i] = std::exp(m_data[i]);
        }

        return result;
    }

    /**
     * @returns The action distribution exponentiated by the constant, element-wise.
    */
    GameActionDist<AS> pow(float rhs) const {
        GameActionDist<AS> result {};

        for (int i = 0; i < AS; ++i) {
            result[i] = std::pow(m_data[i], rhs);
        }

        return result;
    }

private:
    std::array<float, AS> m_data {};
};


/**
 * @tparam AS The size of the action space.
 * 
 * @param lhs The left-hand side action distribution.
 * @param rhs The right-hand side action distribution.
 * 
 * @returns The sum of the two action distributions element-wise.
*/
template <int AS>
GameActionDist<AS> operator+(const GameActionDist<AS>& lhs, const GameActionDist<AS>& rhs) {
    GameActionDist<AS> result {};

    for (int i = 0; i < AS; ++i) {
        result[i] = lhs[i] + rhs[i];
    }

    return result;
}

/**
 * @tparam AS The size of the action space.
 * 
 * @param lhs The left-hand side action distribution.
 * @param rhs The right-hand side action distribution.
 * 
 * @returns The difference of the two action distributions element-wise.
*/
template <int AS>
GameActionDist<AS> operator-(const GameActionDist<AS>& lhs, const GameActionDist<AS>& rhs) {
    GameActionDist<AS> result {};

    for (int i = 0; i < AS; ++i) {
        result[i] = lhs[i] - rhs[i];
    }

    return result;
}

/**
 * @tparam AS The size of the action space.
 * 
 * @param lhs The left-hand side action distribution.
 * @param rhs The right-hand side action distribution.
 * 
 * @returns The product of the two action distributions element-wise.
*/
template <int AS>
GameActionDist<AS> operator*(const GameActionDist<AS>& lhs, const GameActionDist<AS>& rhs) {
    GameActionDist<AS> result {};

    for (int i = 0; i < AS; ++i) {
        result[i] = lhs[i] * rhs[i];
    }

    return result;
}

/**
 * @tparam AS The size of the action space.
 * 
 * @param lhs The left-hand side action distribution.
 * @param rhs The right-hand side action distribution.
 * 
 * @returns The quotient of the two action distributions element-wise.
*/
template <int AS>
GameActionDist<AS> operator/(const GameActionDist<AS>& lhs, const GameActionDist<AS>& rhs) {
    GameActionDist<AS> result {};

    for (int i = 0; i < AS; ++i) {
        result[i] = lhs[i] / rhs[i];
    }

    return result;
}

/**
 * @tparam AS The size of the action space.
 * 
 * @param lhs The action distribution.
 * @param rhs The constant to add.
 * 
 * @returns The sum of the action distribution and the constant.
*/
template <int AS>
GameActionDist<AS> operator+(const GameActionDist<AS>& lhs, float rhs) {
    GameActionDist<AS> result {};

    for (int i = 0; i < AS; ++i) {
        result[i] = lhs[i] + rhs;
    }

    return result;
}

/**
 * @tparam AS The size of the action space.
 * 
 * @param lhs The action distribution.
 * @param rhs The constant to subtract.
 * 
 * @returns The difference of the action distribution and the constant.
*/
template <int AS>
GameActionDist<AS> operator-(const GameActionDist<AS>& lhs, float rhs) {
    GameActionDist<AS> result {};

    for (int i = 0; i < AS; ++i) {
        result[i] = lhs[i] - rhs;
    }

    return result;
}

/**
 * @tparam AS The size of the action space.
 * 
 * @param lhs The action distribution.
 * @param rhs The constant to multiply by.
 * 
 * @returns The product of the action distribution and the constant.
*/
template <int AS>
GameActionDist<AS> operator*(const GameActionDist<AS>& lhs, float rhs) {
    GameActionDist<AS> result {};

    for (int i = 0; i < AS; ++i) {
        result[i] = lhs[i] * rhs;
    }

    return result;
}

/**
 * @tparam AS The size of the action space.
 * 
 * @param lhs The action distribution.
 * @param rhs The constant to divide by.
 * 
 * @returns The quotient of the action distribution and the constant.
*/
template <int AS>
GameActionDist<AS> operator/(const GameActionDist<AS>& lhs, float rhs) {
    GameActionDist<AS> result {};

    for (int i = 0; i < AS; ++i) {
        result[i] = lhs[i] / rhs;
    }

    return result;
}

} // namespace SPRL

#endif