#ifndef SPRL_GAME_ACTION_DIST_HPP
#define SPRL_GAME_ACTION_DIST_HPP

#include <array>

namespace SPRL {

/**
 * Represents a normalized distribution over legal actions.
 * 
 * @tparam AS The size of the action space.
*/
template <int AS>
using GameActionDist = std::array<float, AS>;

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

/**
 * @tparam AS The size of the action space.
 * 
 * @param lhs The action distribution.
 * @param rhs The constant to exponentiate by.
 * 
 * @returns The action distribution exponentiated by the constant.
*/
template <int AS>
GameActionDist<AS> pow(const GameActionDist<AS>& lhs, float rhs) {
    GameActionDist<AS> result {};

    for (int i = 0; i < AS; ++i) {
        result[i] = std::pow(lhs[i], rhs);
    }

    return result;
}

/**
 * @tparam AS The size of the action space.
 * 
 * @param actionDist The action distribution to sum.
 * 
 * @returns The sum of the action distribution.
*/
template <int AS>
float sum(const GameActionDist<AS>& actionDist) {
    float result = 0.0;

    for (int i = 0; i < AS; ++i) {
        result += actionDist[i];
    }

    return result;
}

} // namespace SPRL

#endif