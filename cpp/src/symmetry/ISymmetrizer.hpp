#ifndef SPRL_SYMMETRIZER_HPP
#define SPRL_SYMMETRIZER_HPP

#include "../games/GameNode.hpp"

namespace SPRL {

/// Type alias for the symmetry index.
using SymmetryIdx = int8_t;

/**
 * Interface for a symmetrizer.
 * 
 * Used to apply symmetries to game states and action distributions
 * in compatible ways for games with rules invariant under transformations.
 * 
 * @tparam AS The size of the action space.
 * @tparam State The state that the symmetrizer operates on.
*/
template <typename State, int AS>
class ISymmetrizer {
public:
    using ActionDist = GameActionDist<AS>;

    /**
     * @returns The number of symmetries this symmetrizer can apply.
    */
    virtual int numSymmetries() const = 0;

    /**
     * @param symmetry The symmetry to invert, must be in range `[0, numSymmetries())`.
     * 
     * @returns The inverse of a given symmetry.
    */
    virtual SymmetryIdx inverseSymmetry(SymmetryIdx symmetry) const = 0;

    /**
     * @param state The state to symmetrize.
     * @param symmetries The symmetries to apply, all in range `[0, numSymmetries())`.
     * 
     * @returns A vector of symmetrized states, with the same order as `symmetries`.
    */
    virtual std::vector<State> symmetrizeState(const State& state,
                                               const std::vector<SymmetryIdx>& symmetries) const = 0;

    /**
     * @param actionDist The action distribution to symmetrize.
     * @param symmetries The symmetries to apply, all in range `[0, numSymmetries())`.
     * 
     * @returns A vector of symmetrized action masks, with the same order as `symmetries`.
    */
    virtual std::vector<ActionDist> symmetrizeActionDist(const ActionDist& actionDist,
                                                         const std::vector<SymmetryIdx>& symmetries) const = 0;
};

} // namespace SPRL

#endif