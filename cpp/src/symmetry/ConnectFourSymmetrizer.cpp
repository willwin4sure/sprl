#include "ConnectFourSymmetrizer.hpp"

namespace SPRL {

int ConnectFourSymmetrizer::numSymmetries() const {
    // The only symmetries are the identity (0) and the vertical flip (1).
    return 2;
}

SymmetryIdx ConnectFourSymmetrizer::inverseSymmetry(SymmetryIdx symmetry) const {
    // All symmetries are involutions.
    return symmetry;
}

std::vector<ConnectFourSymmetrizer::State> ConnectFourSymmetrizer::symmetrizeState(
    const State& state,
    const std::vector<SymmetryIdx>& symmetries) const {

    std::vector<State> symmetrizedStates;
    symmetrizedStates.reserve(symmetries.size());

    for (SymmetryIdx symmetry : symmetries) {
        switch (symmetry) {
        case 0: {
            // The identity symmetry.
            symmetrizedStates.push_back(state);
            break;
        }

        case 1: {
            // The vertical flip symmetry.
            std::array<Board, C4_HISTORY_SIZE> symmetrizedBoards;
            auto history = state.getHistory();

            // There should be exactly one valid board in history.
            assert(state.size() == C4_HISTORY_SIZE);

            for (int idx = 0; idx < state.size(); ++idx) {
                Board symmetrizedBoard = history[idx];  // Copy the board.

                // Perform a vertical flip on the board.
                for (int i = 0; i < C4_NUM_ROWS; ++i) {
                    for (int j = 0; j < C4_NUM_COLS / 2; ++j) {
                        int idx1 = ConnectFourNode::toIndex(i, j);
                        int idx2 = ConnectFourNode::toIndex(i, C4_NUM_COLS - 1 - j);
                        std::swap(symmetrizedBoard[idx1], symmetrizedBoard[idx2]);
                    }
                }

                symmetrizedBoards[idx] = std::move(symmetrizedBoard);
            }

            symmetrizedStates.push_back(State {
                std::move(symmetrizedBoards), C4_HISTORY_SIZE, state.getPlayer() });
            break;
        }
        
        default:
            assert(false);
        }
    }

    return symmetrizedStates;
}

std::vector<ConnectFourSymmetrizer::ActionDist> ConnectFourSymmetrizer::symmetrizeActionDist(
    const ActionDist& actionDist,
    const std::vector<SymmetryIdx>& symmetries) const {

    std::vector<ActionDist> symmetrizedActionDists;
    symmetrizedActionDists.reserve(symmetries.size());

    for (SymmetryIdx symmetry : symmetries) {
        switch (symmetry) {
        case 0: {
            // The identity symmetry.
            symmetrizedActionDists.push_back(actionDist);
            break;
        }

        case 1: {
            // The vertical flip symmetry.
            ActionDist symmetrizedActionDist = actionDist;  // Copy the action distribution.

            // Perform a vertical flip on the action distribution.
            for (int i = 0; i < C4_NUM_COLS / 2; ++i) {
                std::swap(symmetrizedActionDist[i], symmetrizedActionDist[C4_NUM_COLS - 1 - i]);
            }

            symmetrizedActionDists.push_back(symmetrizedActionDist);
            break;
        }
        
        default:
            assert(false);
        }
    }

    return symmetrizedActionDists;
}

} // namespace SPRL