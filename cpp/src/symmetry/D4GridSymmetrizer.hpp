#ifndef SPRL_D4_GRID_SYMMETRIZER_HPP
#define SPRL_D4_GRID_SYMMETRIZER_HPP

#include "ISymmetrizer.hpp"

#include "games/GridState.hpp"

#include <functional>
#include <utility>

namespace SPRL {

/**
 * Symmetrizer for a square board with one action possible per cell,
 * plus a pass action, e.g. Go or Othello.
 * 
 * The group of symmetries is the dihedral group D4.
 * 
 * @tparam BOARD_WIDTH The width of the board.
 * @tparam HISTORY_SIZE The size of the history.
*/
template <int BOARD_WIDTH, int HISTORY_SIZE>
class D4GridSymmetrizer : public ISymmetrizer<
    GridState<BOARD_WIDTH * BOARD_WIDTH, HISTORY_SIZE>, BOARD_WIDTH * BOARD_WIDTH + 1> {
public:
    using Board = GridBoard<BOARD_WIDTH * BOARD_WIDTH>;
    using State = GridState<BOARD_WIDTH * BOARD_WIDTH, HISTORY_SIZE>;
    using ActionDist = GameActionDist<BOARD_WIDTH * BOARD_WIDTH + 1>;

    int numSymmetries() const override {

        // Symmetry group is dihedral D_4: we have four rotations and four reflected rotations
        //
        // Mapping:
        //     0: identity
        //     1: single 90 deg cw rotation
        //     2: full 180 deg rotation
        //     3: single 90 deg ccw rotation
        //     4: reflection across vertical axis
        //     5: reflection across vertical axis followed by single 90 deg cw rotation
        //     6: reflection across vertical axis followed by full 180 deg rotation
        //     7: reflection across vertical axis followed by single 90 deg ccw rotation

        return 8;
    }

    SymmetryIdx inverseSymmetry(SymmetryIdx symmetry) const override {
        static constexpr std::array<SymmetryIdx, 8> inverseSymmetries = { 0, 3, 2, 1, 4, 5, 6, 7 };
        return inverseSymmetries[symmetry];
    }

    std::vector<State> symmetrizeState(
        const State& state,
        const std::vector<SymmetryIdx>& symmetries) const override {

        std::vector<State> symmetriesStates;
        symmetriesStates.reserve(symmetries.size());

        std::array<Board, HISTORY_SIZE> history = state.getHistory();

        for (SymmetryIdx symmetry : symmetries) {
            std::array<Board, HISTORY_SIZE> symmetrizedHistory;
            for (int t = 0; t < state.size(); ++t) {
                for (int fromRow = 0; fromRow < BOARD_WIDTH; ++fromRow) {
                    for (int fromCol = 0; fromCol < BOARD_WIDTH; ++fromCol) {
                        auto [toRow, toCol] = s_symmetrizeFunctions[symmetry](fromRow, fromCol);
                        symmetrizedHistory[t][toIndex(toRow, toCol)] = history[t][toIndex(fromRow, fromCol)];
                    }
                }
            }
            symmetriesStates.push_back(State { std::move(symmetrizedHistory), state.size(), state.getPlayer() });            
        }

        return symmetriesStates;
    }

    std::vector<ActionDist> symmetrizeActionDist(
        const ActionDist& actionDist,
        const std::vector<SymmetryIdx>& symmetries) const override {

        std::vector<ActionDist> symmetrizedActionDists;
        symmetrizedActionDists.reserve(symmetries.size());

        for (SymmetryIdx symmetry : symmetries) {
            ActionDist symmetrizedActionDist;

            for (int fromRow = 0; fromRow < BOARD_WIDTH; ++fromRow) {
                for (int fromCol = 0; fromCol < BOARD_WIDTH; ++fromCol) {
                    auto [toRow, toCol] = s_symmetrizeFunctions[symmetry](fromRow, fromCol);
                    symmetrizedActionDist[toIndex(toRow, toCol)] = actionDist[toIndex(fromRow, fromCol)];
                }
            }
            
            // Pass action
            symmetrizedActionDist[BOARD_WIDTH * BOARD_WIDTH] = actionDist[BOARD_WIDTH * BOARD_WIDTH];
            
            symmetrizedActionDists.push_back(symmetrizedActionDist);
        }

        return symmetrizedActionDists;
    }

private:
    static int toIndex(int row, int col) {
        return row * BOARD_WIDTH + col;
    }

    static inline const std::array<std::function<std::pair<int, int>(int, int)>, 8> s_symmetrizeFunctions {
        [](int row, int col) { return std::make_pair(row, col); },
        [](int row, int col) { return std::make_pair(col, BOARD_WIDTH - 1 - row); },
        [](int row, int col) { return std::make_pair(BOARD_WIDTH - 1 - row, BOARD_WIDTH - 1 - col); },
        [](int row, int col) { return std::make_pair(BOARD_WIDTH - 1 - col, row); },
        [](int row, int col) { return std::make_pair(row, BOARD_WIDTH - 1 - col); },
        [](int row, int col) { return std::make_pair(BOARD_WIDTH - 1 - col, BOARD_WIDTH - 1 - row); },
        [](int row, int col) { return std::make_pair(BOARD_WIDTH - 1 - row, col); },
        [](int row, int col) { return std::make_pair(col, row); }
    };
};

} // namespace SPRL

#endif
