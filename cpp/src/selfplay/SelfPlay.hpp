#ifndef SPRL_SELF_PLAY_HPP
#define SPRL_SELF_PLAY_HPP

/**
 * @file SelfPlay.cpp
 * 
 * Provides functionality to generate self-play data.
*/

#include "../games/GameNode.hpp"
#include "../networks/INetwork.hpp"
#include "../selfplay/SelfPlayOptions.hpp"
#include "../symmetry/ISymmetrizer.hpp"
#include "../uct/UCTOptions.hpp"
#include "../uct/UCTTree.hpp"

#include "../utils/random.hpp"
#include "../utils/tqdm.hpp"

#include "../constants.hpp"

#include <iostream>
#include <string>

namespace SPRL {

/**
 * Generates a single game of self-play data using the given game and network, improved with UCT.
 * 
 * @tparam ImplNode The implementation of the game node, e.g. `GoNode`.
 * @tparam State The state of the game, e.g. `GridState`.
 * @tparam ACTION_SIZE The size of the action space.
 * 
 * @param iterationOptions The options for the iteration.
 * @param treeOptions The options for the UCT tree.
 * @param network The neural network to use for evaluation.
 * @param symmetrizer The symmetrizer to use for symmetrizing the network and data (or nullptr).
 * 
 * @returns A tuple of:
 *     1. A vector of states, where each state is a symmetrized version of the game state over time.
 *     2. A vector of action distributions, where each distribution is a symmetrized version
 *        of the action distribution produced by UCT.
 *     3. A vector of outcomes, where each outcome is the reward for the corresponding player
 *        that took an action at any given state.
*/
template <typename ImplNode, typename State, int ACTION_SIZE>
std::tuple<std::vector<State>, std::vector<GameActionDist<ACTION_SIZE>>, std::vector<Value>>
selfPlay(IterationOptions iterationOptions,
         TreeOptions treeOptions,
         INetwork<State, ACTION_SIZE>* network,
         ISymmetrizer<State, ACTION_SIZE>* symmetrizer) {

    using ActionDist = GameActionDist<ACTION_SIZE>;

    std::vector<State> states;
    std::vector<ActionDist> distributions;
    std::vector<Player> players;
    std::vector<Value> outcomes;

    UCTTree<ImplNode, State, ACTION_SIZE> tree { treeOptions, symmetrizer };

    int moveCount = 0;

    while (!tree.getDecisionNode()->isTerminal()) {
        // Decide whether to make a full search or a fast playout.
        bool doFullSearch = GetRandom()() >= iterationOptions.fastPlayoutProb;

        // Perform `numTraversals` many search iterations.
        int numTraversals = doFullSearch
            ? iterationOptions.uctTraversals
            : iterationOptions.uctTraversals * iterationOptions.fastPlayoutFactor;
        
        int traversals = 0;
        while (traversals < numTraversals) {
            // Returns vector of collected leaves and total number of traversals performed.
            auto [leaves, trav] = tree.searchAndGetLeaves(
                iterationOptions.maxBatchSize, iterationOptions.maxQueueSize, iterationOptions.forcedPlayouts, network);

            // If leaves were collected, evaluate and backpropagate them.
            if (leaves.size() > 0) {
                tree.evaluateAndBackpropLeaves(leaves, network, doFullSearch);
            }

            traversals += trav;
        }

        // Generate a PDF from the visit counts.
        ActionDist visits = tree.getDecisionNode()->getEdgeStatistics()->m_numVisits;
        ActionDist pdf = visits / visits.sum();

        // Raise it to 0.98f (temp ~ 1) if early game, else 10.0f (temp -> 0), then renormalize.
        if (moveCount < iterationOptions.earlyGameCutoff) {
            pdf = pdf.pow(iterationOptions.earlyGameExp);

        } else {
            pdf = pdf.pow(iterationOptions.restGameExp);
        }

        pdf = pdf / pdf.sum();

        // Insert training data if doing full search.
        // Includes policy target pruning, decoupled from move selection.

        if (doFullSearch) {
            insertTrainingData(iterationOptions, pdf, tree, states, distributions, players, symmetrizer);
        }

        // Perform move sampling.

        // Generate a CDF from the PDF.
        ActionDist cdf = pdf.cumsum();
        cdf = cdf / cdf[ACTION_SIZE - 1];

        // Sample from the CDF.
        int action = GetRandom().SampleCDF(std::vector<float>(cdf.begin(), cdf.end()));

        // Play the action by rerooting the tree and updating the state.
        tree.advanceDecision(action);

        ++moveCount;
    }

    std::array<Value, 2> rewards = tree.getDecisionNode()->getRewards();

    // Now that we have the rewards, construct the relative outcomes based on the players.

    outcomes.reserve(states.size());

    for (Player player : players) {
        switch (player) {
        case Player::ZERO:
            if (iterationOptions.symmetrizeData && symmetrizer != nullptr) {
                outcomes.resize(outcomes.size() + symmetrizer->numSymmetries(), rewards[0]);

            } else {
                outcomes.push_back(rewards[0]);
            }
            
            break;

        case Player::ONE:
            if (iterationOptions.symmetrizeData && symmetrizer != nullptr) {
                outcomes.resize(outcomes.size() + symmetrizer->numSymmetries(), rewards[1]);

            } else {
                outcomes.push_back(rewards[1]);
            }

            break;

        default:
            assert(false);
            // Should never happen, but we fill with zeros anyway.
            if (iterationOptions.symmetrizeData && symmetrizer != nullptr) {
                outcomes.resize(outcomes.size() + symmetrizer->numSymmetries(), 0.0f);

            } else {
                outcomes.push_back(0.0f);
            }
        }
    }

    return { states, distributions, outcomes };
}

/**
 * Inserts training data into the vectors of states and distributions.
 * 
 * @tparam ImplNode The implementation of the game node, e.g. `GoNode`.
 * @tparam State The state of the game, e.g. `GridState`.
 * @tparam ACTION_SIZE The size of the action space.
 * 
 * @param iterationOptions The options for the iteration.
 * @param pdf The policy distribution that we are sampling from, i.e. exponentiated visit counts.
 * @param tree The UCT tree to extract data from.
 * @param states The vector of states to insert into.
 * @param distributions The vector of distributions to insert into.
 * @param players The vector of players to insert into.
 * @param symmetrizer The symmetrizer to use for symmetrizing the data (or nullptr).
 * @param allSymmetries The vector of all symmetries to use for symmetrizing the data.
 */
template <typename ImplNode, typename State, int ACTION_SIZE>
void insertTrainingData(IterationOptions iterationOptions,
                        const GameActionDist<ACTION_SIZE>& pdf,
                        UCTTree<ImplNode, State, ACTION_SIZE>& tree,
                        std::vector<State>& states,
                        std::vector<GameActionDist<ACTION_SIZE>>& distributions,
                        std::vector<Player>& players,
                        ISymmetrizer<State, ACTION_SIZE>* symmetrizer) {

    // Contains all symmetries if symmetrizeData is true and a symmetrizer is provided.
    std::vector<SymmetryIdx> allSymmetries;
    if (iterationOptions.symmetrizeData && symmetrizer != nullptr) {
        for (int i = 0; i < symmetrizer->numSymmetries(); ++i) {
            allSymmetries.push_back(i);
        }
    }
    
    using ActionDist = GameActionDist<ACTION_SIZE>;
    
    if (iterationOptions.symmetrizeData && symmetrizer != nullptr) {
        // Symmetrize the state and add to data.
        std::vector<State> symmetrizedStates = symmetrizer->symmetrizeState(
            tree.getDecisionNode()->getGameState(), allSymmetries);

        states.reserve(states.size() + symmetrizedStates.size());
        states.insert(states.end(), symmetrizedStates.begin(), symmetrizedStates.end());

    } else {
        states.push_back(tree.getDecisionNode()->getGameState());
    }
    
    // Either use policy target pruning or move sampling pdf.
    ActionDist policyTarget = iterationOptions.policyTargetPruning
        ? tree.getDecisionNode()->getPolicyTarget() : pdf;
    
    if (iterationOptions.symmetrizeData && symmetrizer != nullptr) {
        // Symmetrize the distributions and add to data.
        std::vector<ActionDist> symmetrizedDists = symmetrizer->symmetrizeActionDist(
            policyTarget, allSymmetries);

        distributions.reserve(distributions.size() + symmetrizedDists.size());
        distributions.insert(distributions.end(), symmetrizedDists.begin(), symmetrizedDists.end());

    } else {
        distributions.push_back(policyTarget);
    }

    // Record the player that just took the action.
    players.push_back(tree.getDecisionNode()->getPlayer());
}


/**
 * Runs `numGames` many games of self-play and collates all the data.
 * 
 * @tparam ImplNode The implementation of the game node, e.g. `GoNode`.
 * @tparam State The state of the game, e.g. `GridState`.
 * @tparam ACTION_SIZE The size of the action space.
 * 
 * @param iterationOptions The options for the iteration.
 * @param treeOptions The options for the UCT tree.
 * @param network The neural network to use for evaluation.
 * @param symmetrizer The symmetrizer to use for symmetrizing the network and data (or nullptr).
*/
template <typename ImplNode, typename State, int ACTION_SIZE>
std::tuple<std::vector<State>, std::vector<GameActionDist<ACTION_SIZE>>, std::vector<Value>>
runIteration(IterationOptions iterationOptions,
             TreeOptions treeOptions,
             INetwork<State, ACTION_SIZE>* network,
             ISymmetrizer<State, ACTION_SIZE>* symmetrizer) {

    using ActionDist = GameActionDist<ACTION_SIZE>;

    std::vector<State> allStates;
    std::vector<ActionDist> allDistributions;
    std::vector<Value> allOutcomes;

    for (int t = 0; t < iterationOptions.numGamesPerWorker; ++t) {
        auto [states, distributions, outcomes] = selfPlay<ImplNode, State, ACTION_SIZE>(
            iterationOptions,
            treeOptions,
            network,
            symmetrizer
        );

        // Push on the new state, distribution, and outcome data.
        // This data is ready for training on after the states are embedded.

        allStates.reserve(allStates.size() + states.size());
        allStates.insert(allStates.end(), states.begin(), states.end());

        allDistributions.reserve(allDistributions.size() + distributions.size());
        allDistributions.insert(allDistributions.end(), distributions.begin(), distributions.end());

        allOutcomes.reserve(allOutcomes.size() + outcomes.size());
        allOutcomes.insert(allOutcomes.end(), outcomes.begin(), outcomes.end());

        std::cout << t + 1 << " games played, " << allStates.size() << " states collected.\n";
    }

    assert(allStates.size() == allDistributions.size());
    assert(allStates.size() == allOutcomes.size());

    return { allStates, allDistributions, allOutcomes };
}

} // namespace SPRL

#endif