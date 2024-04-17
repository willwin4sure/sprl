# UCT Algorithm Description

This file contains documentation for our implementation
of UCT tree search, a *policy improvement operator*
used both in training and inference of our agent.

The primary object of the algorithm is a `UCTTree`,
which consists of many `UCTNode` instances linked together.

A high-level description of the algorithm: a tree of game
states is constructed iteratively, with a new leaf added
every iteration. There are two stages:

* In the downward pass, a leaf to be added to the tree is
selected greedily via a UCB-style algorithm. Each node stores
two values for each potential next action: a $Q$ value for the
current average value estimate of taking that action,
and a $U$ value for the optimistic uncertainty we add to the
$Q$ value. $Q$ is tallied over iterations of the tree search
(updated in the upward pass), while $U$ is proportional
to the policy given by the neural network, and inversely
proportional to the number of times the branch has already
been searched.

* In the upward pass, we travel back up to the root and update
all the $Q$ and $U$ values.

Many optimizations on this basic algorithm are also implemented.
For example:

* We implement **subtree re-use**, which allows the network
to capitalize on work from previous moves.

* We **symmetrize** the inputs into the network by choosing a
random symmetric game state. We also leverage symmetry to
increase the amount of training data for the neural network.

* We also batch neural network evaluation using the technique
of **virtual losses**. Since the leaf selection is deterministic,
we pretend we lose on the way down to repeatedly sample leaves,
then push them through the NN together, before backpropagating.

Several other augmentations are also implemented.

* As described in the AlphaGo Zero paper, when we generate the
policy for the root node in each search step, we mix Dirichlet
noise into the network output to construct priors.

* Different from the AlphaGo Zero paper but present in open
source implementations, we initialize the $Q$ values of new
children in the tree to the values of their parents,
rather than to $0$.

## UCT Nodes (`UCTNode.hpp`)

These hold the following state:

* The current game state and its properties: the legal action
mask and whether it is terminal.

* The parent of the node and its children by action to take.
Claims ownership over the children by holding `unique_ptr`s.
Also includes the action taken into the current node.

* Edge statistics for the outgoing edges from this node.
These contain:

  - Priors (`P`) on the children, from the policy network.
  - Accumulated value (`W`) on the children, from traversals.
  - Number of visits (`N`) on the children, from traversals.

* What type of node it currently is for the tree traversal,
based on two bits: `isExpanded` and `isNetworkEvaluated`.
Only non-terminal nodes are categorized into these three
legal combinations (terminal nodes must be handled separately):

  - Neither expanded nor evaluated. We call such nodes **empty**.
    They've just been created, and likely soon
    will be evaluated and then expanded.

  - Evaluted but not expanded. We call such nodes **gray**.
    For example, when the tree is rerooted, all nodes are
    reset to gray. This caches the network evaluations
    while resetting the information on the tree.

  - Both evaluated and expanded. We call such nodes **active**.
    Such nodes are being actively traversed acrossed
    and their statistics are progressively updated.

The design principle of a node is to hold and update *local*
information of the tree.

It has a bunch of functions to get and set the edge statistics
of the current node and its children. It also computes
the current value of a particular action as the sum of an
exploitation term and an exploration term:
$$V(e) = \underbrace{\frac{W(e)}{1+N(e)}}_{Q(e)} +
\underbrace{P(e)\cdot\frac{\sqrt{\sum_{e'}N(e')}}{1+N(e)}}_{U(e)}.$$
This value is from the perspective of the parent node,
in the sense that higher values are more promising for the
player to act at the parent node.

The function `bestAction` can only be applied on active nodes,
and uses the above formula to determine which legal action
is the most promising at the current moment. This is used
in downward traversals of the tree to select nodes.

The function `getAddChild` gets the child of a node given
a particular action. It also creates the node if necessary.
It can be used on any non-terminal nodes.

The function `updateNetworkOutput` is used to turn empty nodes
into gray nodes by setting the `isNetworkEvaluated` bit and
also caching the outputs of the network on the current
game state.

The function `expand` is used to turn gray nodes into active
nodes by setting the `isExpanded` bit and setting the
priors equal to the network policy on legal actions.
It will also add Dirichlet noise if necessary.

The function `pruneChildrenExcept` is used on non-terminal
nodes to destroy all children of a node except for one
particular one. Necessary in subtree reuse.

## UCT Trees (`UCTTree.hpp`)

These hold the following state:

* The root of the tree. Claims ownership over it via a
`unique_ptr`.

* An edge statistics object that is a stand-in for the
parent edge statistics object of the root, to allow
the same access code.

There are four functions that are actually exposed for
modifying the tree.

The function `searchIteration` is a basic implementation
of a single iteration of both a downwards and upwards pass.
This does *not* batch the neural network inference,
so it is considerably slower. 

The function `searchAndGetLeaves` performs multiple downwards
passes. It immediately performs upwards passes if the leaf
is terminal or gray (in which case the value can be immediately
determined), but otherwise adds the leaf to a queue for
batched NN inference. Since the downwards pass is
deterministic, **virtual losses** need to be employed,
where during the downward pass, we pretend we know the
outcome is already a loss, so during the next iteration
we are less likely to travel down the same path. The function
returns a collection of empty leaf nodes that need to be
evaluated by the network and then backed up.

The function `evaluateAndBackpropLeaves` takes in a vector
of empty leaves. It performs batched NN inference on them
and then performs the upwards pass on all of them to
remove the virtual losses.

The function `rerootTree` takes in an action and then
replaces the root node of the tree with the result of playing
that action from the root node. It preserves all the NN
inferences performed on the entire subtree, but resets
all the nodes to gray, so that traversals start from scratch.

There are some private functions `selectLeaf` and `backup`
that handle single downward and upward passes, as well
as `clearSubtree` which recursively destroys edge
statistics and resets expanded bits.
