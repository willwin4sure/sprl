# UCT Algorithm Description

This file contains documentation for our implementation
of UCT tree search, a _policy improvement operator_
used both in training and inference of our agent.
The basic algorithm is described in this first section.
Modifications and improvements are described in the second section.

The primary object of the algorithm is a `UCTTree`,
which consists of many `UCTNode` instances linked together.

A high-level description of the algorithm: a tree of game
states is constructed iteratively, with a new leaf added
every iteration. There are two stages:

-   In the downward pass, a leaf to be added to the tree is
    selected greedily via a bandits UCB-style algorithm. Each node stores
    two values for each potential next action: a $Q$ value for the
    current average value estimate of taking that action,
    and a $U$ value for the optimistic uncertainty we add to the
    $Q$ value. $Q$ is tallied over iterations of the tree search
    (updated in the upward pass), while $U$ is proportional
    to the policy given by the neural network, and inversely
    proportional to the number of times the branch has already
    been searched.

-   In the upward pass, we travel back up to the root and update
    all the $Q$ and $U$ values.

Many optimizations on this basic algorithm are also implemented.
For example:

-   We implement **subtree re-use**, which allows the network
    to capitalize on work from previous moves.

-   We **symmetrize** the inputs into the network by choosing a
    random symmetric game state. We also leverage symmetry to
    increase the amount of training data for the neural network.

-   We also batch neural network evaluation using the technique
    of **virtual losses**. Since the leaf selection is deterministic,
    we pretend we lose on the way down to repeatedly sample leaves,
    then push them through the NN together, before back-propagating.

## UCT Nodes (`UCTNode.hpp`)

These hold the following state:

-   A raw pointer `m_gameNode` to the game node associated
    with this UCT node, i.e. that we are "attached to". This
    holds some game state and all implementation of the game.

-   The parent of the node and its children by action to take.
    Claims ownership over the children by holding `unique_ptr`s.
    Also includes the action taken into the current node.

-   Edge statistics for the outgoing edges from this node.
    These contain:

    -   Priors (`P`) on the children, from the policy network.
    -   Accumulated value (`W`) on the children, from traversals.
    -   Number of visits (`N`) on the children, from traversals.

-   What type of node it currently is for the tree traversal,
    based on two bits: `m_isExpanded` and `m_isNetworkEvaluated`.
    Only non-terminal nodes are categorized into these three
    legal combinations (terminal nodes must be handled separately):

    -   Neither expanded nor evaluated. We call such nodes **empty**.
        They've just been created, and likely soon
        will be evaluated and then expanded.

    -   Evaluated but not expanded. We call such nodes **gray**.
        For example, when the tree is re-rooted, all nodes are
        reset to gray. This caches the network evaluations
        while resetting the information on the tree.

    -   Both evaluated and expanded. We call such nodes **active**.
        Such nodes are being actively traversed across
        and their statistics are progressively updated.

The design principle of a node is to hold and update _local_
information of the tree.

It has a bunch of functions to get and set the edge statistics
of the current node and its children. It also computes
the current value of a particular action as the sum of an
exploitation term and an exploration term:

$$
V(e) = \underbrace{\frac{W(e)}{1+N(e)}}_{Q(e)} +
\underbrace{P(e)\cdot\frac{\sqrt{\sum_{e'}N(e')}}{1+N(e)}}_{U(e)}.
$$

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

The function `addNetworkOutput` is used to turn empty nodes
into gray nodes by setting the `m_isNetworkEvaluated` bit and
also caching the outputs of the network on the current
game state.

The function `expand` is used to turn gray nodes into active
nodes by setting the `m_isExpanded` bit and setting the
priors equal to the network policy on legal actions.
It will also add Dirichlet noise if necessary.

The function `pruneChildrenExcept` is used on non-terminal
nodes to destroy all children of a node except for one
particular one. Necessary in subtree reuse.

## UCT Trees (`UCTTree.hpp`)

The UCT tree is constructed "in parallel" to the game tree.
The tree will own both roots of the trees, and UCT nodes
will have raw pointers into their corresponding game nodes.

In particular, the tree holds:

-   The roots of both trees `m_gameRoot` and `m_uctRoot`.
    Claims ownership over the roots by holding `unique_ptr`s.
    The current position in the game is encoded by
    `m_decisionNode`, and a path from the root to the current
    decision node is always held (though other branches will
    be pruned).

-   An edge statistics object `m_edgeStatistics` that is a
    stand-in for the parent edge statistics object of the root,
    to allow reuse of the same code for accessing data.

There are three functions that are actually exposed for
modifying the tree.

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

The function `advanceDecision` takes in an action and then
replaces the root node of the tree with the result of playing
that action from the root node. It preserves all the NN
inferences performed on the entire subtree, but resets
all the nodes to gray, so that traversals start from scratch.

There are some private functions `selectLeaf` and `backup`
that handle single downward and upward passes, as well
as `clearSubtree` which recursively destroys edge
statistics and resets expanded bits.

# New Techniques

## AlphaGo-Inspired

Several other augmentations are also implemented.

-   As described in the AlphaGo Zero paper, when we generate the
    policy for the root node in each search step, we mix Dirichlet
    noise into the network output to construct priors.

-   AGZ and leading open-source implementations differ on how to
    treat the $Q$ values of empty children. Such children have
    not been evaluated by the network yet, except their $Q$-value
    must be used by the parent during the downwards tree-search.
    (It seems that in all implementations, one does not directly
    evaluate the children with the network until they are expanded.
    This is possibly to save a large constant factor; one does not
    have to waste time obtaining precise evaluations of
    suboptimal moves.)

    The AGZ paper claims to initialize such children to $0$.
    This is widely believed to be obviously bad. We offer
    options to initialize new children to the parent's raw
    evaluation, or to the parent's current ("live") $Q$-value.

## KataGo-Inspired

### Fast Playouts

In games such as Go, game playouts take many moves,
and the final outcome is a noisy binary signal.
The amount of time spent collecting these playouts
dominates the computation time; UCTS is highly data-limited.

It turns out to be advantageous to collect _more_ games of lower
quality. The quality of games is directly controlled by the number
of playouts per move. To balance between quality of training data
and quantity of games, we randomly sample the number of playouts
on each particular move. On proportion $p = 0.25$ of the moves,
we perform a full search, and the results are recorded for training.
For the remainder of the moves, we perform a fast search which uses
significantly less computation, and such results are not recorded.

### Policy Target Pruning

This technique offsets the impact of the Dirichlet noise
on the quality of the data. Introducing Dirichlet noise is important
so that during data collection (in the tree-search), unexpectedly good
moves are revealed and added to the data collection, even if the
current model says they have poor priors. However, of course, most such
moves turn out to be bad. This introduces noise into the training data,
because we are telling our tree search to explore those poor moves more
than it usually would, and we are training our model to
predict the distribution of playouts in the tree search.

The key insight is that there is no reason to believe
that the raw distribution of playouts in the tree search
are good targets for the policy network. One should use solely the
the frozen valuations $Q(e)$ and the original _model outputs_ $P(e)$
to obtain the policy target.

Again, this (quoted from KG) "decouples the policy
target from the dynamics of MCTS." Behavior that is positive for
obtaining good training data (e.g., explorative noise) does not need
to introduce extraneous noise into the training data!

We now claim that a good way to convert $Q(e)$ and $P(e)$ values into
a policy target is to simulate the behavior of an idealized
"un-altered" UCTS search. Specifically, we attempt to model a UCTS
search which began with _exactly correct_ $Q(e)$ values which do not
receive updates during the search, and whose $P(e)$ values are not
subject to dirichlet noise. We run such an idealized model for the same
number of searches as the real UCTS search experienced, and predict
the resulting playout distribution.

To be precise, we let $V$ represent $\argmax_e V(e)$, and
invert the $V(e)$ formula given the final
$Q$ and total move count $N$ to obtain

$$
\hat{N}(e') = \max\left(0, \frac{P(e) \sqrt{N}}{V - Q(e)} - 1\right).
$$

We then select the value of $V$ such that $\sum \hat{N}(e') = N$,
and then we use $\hat{N}(e')$ as the target for the policy network.

### Forced Playouts

Building off of Policy Target Pruning, we are now free to augment
the tree search without worrying about incurring bias in the training
data. Forced Playouts augments the strength of Dirichlet noise by forcing
a node with a high $P(c)$ prior to be selected, independent of its $V(c)$
evaluation -- even if the model believes that the move is horrible, if
Dirichlet noise has selected the move for exploration, it will be played
at least $N_{\text{forced}}$ times. The exact formula is

$$
N_{\text{forced}}(e) = \sqrt{2 \cdot P(e) \cdot (\sum_{e'} N(e'))}.
$$
