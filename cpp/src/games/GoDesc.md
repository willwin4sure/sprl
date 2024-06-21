# Go Implementation Description

We implement Go with positional superko (PSK) and Tromp-Taylor scoring.

## Game Rules

In Go, two players take turns placing stones on a board.
The goal is to surround territory and capture the opponent's stones.
The game ends when both players pass consecutively.
The player with more territory wins.

We call any contiguous set of stones of the same color a **group**.
The **liberties** of a group are the empty spaces adjacent to its stones.
If a group ever has no liberties, it is captured and removed from the board.

When a piece is placed, we first check if it captures any opposing groups;
if so, we remove them. Then, we check if the group that the new piece joins
has any liberties. If not, then the move is deemed an illegal **suicide**
and not included in the action mask.

Finally, there is the rule of PSK. If after any move, the board is in the
same state as it was after any previous move, then the move is considered
illegal.

## Time Complexity Targets

Our goal is to support efficient updates of game state. Note that copying
the state requires `O(BOARD_SIZE)` time, so we do not need to optimize
beyond this target. We drop all polylog factors in our analysis.

The largest bottleneck for reaching this target is calculating the legal
action mask after playing a move. In the new state, we want to be able
to check if each move is legal in `O(1)` to meet our target.

This is a somewhat non-trivial problem: any of the stone placements
in the new state may capture a nontrivial number of opponent groups.
Some of these moves might be illegal suicides, and we will also need
to check if the new state violates PSK! This motivates the design
of our state.

## Zobrist Hashing

In order to efficiently check PSK, we use the famous Zobrist hashing
technique, which provides an *efficiently updatable* hash of the board
state that avoids collisions with high probability. Instead of checking
entire boards against each other, we just check their 64-bit hashes.
See `utils/Zobrist.hpp` for implementation details.

## `GoNode` State and Action Mask Computation

In order to facilitate these efficient queries, we maintain additional
state at each node of the game tree. The first one is `m_zobristHistorySet`,
which is a `std::unordered_set` holding the Zobrist hashes of all previous
nodes in the path from the root to the current node. This allows
efficient checking of PSK violations.

We store a DSU data structure `m_dsu` to hold the groups of stones
on the board, as well as additional state `m_liberties`
and `m_componentZobristValues` for the number of liberties
and total (XOR-ed) Zobrist hash of each group.

Now, when calculating the action mask of a new state:

* We can use liberties data to determine if it is a suicide in `O(1)`.
  In particular, you are a suicide if and only if the new stone
  is not adjacent to any empty squares *and* there is no
  capture of any enemy groups *and* you do not connect to any
  friendly group with at least two liberties (you use up one of them).

* We can use the component Zobrist values to determine if it
  is a PSK violation in `O(1)`. In particular, this allows us
  to efficiently compute the Zobrist hash of the new state
  if we detect that an enemy group would be captured
  (note: we don't actually spend the time to remove the stones!),
  and then check it against a stored set of previous Zobrist hashes.

There is some other basic state involved, such as the `m_board`
and its `m_hash`. We also store the `m_depth` of the game node
since we terminate any games after `2 * GO_BOARD_SIZE` moves
have been played, in order to prevent extremely long games
from slowing training.

You may also notice `s_zobrist`, which is just a static member
variable that holds the Zobrist values for each atomic element,
i.e. (Coord, Piece) pairs.

## Placing Pieces

In this section, we describe the rest of the algorithm, i.e.
how we actually update the state when a piece is placed.
Here, we don't need to be as time efficient since `O(BOARD_SIZE)`
is our target: DFS is fine.

1. First, we figure out the group that the new piece will be
   a part of. We do this by checking the neighbors and then
   updating the DSU data structure and component Zobrist value
   accordingly. We also need to update the liberties of this group
   via a DFS (there isn't a clean, correct way to do this in `O(1)`).
   Now all the liberty counts are correct modulo the removal
   of enemy groups.

2. Next, we check all enemy neighbors of the new stone and deduct
   a liberty for each group, making sure not to overcount the
   same group. If a group has no liberties, we remove it from the board,
   via a DFS. During this process, any friendly groups that
   gain liberties are updated as well.

3. Finally, we update the Zobrist hash of the board by XOR-ing
   all the updates we made using the component Zobrist values.
   We also push in the new Zobrist hash into the history set.

## Scoring Positions

The game terminates when both players pass consecutively or
if the maximum depth is reached. Either way, the current board
state is scored using Tromp-Taylor rules, which resolve the same
way as Chinese scoring if the game is played out.

Any stones still on the board are considered alive, and count
as a single point to the corresponding player. Then, the additional
territory of a player is defined as the set of empty spaces such that
there is no path of orthogonally adjacent empty spaces to a stone
of the enemy player (i.e. the space is entirely enclosed by
friendly stones).

Your score is your territory, plus a **komi** if you are the second
player to move. Komi depends on the board size, e.g. we use 9.0
for 7x7 (which is optimal) and 7.5 for 9x9. Whichever player
has the higher score wins.