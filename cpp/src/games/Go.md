# Go implementation Algorithm Specification

We follow the [Tromp-Taylor rules](https://senseis.xmp.net/?TrompTaylorRules) for the game of Go.

## Get Action Mask

The TT rules do NOT forbid suicidal moves, hence the action mask is exactly the collection of non-empty cells on the board, minus Ko moves.

_Only_ to make the ML problem easier, we will explicitly forbid suicidal moves in the action mask. This incurs constants on time and space.

Notes on Ko:

-   SSK moves are explicitly forbidden in the action mask.
-   A move that violates 8-move PSK is caught in the movement stage via Zobrist hash. It results in an immediate loss for the player who played the move.

## Play Move

The objective is to be able to play a move in O(boardsize) worst-case (polylog factors are suppressed), but also
