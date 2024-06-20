# Smart Game Format (SGF) Specification

## Example Game

```
(;
EV[Pro vs Zen on 9x9]
RO[1]
PB[Ichiriki Ryo]
BR[2p]
PW[Zen]
KM[7]
RE[B+R]
DT[2012-11-25]
SZ[9]
RU[Chinese]

;B[df];W[fd];B[ed];W[ec];B[fe];W[ge];B[ee];W[fc];B[gf];W[cg]
;B[cf];W[he];B[hf];W[fg];B[dg];W[ff];B[gh];W[gg];B[hg];W[hh]
;B[ih];W[hi];B[gi];W[if];B[ii];W[hh];B[hi];W[ie];B[dc];W[db]
;B[cc];W[eh];B[gd])
```

For our purposes, we will not be concerned with any of the metadata properties, and all games will be played on a 9x9 board. Nodes are delimited with `;` and contain a single move for either black or white, index with two letters representing the column and row of the move. The first move is always black.

`B[]` and `W[]` denote passes for black and white respectively; notation such as `B[tt]` is also used, where `t` is outside the range of acceptable columns and rows.

## Todo: Future ideas

The full SGF spec is actually meant to store full gametrees. One can easily imagine that our implementation could extend to all sorts of games, allowing us to store the entire UCT tree during evaluation in human-readable format. This would be useful for future UI allowing users to view the tree and understand the AI's thought process.
