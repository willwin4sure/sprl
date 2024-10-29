import numpy as np

FILES = "abcdefgh"
RANKS = "12345678"
UNDERPROMOTIONS = "nbr"

def main():
    queen_moves = []

    for rank in range(8):
        for file in range(8):
            # Horizontal movement.
            for f in range(8):
                if f != file:
                    queen_moves.append((file, rank, f, rank))
                    
                    
            # Vertical movement.
            for r in range(8):
                if r != rank:
                    queen_moves.append((file, rank, file, r))
                
                    
            # Main diagonal movement.
            for f, r in zip(range(file + 1, 8), range(rank + 1, 8)):
                queen_moves.append((file, rank, f, r))
                
            for f, r in zip(range(file - 1, -1, -1), range(rank - 1, -1, -1)):
                queen_moves.append((file, rank, f, r))
                
                
            # Secondary diagonal movement.
            for f, r in zip(range(file + 1, 8), range(rank - 1, -1, -1)):
                queen_moves.append((file, rank, f, r))
                
            for f, r in zip(range(file - 1, -1, -1), range(rank + 1, 8)):
                queen_moves.append((file, rank, f, r))
                
    assert len(queen_moves) == 1456

    queen_moves_strs = sorted([
        f"{FILES[move[0]]}{RANKS[move[1]]}{FILES[move[2]]}{RANKS[move[3]]}"
        for move in queen_moves
    ])

    knight_moves = []

    # Vertical 3 x 2 blocks.
    for rank in range(6):
        for file in range(7):
            knight_moves.append((file, rank, file + 1, rank + 2))
            knight_moves.append((file + 1, rank + 2, file, rank))
            
            knight_moves.append((file + 1, rank, file, rank + 2))
            knight_moves.append((file, rank + 2, file + 1, rank))
            
    # Horizontal 2 x 3 blocks.
    for rank in range(7):
        for file in range(6):
            knight_moves.append((file, rank, file + 2, rank + 1))
            knight_moves.append((file + 2, rank + 1, file, rank))
            
            knight_moves.append((file + 2, rank, file, rank + 1))
            knight_moves.append((file, rank + 1, file + 2, rank))
            
    assert len(knight_moves) == 336

    knight_moves_strs = sorted([
        f"{FILES[move[0]]}{RANKS[move[1]]}{FILES[move[2]]}{RANKS[move[3]]}"
        for move in knight_moves
    ])

    underpromotions = []

    for promotion in UNDERPROMOTIONS:
        for start_file in range(8):
            for end_file in range(max(0, start_file - 1), min(8, start_file + 2)):
                underpromotions.append((start_file, 6, end_file, 7, promotion))
                
    assert len(underpromotions) == 66

    underpromotions_strs = sorted([
        f"{FILES[move[0]]}{RANKS[move[1]]}{FILES[move[2]]}{RANKS[move[3]]}{move[4]}"
        for move in underpromotions
    ])

    all_moves_strs = queen_moves_strs + knight_moves_strs + underpromotions_strs
    assert len(all_moves_strs) == 1858

    with open("moves.txt", "w") as file:
        for idx, move_str in enumerate(all_moves_strs):
            file.write(f"\"{move_str}\", ")
            if idx % 8 == 7:
                file.write("\n")
                
    mapping = -1 * np.ones((73, 8, 8), dtype=np.int16)

    QUEEN_DIRS = [
        (1, 0), (1, 1), (0, 1), (-1, 1),
        (-1, 0), (-1, -1), (0, -1), (1, -1)
    ]
        
    for idx, move in enumerate(queen_moves_strs):
        start = (FILES.index(move[0]), RANKS.index(move[1]))
        end = (FILES.index(move[2]), RANKS.index(move[3]))
        
        num_squares = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
        direction = QUEEN_DIRS.index((np.sign(end[0] - start[0]), np.sign(end[1] - start[1])))
        
        mapping_idx = 7 * direction + num_squares - 1
        move_idx = idx
        mapping[mapping_idx][start[0]][start[1]] = move_idx
        
    KNIGHT_DIRS = [
        (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]

    for idx, move in enumerate(knight_moves_strs):
        start = (FILES.index(move[0]), RANKS.index(move[1]))
        end = (FILES.index(move[2]), RANKS.index(move[3]))
        
        direction = KNIGHT_DIRS.index((end[0] - start[0], end[1] - start[1]))
        
        mapping_idx = 56 + direction
        move_idx = len(queen_moves_strs) + idx
        mapping[mapping_idx][start[0]][start[1]] = move_idx
        
    for idx, move in enumerate(underpromotions_strs):
        start = (FILES.index(move[0]), RANKS.index(move[1]))
        end = (FILES.index(move[2]), RANKS.index(move[3]))
        
        direction = end[0] - start[0] + 1
        promotion = move[4]
        
        mapping_idx = 64 + 3 * direction + UNDERPROMOTIONS.index(promotion)
        move_idx = len(queen_moves_strs) + len(knight_moves_strs) + idx
        mapping[mapping_idx][start[0]][start[1]] = move_idx
        
    seen = set()

    for map_idx in mapping.flatten():
        if map_idx != -1:
            assert not map_idx in seen
            seen.add(map_idx)
            
    for map_idx in range(len(all_moves_strs)):
        assert map_idx in seen
        
    with open("mapping.txt", "w") as file:
        for idx, map_idx in enumerate(mapping.flatten()):
            file.write(str(map_idx).rjust(4) + ", ")
            
            if idx % 8 == 7:
                file.write("\n")
                
            if idx % 64 == 63:
                file.write("\n")
                
if __name__ == "__main__":
    main()