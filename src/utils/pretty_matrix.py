def pretty_matrix(matrix, col_seperator=None, first_row_seperator=None, first_col_seperator=None):
    # Modified from https://stackoverflow.com/questions/13214809/pretty-print-2d-list
    s = [[str(e) for e in row] for row in matrix]
    rows, cols = len(s), len(s[0])

    # if there is a first_col_seperator, draw it at the beginning of the second column
    if first_col_seperator and cols >= 2:
        for i in range(rows):
            s[i][1] = first_col_seperator + " " + s[i][1]

    if col_seperator:
        for i in range(rows):
            for j in range(1, cols):
                s[i][j] = col_seperator + " " + s[i][j]

    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = ' '.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    table_length = max(map(len, table))
    # if there is a first_row_seperator, place it in between the first and second row.
    if first_row_seperator and rows >= 2:
        table.insert(1, first_row_seperator * table_length)

    return '\n'.join(table)


def pretty_dict_matrix(matrix, col_seperator=None, first_row_seperator=None, first_col_seperator=None, title=""):
    # take the matrix, and add the names of each row and column to the front. Place the title in the upper left.

    row_names = list(matrix.keys())
    col_names = list(matrix[row_names[0]].keys())

    full_matrix = [[title] + col_names] + [[row_name] + [matrix[row_name]
                                                         [col_name] for col_name in col_names] for row_name in row_names]

    return pretty_matrix(full_matrix, col_seperator, first_row_seperator, first_col_seperator)


if __name__ == "__main__":
    # Example usage
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(pretty_matrix(matrix, col_seperator="|",
          first_row_seperator="*", first_col_seperator=None))

    dict_matrix = {
        "A": {"D": 1, "E": 2, "F": 3},
        "B": {"D": 4, "E": 5, "F": 6},
        "C": {"D": 7, "E": 8, "F": 9}
    }
    print(pretty_dict_matrix(dict_matrix, col_seperator=None,
          first_row_seperator="-", first_col_seperator="|", title="Test"))
