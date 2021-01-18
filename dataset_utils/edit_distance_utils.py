import numpy as np
from scipy.optimize import linear_sum_assignment


def edit_matrix(a, b):
    """Construct edit distance matrix using classic dynamic programming algorithm."""
    n = len(a)
    m = len(b)
    d = [[0 for _ in range(m + 1)] for _ in range(n + 1)]

    for i in range(n + 1):
        d[i][0] = i

    for j in range(m + 1):
        d[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                d[i][j] = d[i - 1][j - 1]

            else:
                d[i][j] = min(d[i][j - 1] + 1,      # insert
                              d[i - 1][j] + 1,      # remove
                              d[i - 1][j - 1] + 1)  # replace
    return d


def edit_distance_alignment(a, b, special_token='<empty>', verbose=False):
    """Construct aligned sequences using sequences a and b and their edit distance matrix."""
    d = edit_matrix(a, b)

    prev_ans = []
    upd_ans = []

    if verbose:
        print("==EDIT DISTANCE MATRIX==")
        for row in d:
            print(row)
        print("==ITERATION==")

    # start from d[n][m]
    i = len(a)
    j = len(b)
    while not (i == 0 and j == 0):
        if j - 1 >= 0 and d[i][j] == d[i][j-1] + 1:
            # insert
            if verbose:
                print("Insert")
                print(b[j-1])
            prev_ans.append(special_token)
            upd_ans.append(b[j-1])
            j -= 1

        elif i - 1 >= 0 and d[i][j] == d[i-1][j] + 1:
            # delete
            if verbose:
                print("Delete")
                print(a[i-1])
            prev_ans.append(a[i-1])
            upd_ans.append(special_token)
            i -= 1

        elif i - 1 >= 0 and j - 1 >= 0 and d[i][j] == d[i-1][j-1] or d[i][j] == d[i-1][j-1] + 1:
            # replace / same element
            if verbose:
                if a[i-1] == b[j-1]:
                    print("Same")
                else:
                    print("Replace")
                print(a[i-1])
                print(b[j-1])
            prev_ans.append(a[i-1])
            upd_ans.append(b[j-1])
            i -= 1
            j -= 1
    return prev_ans[::-1], upd_ans[::-1]


def find_best_alignment_between_lists(list_a, list_b):
    """"Find how to align two lists of sequences to minimize total edit distance between those sequences.
    (using scipy function for assignment problem)"""
    edist = np.zeros((len(list_a), len(list_b)))
    for i in range(len(list_a)):
        for j in range(len(list_b)):
            edist[i][j] = edit_matrix(list_a[i], list_b[j])[len(list_a[i])][len(list_b[j])]
    return linear_sum_assignment(edist)


def align_lists(list_a, list_b):
    """Return optimal (i.e. with minimum total edit distance) alignment between two lists."""
    if len(list_a) == 0:
        ans_prev = [["<empty>" for _ in range(len(list_b[i]))] for i in range(len(list_b))]
        return ans_prev, list_b
    elif len(list_b) == 0:
        ans_upd = [["<empty>" for _ in range(len(list_a[i]))] for i in range(len(list_a))]
        return list_a, ans_upd

    rows, cols = find_best_alignment_between_lists(list_a, list_b)

    n = max(len(list_a), len(list_b))

    ans_prev = [0 for _ in range(n)]
    ans_upd = [0 for _ in range(n)]

    for i, j in zip(rows, cols):
        if len(list_a) > len(list_b):
            prev, upd = edit_distance_alignment(list_a[i], list_b[j])
            ans_prev[i] = prev
            ans_upd[i] = upd
        else:
            prev, upd = edit_distance_alignment(list_a[i], list_b[j])
            ans_prev[j] = prev
            ans_upd[j] = upd

    if len(list_a) > len(list_b):
        for i in [_ for _ in range(len(list_a)) if _ not in rows]:
            ans_prev[i] = list_a[i]
            ans_upd[i] = ["<empty>" for _ in range(len(list_a[i]))]

    elif len(list_a) < len(list_b):
        for j in [_ for _ in range(len(list_b)) if _ not in cols]:
            ans_prev[j] = ["<empty>" for _ in range(len(list_b[j]))]
            ans_upd[j] = list_b[j]
    return ans_prev, ans_upd


if __name__ == "__main__":
    xs = ["< version > 1 . 3 . 3 < / version >".split(),
          "< version > 1 . 3 . 4 - SNAPSHOT < / version >".split(),
          "shortVersion = ' 4 . 1 . 8'".split()]

    ys = ["< version > 1 . 3 . 3 - SNAPSHOT < / version >".split(),
          "< version > 1 . 3 . 4 < / version >".split(),
          "shortVersion = ' 4 . 1 . 9'".split()]
    for a, b in zip(xs, ys):
        al = edit_distance_alignment(a, b)
        print("==EXAMPLE==")
        print(a)
        print(b)
        print("==ALIGNMENT==")
        print(al[0])
        print(al[1])
        print()

    xs = [[['-', 'extra', '-', 'android', '-', 'support'], ['-', 'platform', '-', 'tools'], ['-', 'tools'],
           ['-', '-', 'build', '-', 'tools', '-', '23', '.', '0', '.', '3'], ['-', '-', 'android', '-', '23']],

          [['+', '-', 'build', '-', 'tools', '-', '24', '.', '0', '.', '0'], ['+', '-', 'android', '-', '24']],

          [],

          [["-", "version", "=", "1"]],

          [["-", "version", "=", "1"], ["-", "version", "=", "1"], ["-", "version", "=", "1"]]]

    ys = [[['+', '-', 'build', '-', 'tools', '-', '24', '.', '0', '.', '0'], ['+', '-', 'android', '-', '24']],

          [['-', 'extra', '-', 'android', '-', 'support'], ['-', 'platform', '-', 'tools'], ['-', 'tools'],
           ['-', '-', 'build', '-', 'tools', '-', '23', '.', '0', '.', '3'], ['-', '-', 'android', '-', '23']],

          [["+", "version", "=", "1"]],

          [],

          [["+", "version", "=", "1"], ["+", "version", "=", "1"], ["+", "version", "=", "1"]]]

    for list_a, list_b in zip(xs, ys):
        print("==EXAMPLE==")
        print(list_a)
        print(list_b)
        print("==ALIGNMENT==")
        ans_prev, ans_upd = align_lists(list_a, list_b)
        print(ans_prev)
        print(ans_upd)