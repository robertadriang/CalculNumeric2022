import numpy as np


### Extra info:
# Folosesc flatten pt ca vectorii coloana sunt cv de genul [[a],[b],[c]] si cand scad din ei [d,e,f] rezultatul ar fi o matrice
# TODO!!!!: Trebuie adaugate check-uri pe impartiri (Vezi pagina 2 final 3 inceput)


def search_pivot(matrix, l):
    return np.argmax(abs(matrix[l:, l])) + l


def swap_lines(matrix, l1, l2):
    matrix[[l1, l2]] = matrix[[l2, l1]]
    return matrix


def make_superior_triangular(matrix):
    for i in range(len(matrix)):
        for j in range(i):
            matrix[i, j] = 0
    return matrix


def solve_triangular_system(matrix):
    if abs(matrix[-1][-2]) > eps:
        results = [matrix[-1][-1] / matrix[-1][-2]]
    else:
        print("Nu se poate face impartirea in solve_triangular_system")
        exit()

    for i in range(len(matrix) - 2, -1, -1):
        if abs(matrix[i][i]) > eps:
            results.insert(0, (matrix[i][-1] - sum(matrix[i][i + 1:-1] * results)) / matrix[i][i])
        else:
            print("Nu se poate face impartirea in solve_triangular_system")
            exit()
    return results


# Prima norma (pagina 1)
def check_solution(A_init, b_init, solution):
    Ax = np.sum(A_init * solution, axis=1)
    Ax_b = Ax - b_init.flatten()
    return np.linalg.norm(Ax_b, 2) < eps


# Urmatoarele 2 norme (pagina 1)
def check_with_numpy(A_init, b_init, my_solution):
    numpy_solution = np.linalg.solve(A_init, b_init.flatten())
    first_norm = np.linalg.norm(my_solution - numpy_solution, 2)
    A_inverse = np.linalg.inv(A_init)
    A_inverse_b = np.sum(A_inverse * b_init.flatten(), axis=1)
    second_norm = np.linalg.norm(my_solution - A_inverse_b, 2)
    if first_norm < eps and second_norm < eps:
        return True
    return False


# Gauss extins pivotare partiala (primele 3 buline de pe pagina 1)
def g_e_p_p(n, A, b=None):
    print("Functia gauss extins cu pivotare partiala")
    print("Matrix size:", n)
    print("Precision:", eps)
    print("A:", A)
    print("b:", b)

    A_b = np.append(A, b, axis=1)
    A_b = A_b.astype(np.float64)

    l = 0
    line_to_swap = search_pivot(A_b, l)
    if line_to_swap != l:
        A_b = swap_lines(A_b, l, line_to_swap)

    while l < n - 1 and abs(A_b[l, l]) > eps:
        for i in range(l + 1, n):
            if abs(A_b[l, l]) > eps:
                A_b[i, l] = A_b[i, l] / A_b[l, l]
            else:
                print("Nu se poate face impartirea in g_e_p_p")
                exit()
            for j in range(l + 1, n + 1):
                A_b[i, j] = A_b[i, j] - A_b[i, l] * A_b[l, j]
        l += 1
        line_to_swap = search_pivot(A_b, l)
        if line_to_swap != l:
            A_b = swap_lines(A_b, l, line_to_swap)

    if abs(A_b[l, l]) <= eps:
        print("Matricea este singulara")
    else:
        A_b = make_superior_triangular(A_b)
        solution = solve_triangular_system(A_b)
        print("Solution of triangular system:", solution)
        print("Este prima norma mai mica decat precizia:", check_solution(A, b, solution))
        print("Sunt urmatoarele 2 norme mai mici decat precizia:", check_with_numpy(A, b, solution))


def merge_matrixes(A, B):
    # Adaug cate o coloana din B pe rand in A
    for i in range(len(B[0])):
        A = np.insert(A, len(A[0]), B[:, i], axis=1)w
    return A


# Ultima norma de pe pagina 1
def check_inverse(A, my_inverse):
    A_inverse = np.linalg.inv(A)
    norm = np.linalg.norm(my_inverse - A_inverse, 2)
    if norm < eps:
        return True
    return False


def compute_matrix_inverse(A):
    rows = len(A)
    columns = len(A[0])

    # Aici fac o matrice de zerouri de dimensiunea lui A, ii adaug 1 pe diagonala principala si fac merge intre A si matricea obtinuta (identitate)
    identity_matrix = np.zeros(A.shape, dtype=float)
    np.fill_diagonal(identity_matrix, 1.0)
    A_extended = merge_matrixes(A, identity_matrix)
    A_extended = A_extended.astype(np.float64)

    print("Matricea A extinsa:", A_extended)
    columns_extended = len(A_extended[0])

    # Am adaptat putin algorimul ca sa mearga si pt extins.
    # Diferenta e cam doar la indexii coloana
    l = 0
    line_to_swap = search_pivot(A_extended, l)
    if line_to_swap != l:
        A_extended = swap_lines(A_extended, l, line_to_swap)
    while l < rows - 1 and abs(A_extended[l, l]) > eps:
        for i in range(l + 1, rows):
            if abs(A_extended[l, l]) > eps:
                A_extended[i, l] = A_extended[i, l] / A_extended[l, l]
            else:
                print("Nu se poate face impartirea in compute_matrix_inverse")
                exit()
            for j in range(l + 1, columns_extended):
                A_extended[i, j] = A_extended[i, j] - A_extended[i, l] * A_extended[l, j]
        l += 1
        line_to_swap = search_pivot(A_extended, l)
        if line_to_swap != l:
            A_b = swap_lines(A_extended, l, line_to_swap)

    if abs(A_extended[l, l]) <= eps:
        print("Matricea este singulara")
    else:
        A_extended = make_superior_triangular(A_extended)
        inverse = []
        # Din matricea extinsa pastrez doar primele n coloane (unde n e nr de coloane initial al lui a)
        # si ii mai adaug cate o coloana dintre cele extinse (dupa ce am aplicat algoritmul de eliminare
        for i in range(len(identity_matrix[0])):
            matrix_to_solve = merge_matrixes(A_extended[:, :columns], A_extended[:, columns + i:columns + i + 1])
            inverse.append(solve_triangular_system(matrix_to_solve))

        # Eu adaug coloanele ca linii intai de aia fac transpusa la final
        inverse = np.transpose(np.array(inverse))
        print("Inversa matricii A:", inverse)
        print("Este norma dintre matricea calculata si cea din numpy mai mica decat precizia:",
              check_inverse(A, inverse))


if __name__ == '__main__':
    print("Tema 2")
    global eps
    eps = 10 ** -6
    g_e_p_p(3, np.array([[2, 0, 1], [0, 2, 1], [4, 4, 6]]), np.array([[5], [1], [14]]))
    compute_matrix_inverse(np.array([[2, 0, 1], [0, 2, 1], [4, 4, 6]]))
