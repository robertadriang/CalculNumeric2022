import numpy as np


def define_parameters(n, eps, A, b):
    print(n)
    print(eps)
    print(A)
    print(b)


def search_pivot(matrix, l):
    return np.argmax(abs(matrix[l:, l])) + l


def swap_lines(matrix, l1, l2):
    matrix[[l1, l2]] = matrix[[l2, l1]]
    # print(matrix)
    return matrix


def make_superior_triangular(matrix):
    for i in range(len(matrix)):
        for j in range(i):
            matrix[i, j] = 0
    return matrix


def solve_triangular_system(matrix):
    results = [matrix[-1][-1] / matrix[-1][-2]]
    for i in range(len(matrix) - 2, -1, -1):
        results.insert(0, (matrix[i][-1] - sum(matrix[i][i + 1:-1] * results)) / matrix[i][i])
    return results


# Fie xGauss solut¸ia aproximativ˘a calculat˘a. S˘a se verifice solut¸ia afi¸sˆand
# urm˘atoarea norm˘a:
# |AinitxGauss − binit||2
def check_solution(A_init, b_init, solution, eps):
    Ax = np.sum(A_init * solution, axis=1)
    Ax_b = Ax - b_init.flatten()
    return np.linalg.norm(Ax_b, 2) < eps


def check_with_numpy(A_init, b_init, my_solution, eps):
    numpy_solution = np.linalg.solve(A_init, b_init.flatten())
    first_norm = np.linalg.norm(my_solution - numpy_solution, 2)
    A_inverse = np.linalg.inv(A_init)
    # A-1*b
    A_inverse_b = np.sum(A_inverse * b_init.flatten(), axis=1)
    second_norm = np.linalg.norm(my_solution - A_inverse_b, 2)
    if first_norm < eps and second_norm < eps:
        return True
    return False


def g_e_p_p(n, eps, A, b):
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
        print("A_B at while start", A_b)
        for i in range(l + 1, n):

            A_b[i, l] = A_b[i, l] / A_b[l, l]
            print("After:", A_b)
            for j in range(l + 1, n + 1):
                A_b[i, j] = A_b[i, j] - A_b[i, l] * A_b[l, j]
        l += 1
        line_to_swap = search_pivot(A_b, l)
        print("A_b before:", A_b)
        print("Line_to_swap:", line_to_swap)
        print("A_b after:", A_b)
        if line_to_swap != l:
            A_b = swap_lines(A_b, l, line_to_swap)

    if abs(A_b[l, l]) <= eps:
        print("Matricea este singulara")
    else:
        A_b = make_superior_triangular(A_b)
        solution = solve_triangular_system(A_b)
        print("Solution of triangular system:", solution)
        print(check_solution(A, b, solution, eps))
        print(check_with_numpy(A, b, solution, eps))


if __name__ == '__main__':
    print("Tema 2")
    # define_parameters(3,10**-6,np.matrix([[1,-1,1],[-6,1,-1],[3,1,1]]),np.array([2,3,4]))
    g_e_p_p(3, 10 ** -6, np.array([[2, 0, 1], [0, 2, 1], [4, 4, 6]]), np.array([[5], [1], [14]]))
    # g_e_p_p(3, 10 ** -6, np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [4.0, 4.0, 6.0]]), np.array([[5.0], [1.0], [14.0]]))
    # g_e_p_p(4, 10 ** -6, np.matrix([[0.02, 0.01, 0,0], [1, 2, 1,0], [0, 1, 2,1], [0, 0, 100, 200]]), np.array([[0.02], [1], [4],[800]]))
