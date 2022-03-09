import numpy as np
import math
import copy


def check_precision(a, eps):
    if abs(a) < abs(eps):
        return 0
    return a


def is_diagonal_matrix(x):
    return np.count_nonzero(x - np.diag(np.diagonal(x))) == 0


def get_p_q_indexes(A):
    max_abs = 0
    max_i = 0
    max_j = 0
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            if abs(A[i][j]) > max_abs:
                max_abs = abs(A[i][j])
                max_i = i
                max_j = j
    return max_i, max_j


def get_theta_c_s_t(A, p, q):
    alpha = (A[p][p] - A[q][q]) / (2 * A[p][q])
    if alpha >= 0:
        t = -alpha + (alpha ** 2 + 1) ** (1 / 2)
    else:
        t = -alpha - (alpha ** 2 + 1) ** (1 / 2)
    c = 1 / ((1 + t ** 2) ** (1 / 2))
    s = t / ((1 + t ** 2) ** (1 / 2))
    theta = math.atan(t)
    return theta, c, s, t


### Deprecated
def construct_R_p_q(size, p, q, c, s):
    R_p_q = np.identity(size)
    R_p_q[p][p] = c
    R_p_q[q][q] = c
    R_p_q[p][q] = s
    R_p_q[q][p] = -s
    return R_p_q


def construct_new_A(A, p, q, c, s, t, eps):
    for j in range(0, len(A)):
        if j != p and j != q:
            A[p][j] = check_precision(c * A[p][j] + s * A[q][j], eps)
    for j in range(0, len(A)):
        if j != p and j != q:
            A[q][j] = A[j][q] = check_precision(-s * A[j][p] + c * A[q][j], eps)
    for j in range(0, len(A)):
        if j != p and j != q:
            A[j][p] = A[p][j]
    A[p][p] = check_precision(A[p][p] + t * A[p][q], eps)
    A[q][q] = check_precision(A[q][q] - t * A[p][q], eps)
    A[p][q] = 0
    A[q][p] = 0
    return A


def construct_new_U(U, p, q, c, s, eps):
    new_U = copy.deepcopy(U)
    for i in range(len(U)):
        new_U[i][p] = check_precision((c * new_U[i][p]) + (s * new_U[i][q]), eps)
        new_U[i][q] = check_precision((-s * U[i][p]) + (c * new_U[i][q]), eps)
    return new_U


def custom_sum(lambda_J, lambda_b):
    # J - Calculat cu Jacobi
    # b-  biblioteca
    sum = 0
    for ej in lambda_J:
        min = abs(ej - lambda_b[0])
        for ei in lambda_b:
            if abs(ej - ei) < min:
                min = abs(ej - ei)
        sum += min
    print("Sum of min differences:", sum)


def algorithm(A, eps):
    A_initial = copy.deepcopy(A)
    k = 0
    k_max = 100
    U = np.identity(len(A), dtype=float)
    p, q = get_p_q_indexes(A)
    theta, c, s, t = get_theta_c_s_t(A, p, q)
    # print(p, q)
    # print(theta, c, s, t)
    # print(c ** 2 + s ** 2)
    print(A)
    while not is_diagonal_matrix(A) and k <= k_max:
        print("K:", k)
        R_p_q = construct_R_p_q(len(A), p, q, c, s)
        # print("R_p_q", R_p_q)
        A = construct_new_A(A, p, q, c, s, t, eps)
        U = construct_new_U(U, p, q, c, s, eps)
        # print('CUSTOM_U:\n',U)
        # A = np.matmul((np.matmul(R_p_q, A)), np.transpose(R_p_q))
        # for i in range(len(A)):
        #     for j in range(len(A[0])):
        #         A[i][j]=check_precision(A[i][j],eps)
        # U = np.matmul(U, np.transpose(R_p_q))
        # print("U:\n",U)
        # for i in range(len(U)):
        #     for j in range(len(U[0])):
        #         U[i][j]=check_precision(U[i][j],eps)
        p, q = get_p_q_indexes(A)
        theta, c, s, t = get_theta_c_s_t(A, p, q)
        k += 1
    ### Valori proprii
    print('A:', A)
    print('Valori proprii A:', np.diagonal(A))
    ### Aproximarea vectorilor proprii
    print('U:', U)
    print("Steps:", k)

    np.set_printoptions(suppress=True)
    # print('A_init*U', np.matmul(A_initial, U))
    # print('U*Lambda', np.matmul(U, A))
    print("Norm:", np.linalg.norm(np.matmul(A_initial, U) - np.matmul(U, A), np.inf))
    eigenvalues_np, eigenvectors_np = np.linalg.eigh(A_initial)
    print("Eigenvalues:", eigenvalues_np)
    print("Eigenvectors:\n", eigenvectors_np)
    custom_sum(np.diagonal(A), eigenvalues_np)
    u, s, v_transpose = np.linalg.svd(A_initial)
    print("u:", u)
    print('s:', s)
    print('v_transpose:', v_transpose)

    print('Valorile singule ale matricei:', s)
    print('Rangul matricii:', np.count_nonzero(s[abs(s) >= eps]))
    print('Numarul de conditionare:', np.max(s) / np.min(s[abs(s) >= eps]))
    print('Pseudoinversa Moore-Penrose NP:\n', np.linalg.pinv(A_initial))

    S_i = np.zeros([len(u), len(v_transpose)])
    np.fill_diagonal(S_i, [0 if check_precision(e, eps) == 0 else 1 / e for e in s])
    A_i = np.matmul(np.matmul(np.transpose(v_transpose), S_i), np.transpose(u))
    print('Pseudoinversa Moore-Penrose calculata:\n', A_i)

    A_transpose = np.transpose(A_initial)
    A_t_A = np.matmul(A_transpose, A_initial)
    A_t_A_inverse = np.linalg.pinv(A_t_A)
    A_j = np.matmul(A_t_A_inverse, A_transpose)
    print("A_j:", A_j)
    print("Norm:", np.linalg.norm(A_i - A_j, 1))
    # A_j = np.matmult(np.linalg.inv(np.matmul(np.transpose(A_initial), A_initial)), np.transpose(A_initial))
    # print('Matricea pseudo inversa:', A_j)


if __name__ == '__main__':
    print("Tema 5")
    A = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])  # Merge OK
    # A = np.array([[1.0, 1.0, 2.0], [1.0, 1.0, 2.0], [2.0, 2.0, 2.0]])
    # A = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    # A = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    #A = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0], [4.0, 5.0, 6.0, 7.0]])
    algorithm(A, eps=10 ** -7)
