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
    # print(A[p][p])
    # print(A[q][q])
    # print(A[p][q])
    # print(A[p][q])
    alpha = (A[p][p] - A[q][q]) / (2 * A[p][q])
    # print('@',alpha)
    if alpha >= 0:
        t = -alpha + (alpha ** 2 + 1) ** (1 / 2)
    else:
        t = -alpha - (alpha ** 2 + 1) ** (1 / 2)
    c = 1 / ((1 + t ** 2) ** (1 / 2))
    s = t / ((1 + t ** 2) ** (1 / 2))
    theta = math.atan(t)
    return theta, c, s, t


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
        new_U[i][p] = check_precision(c * U[i][p] + s * U[i][q], eps)
        new_U[i][q] = check_precision(-s * U[i][p] + c * U[i][q], eps)
    return new_U


def algorithm(A, eps):
    A_initial = copy.deepcopy(A)
    k = 0
    k_max = 100
    U = np.identity(len(A), dtype=float)
    p, q = get_p_q_indexes(A)
    theta, c, s, t = get_theta_c_s_t(A, p, q)
    print(p, q)
    print(theta, c, s, t)
    print(c ** 2 + s ** 2)
    print(A)
    while not is_diagonal_matrix(A) and k <= k_max:
        print("K:", k)
        # R_p_q = construct_R_p_q(len(A), p, q, c, s)
        # print("R_p_q", R_p_q)
        A = construct_new_A(A, p, q, c, s, t, eps)
        U = construct_new_U(U, p, q, c, s, eps)
        # A = np.matmul((np.matmul(R_p_q, A)), np.transpose(R_p_q))
        # for i in range(len(A)):
        #     for j in range(len(A[0])):
        #         A[i][j]=check_precision(A[i][j],eps)
        # U = np.matmul(U, np.transpose(R_p_q))
        # for i in range(len(U)):
        #     for j in range(len(U[0])):
        #         U[i][j]=check_precision(U[i][j],eps)
        p, q = get_p_q_indexes(A)
        # if not is_diagonal_matrix(A):
        theta, c, s, t = get_theta_c_s_t(A, p, q)
        # else:
        #    break
        k += 1
    print('A:', A)
    print('U:', U)
    print("Steps:", k)
    A_final = np.transpose(U) * A_initial * U


if __name__ == '__main__':
    print("Tema 5")
    # A = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    # A = np.array([[1.0, 1.0, 2.0], [1.0, 1.0, 2.0], [2.0, 2.0, 2.0]])
    # A = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    A = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    A = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0], [4.0, 5.0, 6.0, 7.0]])
    algorithm(A, eps=10 ** -6)
