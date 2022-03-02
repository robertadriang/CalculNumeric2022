import re
import numpy as np
import copy
import matplotlib.pyplot as plt


def check_precision(x, eps):
    if x < 0:
        if x > -eps:
            return -eps
    elif x > 0:
        if x < eps:
            return eps
    return x


############### NU MAI RETINEM DIAGONALA PRINCIPALA IN STRUCTURA RARA
def parse_file_to_structure(name):
    print(name)
    matrix_size = 0

    data_regex = re.compile("-?\d*.?\d* , \d* , \d*")
    size_regex = re.compile("\d+")

    for line in open(name, 'r'):
        line = line.strip()
        # Daca e linie cu date de adaugat in matrice (ex. 506, 0, 0)
        if re.match(data_regex, line):
            #### TODO aici vezi daca are prea multe decimale ce faci
            parsed_line = [float(x.strip()) for x in line.split(',')]
            parsed_line[1:] = [int(e) for e in parsed_line[1:]]

            if parsed_line[-1] > parsed_line[-2]:
                print("Matricea nu este superior inferioara")
                exit()

            if parsed_line[-1] == parsed_line[-2]:
                diagonal_vector[parsed_line[-1]] += parsed_line[0]
                continue

            # Daca nu avem niciun element pe acea linie(in matrice) cand parsam fisierul
            if rare_matrix[parsed_line[1]] == []:
                rare_matrix[parsed_line[1]].append([parsed_line[0], parsed_line[-1]])
            # Daca avem element pe acea linie (in matrice)
            else:
                found = 0
                # Daca avem elemente pe acea linie verificam daca
                # exista un element care sa aiba aceeasi pozitie (linie/coloana) adunam
                # la elementul deja existent altfel inseram unul nou
                for index, tuple in enumerate(rare_matrix[parsed_line[1]]):
                    if tuple[1] == parsed_line[-1]:
                        rare_matrix[parsed_line[1]][index][0] += parsed_line[0]
                        found = 1
                        break
                if not found:
                    rare_matrix[parsed_line[1]].append([parsed_line[0], parsed_line[-1]])

        elif re.match(size_regex, line):
            matrix_size = int(line.strip())
            rare_matrix = [[] for _ in range(matrix_size)]
            diagonal_vector = [0.0 for _ in range(matrix_size)]

    # print(rare_matrix)
    # print(sum([len(e) for e in rare_matrix]))
    return rare_matrix, diagonal_vector


def parse_result_vector(name):
    with open(name, 'r') as fd:
        size = fd.readline()
        result = []
        for line in fd:
            try:
                result.append(float(line.strip()))
            except ValueError:
                pass
    return result


def solve_with_jacobi(matrix, matrix_diag, b, eps=10 ** -6):
    x_step = [0 for _ in range(len(b))]
    delta_x = 1
    k_max = 10000
    k = 0

    while delta_x >= eps and k <= k_max and delta_x <= 10 ** 8:
        x_next_step = [0 for _ in range(len(b))]
        for l_index, line in enumerate(matrix):
            for element in line:
                x_next_step[l_index] += element[0] * x_step[element[1]]
                x_next_step[element[1]] += element[0] * x_step[l_index]

        for i in range(len(x_step)):
            x_next_step[i] = (b[i] - x_next_step[i]) / matrix_diag[i]

        delta_x_vector = []
        for i in range(len(x_next_step)):
            delta_x_vector.append(x_next_step[i] - x_step[i])

        sum = 0
        for i in range(len(delta_x_vector)):
            sum += delta_x_vector[i] * delta_x_vector[i]

        delta_x = sum ** (1 / 2)
        k += 1
        print(k, delta_x)
        x_step = [e for e in x_next_step]
    else:
        x_next_step = [0 for _ in range(len(b))]
        for l_index, line in enumerate(matrix):
            for element in line:
                x_next_step[l_index] += element[0] * x_step[element[1]]
                x_next_step[element[1]] += element[0] * x_step[l_index]

        for i in range(len(x_step)):
            x_next_step[i] = (b[i] - x_next_step[i]) / matrix_diag[i]

        delta_x_vector = []
        for i in range(len(x_next_step)):
            delta_x_vector.append(x_next_step[i] - x_step[i])

        sum = 0
        for i in range(len(delta_x_vector)):
            sum += delta_x_vector[i] * delta_x_vector[i]

        delta_x = sum ** (1 / 2)
        k += 1
        print(k, delta_x)
        x_step = [e for e in x_next_step]
    if delta_x <= eps:
        print("Nr pasi efectuati:", k)
        print(x_next_step)
        print(len(x_next_step))
        plt.hist(x_next_step, bins=100)
        plt.show()
        check_solution(matrix, matrix_diag, b, x_next_step, eps)
    else:
        print(delta_x)
        print("Divergenta")
        print(x_next_step)
        plt.hist(x_next_step, bins=100)
        plt.show()
        check_solution(matrix, matrix_diag, b, x_next_step, eps)


def get_dot_product(matrix_line, solution):
    sum = 0
    for element in matrix_line:
        sum += element[0] * solution[element[1]]
    return sum


def check_solution(A_triangle, A_diagonal, b, solution, eps):
    # Adaugam diagonala principala la matricea rara
    A_triangle = copy.deepcopy(A_triangle)
    for index, element in enumerate(A_diagonal):
        A_triangle[index].append([element, index])

    A_x_vector = []
    for i in range(len(A_triangle)):
        if i % 1000 == 0:
            print(f"{i} termeni calculati pentru inmultirea A*x_aproximat")
        # Luam integral linia curenta si cautam pe urmatoarele coloane termenii de pe acea linie
        line = [e for e in A_triangle[i]]
        line_from_column = []
        for index, l in enumerate(A_triangle[i + 1:]):
            for element in l:
                if element[1] == i:
                    line_from_column.append([element[0], index + i + 1])
        line += line_from_column
        A_x_vector.append(get_dot_product(line, solution))

    difference_vector = []
    for i in range(len(A_x_vector)):
        difference_vector.append(A_x_vector[i] - b[i])
    print("Norma INF:", np.linalg.norm(difference_vector, np.inf))


### TODO VEZI PRECIZIA
if __name__ == '__main__':
    print("Tema 4")

    #### 3 nu merge
    #### 4 nu merge
    A_1, A_1_diag = parse_file_to_structure('a_4.txt')
    b_1 = parse_result_vector('b_4.txt')
    for element in A_1_diag:
        if element == 0:
            print(A_1_diag)
            print("Element nenul in diagonala! Nu se poate rezolva cu Jacobi")
            exit()
    solve_with_jacobi(A_1, A_1_diag, b_1, 10 ** -8)
