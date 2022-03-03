import math
import random
import numpy as np
import matplotlib.pyplot as plt


def f1(x):
    return x ** 2 - 12 * x + 30


def f2(x):
    return round(math.sin(x) - math.cos(x),ndigits=decimals)


def f3(x):
    return x ** 3 - 3 * x + 15


def generate_points_from_interval(start, end, n):
    result_set = set()
    result_set.add(start)
    result_set.add(end)
    while len(result_set) <= n:
        result_set.add(random.uniform(start, end))
    return list(sorted(list(result_set)))

### Asa e dat in fisier
def generate_aitken_schema(x, y):
    aitken_schema = np.zeros((len(x), len(y) - 1), dtype=float)
    # First step in aitken schema:
    for i in range(1, len(aitken_schema)):
        aitken_schema[i][0] = (y[i] - y[i - 1])/(x[i] - x[i - 1])
    #print(aitken_schema)
    # Restul pasilor
    for j in range(1, len(aitken_schema[0])):
        for i in range(j + 1, len(aitken_schema)):
            aitken_schema[i][j] = (aitken_schema[i][j - 1] - aitken_schema[i - 1][j - 1]) / (x[i] - x[i - j - 1])
    #print(aitken_schema)
    return np.insert(np.diagonal(aitken_schema, -1), 0, y[0])

# Optimizare creier
def generate_aitken_schema_vector(x, y):
    aitken_schema = np.zeros((len(x), len(y) - 1), dtype=float)
    aitken_vector=[e for e in y]
    # First step in aitken schema:
    for i in range(1, len(aitken_schema)):
        aitken_vector[i] = (y[i] - y[i - 1])/(x[i] - x[i - 1])
    # Restul pasilor
    for j in range(1, len(aitken_schema[0])):
        for i in reversed(range(j + 1, len(aitken_schema))):
            aitken_vector[i] = (aitken_vector[i] - aitken_vector[i - 1]) / (x[i] - x[i - j - 1])
    return aitken_vector


def compute_L_polynom(value, x_array, y_array, eps=10 ** -8):
    Ln = y_array[0]
    right_term = 1
    for i in range(1, len(y_array)):
        right_term *= (value - x_array[i - 1])
        Ln += y_array[i] * right_term

    return Ln


def check_precision(a):
    if abs(a) < abs(eps):
        return 0
    return round(a, ndigits=decimals)


def test_function(start, end, nr_of_points, precision, function):
    global decimals, eps
    decimals = precision
    eps = 10 ** -precision
    points = generate_points_from_interval(start, end, nr_of_points)
    computed_results = [function(e) for e in points]
    # new_y = generate_aitken_schema(points, computed_results)
    # print("INITIAL:",new_y)
    new_y=generate_aitken_schema_vector(points,computed_results)
    approximated_results = []
    for index, i in enumerate(points):
        print("x:", i)
        approximated = compute_L_polynom(i, points, new_y)
        approximated_results.append(approximated)
        print("Aproximated:", approximated)
        print("Function result:", computed_results[index])
        print("Error:", abs(approximated - computed_results[index]))

    fig, ax = plt.subplots(2)
    ax[0].plot(points, computed_results)
    ax[1].plot(points, approximated_results)
    plt.show()


# utilizand forma Newton a polinomului de interpolare Lagrange ¸si schema
# lui Aitken de calcul al diferent¸elor divizate; sa se afiseze Ln(¯x) si
# |Ln(¯x) − f(¯x)|;
if __name__ == '__main__':
    print("Tema 6")
    #test_function(1.0,5.0,50,8,f1)
    #test_function(0,1.5,30,6,f2)
    test_function(0, 2, 50, 6, f3)

    # precision = 6
    # x_1_0 = 1
    # x_1_n = 5
    #
    # points = generate_points_from_interval(x_1_0, x_1_n, 200, decimals=precision)
    # # points = [1, 2, 3, 4, 5]
    # results = [f1(e) for e in points]
    # new_y = generate_aitken_schema(points, results, eps=10 ** -precision)
    # computed_results = []
    # approximated_results = []
    # for i in points:
    #     print("x:", i)
    #     approximated = compute_L_polynom(i, points, new_y, eps=10 ** -precision)
    #     approximated_results.append(approximated)
    #     print("Aproximated:", approximated)
    #     computed_results.append(f1(i))
    #     print("Function result:", f1(i))
    #     print("Error:", abs(approximated - f1(i)))
    #
    # fig, ax = plt.subplots(2)
    # ax[0].plot(points, computed_results)
    # ax[1].plot(points, approximated_results)
    # plt.show()
    x_2_0 = 0
    x_2_n = 1.5
    #
    # print(f2(x_2_0))
    # points = generate_points_from_interval(x_2_0, x_2_n, 100,decimals=precision)
    # #points = [1, 2, 3, 4, 5]
    # results = [f2(e) for e in points]
    # new_y = generate_aitken_schema(points, results,eps=10**-precision)
    # print(new_y)
    #
    #
    # for i in points:
    #     print("I:", i)
    #     approximated= compute_L_polynom(i, points, new_y,eps=10**-precision)
    #     print("Aproximated:",approximated)
    #     print("Function result:", f2(i))
    #     print("Error:", abs(approximated - f2(i)))

    x_3_0 = 0
    x_3_n = 2

    # print(f3(x_3_0))
    # points = generate_points_from_interval(x_3_0, x_3_n, 10)
    # #points = [1, 2, 3, 4]
    # results = [f3(e) for e in points]
    #
    # new_y = generate_aitken_schema(points, results)
    # print(new_y)
    #
    # for i in points:
    #     print("I:", i)
    #     print("Aproximated:", compute_L_polynom(i, points, new_y))
    #     print("Function result:", f3(i))
    #     print("Error:", abs(compute_L_polynom(i, points, new_y) - f3(i)))
