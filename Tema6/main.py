import math
import random
import numpy as np


def f1(x):
    return x ** 2 - 12 * x + 30


def f2(x):
    return math.sin(x) - math.cos(x)


def f3(x):
    return x ** 3 - 3 * x + 15


def generate_points_from_interval(start, end, n,decimals):
    result_set = set()
    result_set.add(start)
    result_set.add(end)
    while len(result_set) <= n:
        result_set.add(round(random.uniform(start, end),ndigits=decimals))
    return list(sorted(list(result_set)))


def generate_aitken_schema(x, y,eps):
    print(x)
    print(y)
    aitken_schema = np.zeros((len(x), len(y) - 1), dtype=float)
    print(aitken_schema)

    # First step in aitken schema:
    for i in range(1, len(aitken_schema)):
        aitken_schema[i][0] = (y[i] - y[i - 1]) / (x[i] - x[i - 1])
    print(aitken_schema)
    # Restul pasilor
    for j in range(1, len(aitken_schema[0])):
        for i in range(j + 1, len(aitken_schema)):
            #### AICI E PROBABIL O EROARE :(
            aitken_schema[i][j] = check_precision((aitken_schema[i][j - 1] - aitken_schema[i - 1][j - 1]) / (x[i] - x[i-j-1]),eps)
    print(aitken_schema)
    return np.insert(np.diagonal(aitken_schema, -1), 0, y[0])


def compute_L_polynom(value, x_array, y_array,eps=10**-8):
    Ln = y_array[0]
    right_term = 1
    for i in range(1, len(y_array)):
        right_term *= (value - x_array[i - 1])
        Ln += check_precision(y_array[i] * right_term,eps)
    return Ln

def check_precision(a, eps):
    if abs(a) < abs(eps):
        return 0
    return a


# utilizand forma Newton a polinomului de interpolare Lagrange ¸si schema
# lui Aitken de calcul al diferent¸elor divizate; sa se afiseze Ln(¯x) si
# |Ln(¯x) − f(¯x)|;
if __name__ == '__main__':
    print("Tema 6")

    precision=6
    x_1_0 = 1
    x_1_n = 5

    print(f1(x_1_0))
    points = generate_points_from_interval(x_1_0, x_1_n, 100,decimals=precision)
    #points = [1, 2, 3, 4, 5]
    results = [f1(e) for e in points]
    new_y = generate_aitken_schema(points, results,eps=10**-precision)
    print(new_y)

    check_points=[i*0.2 for i in range(20)]
    for i in check_points:
        print("I:", i)
        approximated= compute_L_polynom(i, points, new_y,eps=10**-precision)
        print("Aproximated:",approximated)
        print("Function result:", f1(i))
        print("Error:", abs(approximated - f1(i)))

    x_2_0 = 0
    x_2_n = 1.5

    # print(f2(x_2_0))
    # points = generate_points_from_interval(x_2_0, x_2_n, 10)
    # #points = [1, 2, 3, 4]
    # results = [f2(e) for e in points]
    # new_y = generate_aitken_schema(points, results)
    # print(new_y)
    #
    # for i in points:
    #     print("I:", i)
    #     print("Aproximated:", compute_L_polynom(i, points, new_y))
    #     print("Function result:", f2(i))
    #     print("Error:", abs(compute_L_polynom(i, points, new_y) - f2(i)))

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
