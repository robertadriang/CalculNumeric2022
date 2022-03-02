import math
import random
from decimal import Decimal


# Exercise 1
def machine_precision():
    u = 1
    count = 0
    while 1 + u != 1:
        u /= 10
        count += 1
        # print(f"U:{u}, Count:{count}, 1+u={1+u}")
    return u * 10, count - 1


# Exercise 2
def custom_addition(u):
    a = 1.0
    b = u / 10
    c = u / 10
    print("(a+b)+c = ", (a + b) + c)
    print("a+(b+c) = ", (a + (b + c)))


def custom_product():
    while True:
        a = random.random()
        b = random.random()
        c = random.random()

        product_1 = (a * b) * c
        product_2 = a * (b * c)
        if product_1 != product_2:
            print("a=", a)
            print("b=", b)
            print("c=", c)
            print('(a*b)*c = {:.20}'.format(product_1))
            print('a*(b*c) = {:.20}'.format(product_2))
            break


# Exercise 3
### Parse the given text file and return a dictionary of params
def parse_file(file):
    with open('date_T1.txt', 'r') as fp:
        lines = fp.readlines()
        available_functions = ['Sinus', 'Cosinus', 'Ln']
        available_terms = ['a', 'b']

        current_function = None
        current_term = None

        functions_params = dict()

        for line in lines:
            if line.strip() in available_functions:
                current_function = line.strip()
                functions_params[current_function] = dict()
                continue
            elif line.strip() in available_terms:
                current_term = line.strip()
                functions_params[current_function][current_term] = []
                continue
            else:
                functions_params[current_function][current_term].append(Decimal(line.strip()))
        return functions_params


### Compute Q or P polynom based on
# y - the input param
# function - Sinus/Cosinus/Ln
# polynom_type (P or Q)
def compute_polynom(y, function, polynom_type):
    global p
    if polynom_type == 'P_4':
        return p[function]['a'][0] + y * (p[function]['a'][1] + y * (
                p[function]['a'][2] + y * (p[function]['a'][3] + y * p[function]['a'][4])))
    elif polynom_type == 'Q_4':
        return p[function]['b'][0] + y * (p[function]['b'][1] + y * (
                p[function]['b'][2] + y * (p[function]['b'][3] + y * p[function]['b'][4])))


### When a numbers is between -(10^-12) and (10^-12) replace it with -(10^-12) or (10^-12)
def replace_small_numbers(a):
    if abs(a) < 10 ** (-12):
        if a < 0:
            return -1 * (10 ** (-12))
        return 10 ** (-12)
    return a


def sinus_approximation(x):
    if x < -1 or x > 1:
        print("Error! -1 <= x<= 1 for sinus")
        return
    return x * compute_polynom(x * x, 'Sinus', 'P_4') / replace_small_numbers(compute_polynom(x * x, 'Sinus', 'Q_4'))


def cosinus_approximation(x):
    if x < -1 or x > 1:
        print("Error! -1 <= x<= 1 for cosinus")
        return
    return compute_polynom(x * x, 'Cosinus', 'P_4') / replace_small_numbers(compute_polynom(x * x, 'Cosinus', 'Q_4'))


def ln_approximation(x):
    if x < 1 / (2 ** (1 / 2)) or x > 2 ** (1 / 2):
        print("Error! 1/sqrt(2) <= x<= sqrt(2) for ln")
        return
    z = Decimal((x - 1) / replace_small_numbers(x + 1))
    return z * compute_polynom(z * z, 'Ln', 'P_4') / replace_small_numbers(compute_polynom(z * z, 'Ln', 'Q_4'))


def sin_stats(x):
    library_result = math.sin(Decimal(math.pi * 1 / 4 * x))
    approximated_result = sinus_approximation(Decimal(x))
    print(f"[SIN] Input argument: {x}")
    print("[SIN]Result from library=", library_result)
    print("[SIN] Approximated result=", approximated_result)
    print("[SIN] Difference between results=", abs(Decimal(library_result) - approximated_result))
    print('')


def cos_stats(x):
    library_result = math.cos(Decimal(math.pi * 1 / 4 * x))
    approximated_result = cosinus_approximation(Decimal(x))
    print(f"[COS] Input argument: {x}")
    print("[COS]Result from library=", library_result)
    print("[COS] Approximated result=", approximated_result)
    print("[COS] Difference between results=", abs(Decimal(library_result) - approximated_result))
    print('')


def ln_stats(x):
    library_result = math.log(1.1)
    approximated_result = ln_approximation(1.1)
    print(f"[LN] Input argument: {x}")
    print("[LN]Result from library=", library_result)
    print("[LN] Approximated result=", approximated_result)
    print("[LN] Difference between results=", abs(Decimal(library_result) - approximated_result))
    print('')


if __name__ == '__main__':
    global p
    u, count = machine_precision()
    print(f"u:{u}, count={count}")
    custom_addition(u)
    custom_product()
    p = parse_file('date_T1.txt')
    #print(p)

    x = 0.66
    sin_stats(x)
    cos_stats(x)
    x = 1.23
    ln_stats(x)
