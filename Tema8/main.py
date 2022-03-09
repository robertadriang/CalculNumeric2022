import math
import random


def f1(x):
    return (1 / 3) * (x ** 3) - (2 * (x ** 2)) + 2 * x + 3


def f2(x):
    return x ** 2 + math.sin(x)


def f3(x):
    return x ** 4 - 6 * (x ** 3) + 13 * (x ** 2) - 12 * x + 4


### First derivative decorators (G1,G2)
def fd_1_builder(f):
    def fd_1(x, h):
        return (3 * f(x) - 4 * f(x - h) + f(x - 2 * h)) / (2 * h)

    return fd_1


def fd_2_builder(f):
    def fd_2(x, h):
        return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h)

    return fd_2


### Second derivative decorators (
def sd_builder(f):
    def sd(x, h):
        return (-f(x + 2 * h) + 16 * f(x + h) - 30 * f(x) + 16 * f(x - h) - f(x - 2 * h)) / (12 * h)

    return sd


def steffensen_return_point(f, start, end, fi, fii, h, k_max):
    k = 0
    while True:
        x = random.uniform(start, end)
        fi_x = fi(x, h)
        delta_x = (fi_x ** 2) / (fi(x + fi_x, h) - fi_x)
        while abs(delta_x) >= eps and k <= k_max and abs(delta_x) <= 10 ** 8:
            fi_x = fi(x, h)
            if abs(fi(x + fi_x, h) - fi_x) <= eps:
                print("1.Checking second derivative for:", x)
                if fii(x, h) > 0:
                    return x
                else:
                    print("Second derivative <0")
                break
            delta_x = (fi_x ** 2) / (fi(x + fi_x, h) - fi_x)
            x = x - delta_x
            k += 1
        else:
            fi_x = fi(x, h)
            if abs(fi(x + fi_x, h) - fi_x) <= eps:
                print("2.Checking second derivative for:", x)
                if fii(x, h) > 0:
                    return x
                print("Second derivative <0")
                continue
            delta_x = (fi_x ** 2) / (fi(x + fi_x, h) - fi_x)
            x = x - delta_x
            k += 1

        if abs(delta_x) < eps:
            print("3.Checking second derivative for:", x)
            if fii(x, h) > 0:
                return x
            print("Second derivative <0")
            continue
        else:
            pass


if __name__ == '__main__':
    print("Tema 8")
    global eps
    eps = 10 ** -10

    print("Minimizing f1:")
    fi = fd_1_builder(f1)
    fii = sd_builder(f1)
    point = steffensen_return_point(f1, 0, 4, fi, fii, 10 ** -5, 300)
    print("Point found with G1:",point)
    print('\n')
    fi = fd_2_builder(f1)
    point = steffensen_return_point(f1, 0, 4, fi, fii, 10 ** -5, 300)
    print("Point found with G2:",point)
    print('\n\n\n')


    print("Minimizing f2:")
    fi = fd_1_builder(f2)
    fii = sd_builder(f2)
    point = steffensen_return_point(f2, -4, 4, fi, fii, 10 ** -5, 300)
    print("Point found with G1:",point)
    print('\n')
    fi = fd_2_builder(f2)
    point = steffensen_return_point(f2, -4, 4, fi, fii, 10 ** -5, 300)
    print("Point found with G2:",point)
    print('\n\n\n')

    print("Minimizing f3:")
    fi = fd_1_builder(f3)
    fii = sd_builder(f3)
    point = steffensen_return_point(f3, 0, 3, fi, fii, 10 ** -5, 300)
    print("Point found with G1:",point)
    print('\n')
    fi = fd_2_builder(f3)
    point = steffensen_return_point(f3, 0, 3, fi, fii, 10 ** -5, 300)
    print("Point found with G2:",point)
    print('\n\n\n')