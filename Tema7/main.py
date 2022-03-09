import random


def f1(x):
    return (x - 1) * (x - 2) * (x - 3)


def f2(x):
    return (x - 2 / 3) * (x - 1 / 7) * (x + 1) * (x - 3 / 2)


def f3(x):
    return (x - 1) * (x - 1 / 2) * (x - 3) * (x - 1 / 4)


def f4(x):
    return ((x - 1) ** 2) * ((x - 2) ** 2)


f1_params = [1.0, -6.0, 11.0, -6.0]
f2_params = [42.0, -55.0, -42.0, 49.0, -6.0]
f3_params = [8.0, -38.0, 49.0, -22.0, 3.0]
f4_params = [1.0, -6.0, 13.0, -12.0, 4.0]


def horner(coeff, value):
    res = coeff[0]
    for i in range(1, len(coeff)):
        res = coeff[i] + res * value
    return res

def check_precision(value,eps):
    if abs(value)<eps:
        if value<0:
            return -eps
        return eps
    return value

def compute_polynoms(coeff, x):
    P_x_k = horner(coeff, x)
    P_x_plus_P_x = horner(coeff, x + P_x_k)
    P_x_minus_P_x = horner(coeff, x - P_x_k)
    y = x - (2 * (P_x_k ** 2)) / check_precision((P_x_plus_P_x - P_x_minus_P_x),eps)
    P_y_k = horner(coeff, y)
    return P_x_k, P_x_plus_P_x, P_x_minus_P_x, y, P_y_k

def element_in_set(value,set):
    for e in set:
        if abs(value-e)<eps:
            return True
    return False


def dehghan_method(coeff, fun, k_max,nr_points):
    A = max([abs(e) for e in coeff])
    R = (abs(coeff[0]) + A) / abs(coeff[0])
    root_set=set()
    for i in range(nr_points):
        x=random.uniform(-R,R)
        k=0
        P_x_k, P_x_plus_P_x, P_x_minus_P_x, y, P_y_k = compute_polynoms(coeff, x)
        delta_x_k = (2 * P_x_k * (P_x_k + P_y_k)) / check_precision((P_x_plus_P_x - P_x_minus_P_x), eps)
        while eps <= abs(delta_x_k) <= 10 ** 8 and k <= k_max:
            if abs(P_x_k) <= eps / 10:
                delta_x_k = 0
            else:
                P_x_k, P_x_plus_P_x, P_x_minus_P_x, y, P_y_k = compute_polynoms(coeff, x)
                delta_x_k = (2 * P_x_k * (P_x_k + P_y_k)) / check_precision((P_x_plus_P_x - P_x_minus_P_x),eps)
                x = x - delta_x_k
                k += 1
        else:
            if abs(P_x_k) <= eps / 10:
                delta_x_k = 0
            else:
                P_x_k, P_x_plus_P_x, P_x_minus_P_x, y, P_y_k = compute_polynoms(coeff, x)
                delta_x_k = (2 * P_x_k * (P_x_k + P_y_k)) / check_precision((P_x_plus_P_x - P_x_minus_P_x), eps)
                x = x - delta_x_k
                k += 1

        if abs(delta_x_k) < eps:
            if not element_in_set(x,root_set):
                root_set.add(x)
                print("Root:",x)
        else:
            pass
            #print("Divergenta")

    with open(f'{fun.__name__}roots.txt','w') as fd:
        for e in root_set:
            fd.write(str(e)+'\n')




if __name__ == '__main__':
    print("Tema 7")
    global eps
    eps=10**-10
    dehghan_method(f1_params, f1, 10000,100)
    dehghan_method(f2_params, f2, 10000,100)
    dehghan_method(f3_params, f3, 10000,100)
    dehghan_method(f4_params, f4, 10000,1000)