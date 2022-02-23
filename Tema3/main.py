import re
import numpy as np
import copy
from functools import reduce

# Method 2 (search for max)
def parse_file_to_structure(name):
    print(name)
    matrix_size = 0

    data_regex = re.compile("\d*.?\d*, \d*, \d*")
    size_regex = re.compile("\d+")

    for line in open(name, 'r'):
        line = line.strip()
        # Daca e linie cu date de adaugat in matrice (ex. 506, 0, 0)
        if re.match(data_regex, line):
            #### TODO aici vezi daca are prea multe decimale ce faci
            parsed_line = [float(x.strip()) for x in line.split(',')]
            parsed_line[1:] = [int(e) for e in parsed_line[1:]]

            if parsed_line[-1]>parsed_line[-2]:
                print("Matricea nu este superior inferioara")
                exit()

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
    # print(rare_matrix)
    # print(sum([len(e) for e in rare_matrix]))
    return rare_matrix

#### TODO VEZI DACA NU E MAI BINE SA FACI SORTARE + CAUTARE BINARA
def add_matrixes(A,B):
    A=copy.deepcopy(A)
    for l_index,line in enumerate(B):
        for c_index,element in enumerate(line):
            found=0
            #print("A:",A[l_index])
            #print("B:",B[l_index])

            for a_index,a_tuple in enumerate(A[l_index]):
                if a_tuple[1]==element[1]:
                    A[l_index][a_index][0]+=element[0]
                    found=1
                    break
            if not found:
                A[l_index].append(element)
    return A


def compare_two_matrixes(A, B):
    for i in range(len(A)):
        #### TODO vezi daca nu trebuie comparat termen cu termen
        if sorted(A[i])!=sorted(B[i]):
            print("The two matrixes are not equal")
            return False
    return True


def get_dot_product(a, b):
    a=[e for e in a]
    a=a+b
    a.sort(key=lambda e:e[-1])
    sum=0
    for i in range(len(a)-1):
        if a[i][1]==a[i+1][1]:
            sum+=a[i][0]*a[i+1][0]
    return sum


# 1. For any integer n, A^{n} is symmetric if A is symmetric.
# 2. DACA MATRICEA E SIMETRICA IN LOC SA INMULTESTI LINIE*COLOANA POTI INMULTI LINIE*LINIE
def square_simetric_sparse_matrix(A):
    print("Starting multiplying matrixes")
    A_squared= rare_matrix = [[] for _ in range(len(A))]
    for i in range(len(A)):
        if i%100==0:
            #print(A_squared[:i])
            print(i)
        line=[e for e in A[i]]

        line_from_column=[]
        for index,l in enumerate(A[i+1:]):
            for element in l:
                if element[1]==i:
                    line_from_column.append([element[0],index+i+1])
        line+=line_from_column

        for j in range(i+1):
            column_from_line=[e for e in A[j]]
            column_from_column=[]
            for index,l in enumerate(A[j+1:]):
                for element in l:
                    if element[1]==j:
                        column_from_line.append([element[0],index+j+1])


            #column_from_column=[[e[0][0],index] for index,e in enumerate(A[j+1:]) if 0 in [f[1] for f in e]]# ]
            column=column_from_line+column_from_column
            term=get_dot_product(line, column)
            if term!=0:
                A_squared[i].append([term,j])

    return A_squared


if __name__ == '__main__':
    print("Tema 3")
    A=parse_file_to_structure('a.txt')
    B=parse_file_to_structure('b.txt')
    A_plus_B_computed=add_matrixes(A,B)
    A_plus_B_parsed=parse_file_to_structure('a_plus_b.txt')
    print("Are the two matrixes equal:", compare_two_matrixes(A_plus_B_computed, A_plus_B_parsed))
    A_ori_A_parsed=parse_file_to_structure('a_ori_a.txt')

    #Tentativa sortare
    print("Sorting")
    for i in range(len(A)):
        if i%100==0:
            print(i)
        A[i].sort(key=lambda e:e[1])
        A_ori_A_parsed[i].sort(key=lambda e:e[1])

    A_squared=square_simetric_sparse_matrix(A)

    print("Sorting after solving")
    for i in range(len(A)):
        if i%100==0:
            print(i)
        A_squared[i].sort(key=lambda e:e[1])
        A_ori_A_parsed[i].sort(key=lambda e:e[1])
    compare_two_matrixes(A_squared, A_ori_A_parsed)
