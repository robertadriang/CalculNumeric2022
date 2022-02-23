import re
import numpy as np


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
    #print(rare_matrix)
    #print(sum([len(e) for e in rare_matrix]))
    return rare_matrix

#### TODO VEZI DACA NU E MAI BINE SA FACI SORTARE + CAUTARE BINARA
def add_matrixes(A,B):
    for l_index,line in enumerate(B):
        for c_index,element in enumerate(line):
            found=0
            for a_index,a_tuple in enumerate(A[l_index]):
                if a_tuple[1]==element[1]:
                    A[l_index][a_index][0]+=element[0]
                    found=1
                    break
            if not found:
                A[l_index].append(element)
    return A


def compare_two_matrixed(A,B):
    for i in range(len(A)):
        #### TODO vezi daca nu trebuie comparat termen cu termen
        if sorted(A[i])!=sorted(B[i]):
            print("The two matrixes are not equal")
            return False
    # ceva=[sorted(e) for e in A]
    # ceva_2=[sorted(e) for e in B]
    # print("@",ceva[0:5])
    # print("#",ceva_2[0:5])
    return True

if __name__ == '__main__':
    print("Tema 3")
    A=parse_file_to_structure('a.txt')
    B=parse_file_to_structure('b.txt')
    A_plus_B_computed=add_matrixes(A,B)
    A_plus_B_parsed=parse_file_to_structure('a_plus_b.txt')
    print("Are the two matrixes equal:",compare_two_matrixed(A_plus_B_computed, A_plus_B_parsed))
