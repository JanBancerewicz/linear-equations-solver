import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg
import time

from parameters import a1,a2,a3,f

def generate_matrix(N, new_a1=a1, new_a2=a2, new_a3=a3):
    # tworzenie macierzy A
    main_diag = np.diag(np.array([np.float64(new_a1)]*N), k=0)
    A = main_diag

    # a2
    a2_diag_upper = np.diag(np.array([np.float64(new_a2)]*(N-1)), k=1)
    A += a2_diag_upper
    a2_diag_lower = np.diag(np.array([np.float64(new_a2)]*(N-1)), k=-1)
    A += a2_diag_lower

    # a3
    a3_diag_upper = np.diag(np.array([np.float64(new_a3)]*(N-2)), k=2)
    A += a3_diag_upper
    a3_diag_lower = np.diag(np.array([np.float64(new_a3)]*(N-2)), k=-2)
    A += a3_diag_lower

    return A


def print_things(x, norm, calc_time, N, iters=0, strr="", matrix="A"):

    print(strr + f" dla N: {N} oraz macierzy {matrix}")
    print("x = ",x)
    if norm.dtype == np.float64 and norm.size > 1:
        print("norma = ", norm[-1])
    else:
        print("norma = ", norm)
    print("liczba iteracji = ", iters) if iters != 0 else None
    print("czas wykonania = ", calc_time)
    print()


def solve_direct(N, matrix="A", write=True):
    if matrix == "C": # zasymuluj macierz C
        a1 = 3
    else:
        a1 = 5

    A = generate_matrix(N, a1)
    b = np.array([np.sin(n * (np.float64(f + 1))) for n in range(1, N+1)])

    start = time.time()
    x = np.linalg.solve(A, b)
    calc_time = time.time() - start
    norm = np.linalg.norm(A @ x - b) # liczenie normy residuum
    if write:
        print(f"dokladne rozwiazanie dla N: {N} oraz macierzy {matrix}")
        print("x = ", x)
        print("norma = ", norm)
        print("czas wykonania = ", calc_time)
        print()

        f1 = open("data/direct/solution"+ "_"+ matrix+".txt", "w")
        f1.write(f"[Reference only] Direct solution from numpy.linalg.solve() for N: {N} and matrix {matrix};\t")
        f1.write("norm: " + str(norm) + ";\t" + "time: " + str(round(calc_time, 5)) + "\n")
        for item in x:
            f1.write("%s\n" % item)
        f1.close()

    return x