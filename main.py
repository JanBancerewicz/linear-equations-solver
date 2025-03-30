import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import time

# N= 1299
# a1=5
# a2=-1
# a3=-1
# c=9 # przedostatnia cyfra
# d=9 # ostatnia cyfra
# e=0 # czwarta cyfra
# f=8 # trzecia cyfra

N= 1258
a1=5
a2=-1
a3=-1
c=5
d=8
e=0
f=8



# układ rownan a1 = 5
# a2 = -1
# a3 = -1

# Ax = b
# rozmiar macierzy: 1299 x 1299

# wartosci b: sin(n · (9))

# generate linear equations:
# 5 - - 0 0 0 0
# - 5 - - 0 0 0
# - - 5 - - 0 0
# 0 - - 5 - - 0
# 0 0 - - 5 - -




# tworzenie macierzy A
main_diag = np.diag(np.array([np.float64(a1)]*N), k=0)
A = main_diag

#a2
a2_diag_upper =np.diag(np.array([np.float64(a2)]*(N-1)), k=1)
A += a2_diag_upper
a2_diag_lower = np.diag(np.array([np.float64(a2)]*(N-1)), k=-1)
A += a2_diag_lower

#a3
a3_diag_upper =np.diag(np.array([np.float64(a3)]*(N-2)), k=2)
A += a3_diag_upper
a3_diag_lower = np.diag(np.array([np.float64(a3)]*(N-2)), k=-2)
A += a3_diag_lower

# print(A)

# tworzenie wektora b
b = np.array( [np.sin(n*(np.float64(f+1))) for n in range(1, N+1)] )

# print(b)


def jacobi_method(N, A, b, max_iter=1000, min_residuum=1e-12):
    start = time.time()
    x = np.ones(N, dtype=np.float64)

    D = np.diag(np.diag(A))  # macierz diagonalna
    L = np.tril(A, k=-1)  # macierz trójkątna dolna
    U = np.triu(A, k=1)  # macierz trójkątna górna

    D_inv = np.diag(1.0 / np.diag(D))  # odwrotność D, czyli odwrotność każdego elementu na diagonali
    M = -D_inv @ (L + U)  # macierz iteracyjna
    w = D_inv @ b  # wektor iteracyjny

    residuum = A @ x - b
    inorm = np.linalg.norm(residuum)
    r_norm = [inorm]
    iters = 0

    while inorm > min_residuum and iters < max_iter:
        # iteracja Jacobiego: x = Mx + w
        x = M @ x + w
        residuum = A @ x - b
        inorm = np.linalg.norm(residuum)
        r_norm.append(inorm)
        iters += 1

    calc_time = time.time() - start
    return x, np.array(r_norm), iters, calc_time


def gauss_seidel_method_good(N, A, b, max_iter=1000, min_residuum=1e-12):
    start = time.time()
    x = np.ones(N, dtype=np.float64)  # Wymuszamy float64

    # Rozkład macierzy A = L + D + U
    U = np.triu(A, k=1)
    L_D = np.tril(A)  # L + D (trójkąt dolny z diagonalą)

    # Inicjalizacja normy residuum
    residuum = A @ x - b
    inorm = np.linalg.norm(residuum)
    r_norm = [inorm]
    iters = 0

    while inorm > min_residuum and iters < max_iter:
        # Iteracja Gaussa-Seidla: x = (L + D)^{-1} @ (b - U @ x)
        for i in range(N): # podstawienie w przód
            sum_ = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i + 1:], x[i + 1:])
            x[i] = (b[i] - sum_) / A[i, i]

        residuum = A @ x - b
        inorm = np.linalg.norm(residuum)
        r_norm.append(inorm)
        iters += 1

    calc_time = time.time() - start
    return x, np.array(r_norm), iters, calc_time


def gauss_seidel_method_slow(N, A, b, max_iter=1000, min_residuum=1e-12):
    start = time.time()
    x = np.ones(N)

    D = np.diag(np.diag(A))
    U = np.triu(A, k=1)
    L = np.tril(A, k=-1)

    # Macierz T = D + L oraz wektor w = T^{-1}b
    T = D + L

    # liczenie normy residuum
    residuum = A @ x - b
    inorm = np.linalg.norm(residuum)
    r_norm = [inorm]
    iters = 0

    while ( inorm > min_residuum and iters < max_iter ):
        # Iteracja Gaussa-Seidla: x = T^{-1}(-Ux + b)
        x = np.linalg.solve(T, -U @ x + b)
        residuum = A @ x - b
        inorm = np.linalg.norm(residuum)
        r_norm.append(inorm)
        iters += 1

    calc_time = time.time() - start

    return x, np.array(r_norm), iters, calc_time


def gauss_seidel_method(N, A, b, max_iter=1000, min_residuum=1e-12):
    start = time.time()
    x = np.ones(N, dtype=np.float64)

    # wyznaczanie U i L+D
    U = np.triu(A, k=1)
    L_D = np.tril(A)  # L + D (trójkąt dolny z diagonalą)

    residuum = A @ x - b
    inorm = np.linalg.norm(residuum)
    r_norm = [inorm]
    iters = 0

    while inorm > min_residuum and iters < max_iter:
        # iteracja Gaussa-Seidla: x = (L + D)^{-1} @ (b - U @ x)
        right_side = b - U @ x  # prawa strona rownania
        x_new = np.zeros(N)

        for i in range(N): # podstawienie w przód - rozwiazanie układu równań
            x_new[i] = (right_side[i] - np.dot(L_D[i, :i], x_new[:i])) / L_D[i, i]
        x = x_new

        residuum = A @ x - b
        inorm = np.linalg.norm(residuum)
        r_norm.append(inorm)
        iters += 1

    calc_time = time.time() - start
    return x, np.array(r_norm), iters, calc_time



##################################### git #################################


###########
###########
# ---ZADANIE B---
# JACOBI
# x =  [ 2.17059088e-09  3.49923677e-09  5.12846630e-09 ... -2.11834752e-02
#  -1.98498487e-02 -6.72960960e-02]
# norma =  9.67757579006773e-07
# liczba iteracji =  78
# czas wykonania =  0.034583091735839844
#
# GAUSS-SEIDEL
# x =  [ 7.87125621e-09  1.12922911e-08  1.49310540e-08 ... -2.11834801e-02
#  -1.98498521e-02 -6.72960981e-02]
# norma =  9.379467108527376e-07
# liczba iteracji =  43
# czas wykonania =  0.5343911647796631

################################# moje ###################################

# JACOBI
# x =  [ 0.09197091 -0.13146997  0.17920602 ...  0.08755861 -0.04942018
#  -0.05146175]
# norma =  9.449223214239874e-13
# liczba iteracji =  140
# czas wykonania =  0.824000358581543
#
# GAUSS-SEIDEL
# x =  [ 0.09197091 -0.13146997  0.17920602 ...  0.08755861 -0.04942018
#  -0.05146175]
# norma =  9.58506155867791e-13
# liczba iteracji =  77
# czas wykonania =  3.4350228309631348

############################ moje new zad B ####################################
# JACOBI
# x =  [ 0.09197091 -0.13146997  0.17920602 ...  0.08755861 -0.04942018
#  -0.05146175]
# norma =  9.449223214239874e-13
# liczba iteracji =  140
# czas wykonania =  0.1600172519683838
#
# GAUSS-SEIDEL
# x =  [ 0.09197091 -0.13146997  0.17920602 ...  0.08755861 -0.04942018
#  -0.05146175]
# norma =  9.585205724301439e-13
# liczba iteracji =  77
# czas wykonania =  0.25699949264526367



x, norm, iters, calc_time = jacobi_method(N, A, b)
print("JACOBI")
print("x = ", x)
print("norma = ", norm[-1])
print("liczba iteracji = ", iters)
print("czas wykonania = ", calc_time)
print()

x, norm, iters, calc_time = gauss_seidel_method(N, A, b)
print("GAUSS-SEIDEL")
print("x = ", x)
print("norma = ", norm[-1])
print("liczba iteracji = ", iters)
print("czas wykonania = ", calc_time)
print()
###########

