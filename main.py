import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg
import time

N= 1299
a1=5
a2=-1
a3=-1
c=9 # przedostatnia cyfra
d=9 # ostatnia cyfra
e=0 # czwarta cyfra
f=8 # trzecia cyfra



# N= 1258
# a1=5
# a2=-1
# a3=-1
# c=5
# d=8
# e=0
# f=8



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


# tworzenie macierzy C
a1c = 3
main_diag = np.diag(np.array([np.float64(a1c)]*N), k=0)
C = main_diag

#c2
c2_diag_upper =np.diag(np.array([np.float64(a2)]*(N-1)), k=1)
C += c2_diag_upper
a2_diag_lower = np.diag(np.array([np.float64(a2)]*(N-1)), k=-1)
C += a2_diag_lower

#a3
a3_diag_upper =np.diag(np.array([np.float64(a3)]*(N-2)), k=2)
C += a3_diag_upper
a3_diag_lower = np.diag(np.array([np.float64(a3)]*(N-2)), k=-2)
C += a3_diag_lower

# print(C)


def jacobi_method(N, A, b, max_iter=1000, min_residuum=1e-9):
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


def gauss_seidel_method_good(N, A, b, max_iter=1000, min_residuum=1e-9):
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


def gauss_seidel_method_slow(N, A, b, max_iter=1000, min_residuum=1e-9):
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


def gauss_seidel_method(N, A, b, max_iter=1000, min_residuum=1e-9):
    start = time.time()
    x = np.ones(N, dtype=np.float64)

    # wyznaczanie U i L+D
    U = np.triu(A, k=1)
    L_D = np.tril(A)  # L + D (trójkąt dolny z diagonalą)

    residuum = A @ x - b # liczenie residuum
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


def lu_solver_illegal(N, A, b):
    start = time.time()
    P, L, U = scipy.linalg.lu(A)  # Faktoryzacja LU

    # Rozwiązanie układu równań przy użyciu podstawiania w przód i wstecz
    y = scipy.linalg.solve_triangular(L, P @ b, lower=True)
    x = scipy.linalg.solve_triangular(U, y, lower=False)

    r_norm = np.linalg.norm(A @ x - b) # norma residuum
    calc_time = time.time() - start

    return x, np.array(r_norm), calc_time


############## git solver ###################
def lu_solver(N, A, b):
    start = time.time()
    A = A.astype(np.float64, copy=True)  # Kopia robocza
    P = np.arange(N)  # Wektor permutacji zamiast pełnej macierzy

    # Rozkład LU z pivotingiem
    for k in range(N - 1):
        # Wybór elementu głównego
        pivot = np.argmax(np.abs(A[k:, k])) + k
        if pivot != k:
            A[[k, pivot]] = A[[pivot, k]]  # Zamiana wierszy
            P[[k, pivot]] = P[[pivot, k]]  # Śledzenie permutacji
            b[[k, pivot]] = b[[pivot, k]]  # Równoczesna zamiana w b

        # Eliminacja Gaussa
        A[(k + 1):, k] /= A[k, k]
        A[(k + 1):, (k + 1):] -= np.outer(A[(k + 1):, k], A[k, (k + 1):])

    # Rozdzielenie L i U z pojedynczej macierzy A
    L = np.tril(A, -1) + np.eye(N)
    U = np.triu(A)

    # Rozwiązanie Ly = Pb (podstawienie w przód)
    y = np.zeros(N)
    for i in range(N):
        y[i] = b[i] - L[i, :i] @ y[:i]

    # Rozwiązanie Ux = y (podstawienie wstecz)
    x = np.zeros(N)
    for i in range(N - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]

    # Obliczenie normy residuum
    residuum = A @ x - b
    r_norm = np.linalg.norm(residuum)
    calc_time = time.time() - start
    return x, r_norm, calc_time


##################################### second #################################

def lu_solver_optimized(A, b): # ten najlepszy i guess kurde
    """Ultra-fast LU solver with partial pivoting (pure NumPy)"""
    start_time = time.time()
    n = A.shape[0]
    A = A.astype(np.float64)  # Konwersja typu
    b = b.astype(np.float64)
    P = np.arange(n)  # Wektor permutacji

    # Rozkład LU z częściowym wyborem elementu głównego
    for k in range(n - 1):
        # Znajdź pivot (największy element w kolumnie)
        max_row = k + np.argmax(np.abs(A[k:, k]))

        # Zamiana wierszy jeśli potrzebne
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]  # Zamiana wierszy w A
            P[[k, max_row]] = P[[max_row, k]]  # Aktualizacja permutacji
            b[[k, max_row]] = b[[max_row, k]]  # Zamiana w wektorze b

        # Eliminacja Gaussa (zoptymalizowana wersja)
        pivot = A[k, k]
        A[k + 1:, k] /= pivot
        A[k + 1:, k + 1:] -= A[k + 1:, k:k + 1] * A[k:k + 1, k + 1:]

    # Rozwiązanie Ly = Pb (podstawienie w przód)
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        y[i] = b[i] - np.dot(A[i, :i], y[:i])

    # Rozwiązanie Ux = y (podstawienie wstecz)
    x = np.empty(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    # Obliczenie normy residuum
    residuum = np.dot(A, x) - b
    r_norm = np.sqrt(np.dot(residuum, residuum))  # Szybsze niż np.linalg.norm

    total_time = time.time() - start_time
    return x, r_norm, total_time

####################################### teraz zadziala ########
def optimized_lu_solver(A, b):
    start = time.time()
    """Optymalizowana metoda LU z częściowym wyborem elementu głównego"""
    n = A.shape[0]
    A = A.astype(np.float64)  # Tylko konwersja typu
    b = b.astype(np.float64)
    P = np.arange(n)

    # Rozkład LU
    for k in range(n - 1):
        # Wybór pivota
        pivot = k + np.argmax(np.abs(A[k:, k]))
        if pivot != k:
            A[[k, pivot]] = A[[pivot, k]]
            P[k], P[pivot] = P[pivot], P[k]
            b[k], b[pivot] = b[pivot], P[k]

        # Eliminacja Gaussa
        pivot_inv = 1.0 / A[k, k]
        A[k + 1:, k] *= pivot_inv
        for i in range(k + 1, n):
            A[i, k + 1:] -= A[i, k] * A[k, k + 1:]

    # Rozwiązanie Ly = Pb
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(A[i, :i], y[:i])

    # Rozwiązanie Ux = y
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = (y[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    # Norma residuum
    residuum = np.dot(A, x) - b
    r_norm = np.linalg.norm(residuum)

    calc_time = time.time() - start
    return x, r_norm, calc_time

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

############################ moje new zad B fix ####################################
# JACOBI
# x =  [ 0.09197091 -0.13146997  0.17920602 ...  0.08755861 -0.04942018
#  -0.05146175]
# norma =  9.561237219316192e-10
# liczba iteracji =  109
# czas wykonania =  0.15101003646850586
#
# GAUSS-SEIDEL
# x =  [ 0.09197091 -0.13146997  0.17920602 ...  0.08755861 -0.04942018
#  -0.05146175]
# norma =  9.481026464728995e-10
# liczba iteracji =  60
# czas wykonania =  0.2152857780456543
#
# LU
# x =  [ 0.09197091 -0.13146997  0.17920602 ...  0.08755861 -0.04942018
#  -0.05146175]
# norma =  2.568899581585477e-15
# czas wykonania =  0.0480039119720459

################### lu new ###########################
# LU
# x =  [ 0.09197091 -0.13146997  0.17920602 ...  0.08755861 -0.04942018
#  -0.05146175]
# norma =  2.57000366353982e-15
# czas wykonania =  2.522589921951294
################## LU kiep ########################
# LU
# x =  [ 0.09197091 -0.13146997  0.17920602 ...  0.08755861 -0.04942018
#  -0.05146175]
# norma =  2.287545829914594
# czas wykonania =  2.3730783462524414
######################## another kiep ##########################

# zad2

# x, norm, iters, calc_time = jacobi_method(N, A, b)
# print("JACOBI")
# print("x = ", x)
# print("norma = ", norm[-1])
# print("liczba iteracji = ", iters)
# print("czas wykonania = ", calc_time)
# print()
#
# x, norm, iters, calc_time = gauss_seidel_method(N, A, b)
# print("GAUSS-SEIDEL")
# print("x = ", x)
# print("norma = ", norm[-1])
# print("liczba iteracji = ", iters)
# print("czas wykonania = ", calc_time)
# print()
#
# x, norm, calc_time = lu_solver(N, A, b)
# print("LU")
# print("x = ", x)
# print("norma = ", norm)
# print("czas wykonania = ", calc_time)
# print()

###########



#zad3
x, norm, iters, calc_time = jacobi_method(N, C, b)
print("JACOBI")
print("x = ", x)
print("norma = ", norm[-1])
print("liczba iteracji = ", iters)
print("czas wykonania = ", calc_time)
print()

x, norm, iters, calc_time = gauss_seidel_method(N, C, b)
print("GAUSS-SEIDEL")
print("x = ", x)
print("norma = ", norm[-1])
print("liczba iteracji = ", iters)
print("czas wykonania = ", calc_time)
print()

x, norm, calc_time = lu_solver(N, C, b)
print("LU")
print("x = ", x)
print("norma = ", norm)
print("czas wykonania = ", calc_time)
print()

###########################################################################
##############################   zad3    ##################################
###########################################################################

# JACOBI
# x =  [1.91461060e+123 3.09700924e+123 4.55760897e+123 ... 4.55657669e+123
#  3.09630728e+123 1.91417639e+123]
# norma =  2.992595186160183e+126
# liczba iteracji =  1000
# czas wykonania =  1.4479994773864746
#
# GAUSS-SEIDEL
# x =  [3.39999228e+249 6.64339371e+249 1.16177657e+250 ... 2.97119054e+299
#  2.38972354e+299 1.78697136e+299]
# norma =  inf
# liczba iteracji =  1000
# czas wykonania =  6.794004440307617
#
# LU
# x =  [ 0.07222212 -0.32640397  0.13095184 ... -0.36473574  0.1313745
#  -0.37978704]
# norma =  17.467811774312487
# czas wykonania =  4.290575981140137
###############################################################################

###########################################################################
##############################   zad5    ##################################
###########################################################################

# trzy odpalenia, dla kazdej metody => 3 pliki

# petla: 100 300 500 800 1000 1300 1500 1800 2000 2500 3000 3500 4000
# zakladajac macierz A => powinny sie zbiegac
# czasy + rozmiar macierzy w pliku, tyle wystarczy