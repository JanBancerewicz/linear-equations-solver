import time
import numpy as np
import scipy.linalg


def lu_from_library(N, A, b):
    start = time.perf_counter()
    P, L, U = scipy.linalg.lu(A)  # przeprowadzenie faktoryzacji LU za pomocą wbudowanej funkcji dla porównania z własną implementacją
    # P, L, U = scipy.linalg.lu(A)  # przeprowadzenie faktoryzacji LU za pomocą wbudowanej funkcji dla porównania z własną implementacją

    # Zastosuj permutację do wektora b: b' = P @ b
    b_permuted = P @ b

    # print("permutacja macierzy P: \n", P)

    # Rozwiązanie układu równań przy użyciu podstawiania w przód i wstecz
    y = scipy.linalg.solve_triangular(L, b_permuted, lower=True)
    x = scipy.linalg.solve_triangular(U, y, lower=False)


    r_norm = np.linalg.norm(A @ x - b) # liczenie normy residuum
    end = time.perf_counter()
    calc_time = end - start

    return x, np.array(r_norm), calc_time


def lu_short(N, A, bandwidth=0):

    L = np.eye(N, dtype=np.float64)
    U = A.astype(np.float64, copy=True)

    if bandwidth > 0:
        for i in range(N - 1):
            # wypełnienie mnożników dla niezerowych elementów w paśmie
            start_row = i + 1
            end_row = min(i + 1 + bandwidth, N)
            L[start_row:end_row, i] = U[start_row:end_row, i] / U[i, i]

            # eliminacja gaussa z pominięciem zerowych elementów
            start_col = i
            end_col = min(i + 1 + 2 * bandwidth, N)
            U[start_row:end_row, start_col:end_col] -= np.outer(L[start_row:end_row, i], U[i, start_col:end_col])
    else:
        # przypadek ogólny
        for i in range(N - 1):
            start = i + 1
            L[start:, i] = U[start:, i] / U[i, i] # wypelnienie i-tej kolumny L mnożnikami obliczanymi dzieląc kolumnę U pod diagonalą przez U[i,i]
            U[start:, i:] -= L[start:, i, None] * U[i, i:] # eliminacja gaussa - odejmowanie przemnożonych wierszy, aby osiągnąć schodkową macierz U
    return L, U



def lu_method(N, A, b, banded_matrix_optimalization=False):
    start = time.time()

    bandwidth = 2 if banded_matrix_optimalization else 0

    L, U = lu_short(N,A, bandwidth)

    y = np.zeros(N, dtype=np.float64)
    x = np.zeros(N, dtype=np.float64)

    for i in range(N): # podstawienie w przód
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    for i in range(N - 1, -1, -1): # podstawienie wstecz
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    r_norm = np.linalg.norm(A @ x - b) # liczenie normy residuum
    calc_time = time.time() - start

    return x, np.array(r_norm), calc_time
