import numpy as np
import matplotlib.pyplot as plt


def plot_norms_iterative(norm_j, norm_gs, label="", show=False, save=True):
    iters_j = len(norm_j)
    iters_gs = len(norm_gs)

    plt.figure(figsize=(12, 6))

    plt.plot(range(1, iters_j + 1), norm_j, label='Jacobi', color='green')
    plt.plot(range(1, iters_gs + 1), norm_gs, label='Gauss-Seidel', color='blue')

    plt.axhline(y=1e-9, color='red', linestyle='--', linewidth=1.5, label=r'Limit: $\epsilon = 10^{-9}$')

    plt.yscale('log')  # Skala logarytmiczna na osi Y
    plt.xlabel('Iteracje')
    plt.ylabel('Norma (skala logarytmiczna)')
    plt.title('Porówanie zmian norm residuum dla metod iteracyjnych dla macierzy '+label)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    if save:
        plt.savefig('data/diagrams/norms_iterative_'+label+'.png', dpi=300)

    if show:
        plt.show()


def plot_full_simulation(tag ="linear", path="data/full_simulation.csv", show=False, save=True):
    # Wczytanie danych z pliku CSV
    data = np.genfromtxt(path, delimiter=';', skip_header=1)

    # Kolumny: size; jacobi; gauss-seidel; optimized_lu; full_lu; sp_linalg_lu; np_linalg_solve
    N = data[:, 0]
    time_jacobi = data[:, 1]
    time_gs = data[:, 2]
    time_lu_opt = data[:, 3]
    time_lu_full = data[:, 4]
    time_lu_lib = data[:, 5]
    time_solve = data[:, 6]

    plt.figure(figsize=(12, 6))

    plt.plot(N, time_jacobi, label='Jacobi', marker='o', color='green')
    plt.plot(N, time_gs, label='Gauss-Seidel', marker='s', color='blue')
    plt.plot(N, time_lu_opt, label='LU (opt)', marker='*', color='orange')
    plt.plot(N, time_lu_full, label='LU (full)', marker='p', color='red')
    plt.plot(N, time_lu_lib, label='sp.linalg.lu', marker='D', color='magenta')
    plt.plot(N, time_solve, label='np.linalg.solve', marker='X', color='purple')

    if tag == "log": # jeśli log to skala logarytmiczna
        plt.yscale('log')
    elif tag == "loglog": # jeśli loglog to obie skale logarytmiczne
        plt.xscale('log')
        plt.yscale('log')
    plt.xlabel('Rozmiar macierzy (N)', fontsize=12)
    plt.ylabel('Czas obliczeń (s)', fontsize=12)
    plt.title('Czas działania poszczególnych metod w zależności od rozmiaru macierzy', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if save:
        plt.savefig('data/diagrams/full_simulation_'+tag+'.png', dpi=300)

    if show:
        plt.show()
