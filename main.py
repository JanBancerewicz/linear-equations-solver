import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg
import time


from jacobi import jacobi_method
from gauss_seidel import gauss_seidel_method
from lu import lu_method, lu_from_library

from helpers import generate_matrix, print_things, solve_direct

from parameters import a1, a2, a3, f, N


np.set_printoptions(threshold=20) # ustawienie limitu na 20 elementów podczas wyświetlania tablicy


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



#zad3 ######################################################## declarations
# x, norm, iters, calc_time = jacobi_method(N, C, b)
# print("JACOBI")
# print("x = ", x)
# print("norma = ", norm[-1])
# print("liczba iteracji = ", iters)
# print("czas wykonania = ", calc_time)
# print()
#
# x, norm, iters, calc_time = gauss_seidel_method(N, C, b)
# print("GAUSS-SEIDEL")
# print("x = ", x)
# print("norma = ", norm[-1])
# print("liczba iteracji = ", iters)
# print("czas wykonania = ", calc_time)
# print()
#
# x, norm, calc_time = lu_solver(N, C, b)
# print("LU")
# print("x = ", x)
# print("norma = ", norm)
# print("czas wykonania = ", calc_time)
# print()

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
########################################################################### #

# trzy odpalenia, dla kazdej metody => 3 pliki

# petla: 100 300 500 800 1000 1300 1500 1800 2000 2500 3000 3500 4000
# zakladajac macierz A => powinny sie zbiegac
# czasy + rozmiar macierzy w pliku, tyle wystarczy


############################################ TESTUJE TU #######################

def simulate(N, method=0, matrix="A", write=False, nametag=""):
    if matrix == "C": # zasymuluj macierz C
        a1 = 3
    else:
        a1 = 5


    Asim = generate_matrix(N, new_a1=a1) # generowanie macierzy A
    bsim = np.array( [np.sin(n*(np.float64(f+1))) for n in range(1, N+1)] ) # tworzenie wektora b

    plotresults = [] # chcemy plotowac:tablica norm, czas liczba iteracji
    if method == 1:
        x, norm, iters, calc_time = jacobi_method(N, Asim, bsim)
        if write:
            print_things(x, norm, calc_time, N, iters, strr="jacobi", matrix=matrix)
            f1 = open("data/iter/jacobi_norm_"+ matrix+".txt", "w")
            for item in norm:
                f1.write("%s\n" % item)
            f1.close()

            f2 = open("data/iter/jacobi_solve_"+matrix+".txt", "w")
            f2.write(f"Jacobi method for N: {N} and matrix {matrix};\t")
            f2.write("iters: " + str(iters) + ";\t"+"time: " + str(round(calc_time, 5)) +"\n")
            for item in x:
                f2.write("%s\n" % item)
            f2.close()
        plotresults.append([norm, calc_time, iters])

    if method == 2:
        x, norm, iters, calc_time = gauss_seidel_method(N, Asim, bsim)
        if write:
            print_things(x, norm, calc_time, N, iters, strr="gauss-seidel", matrix=matrix)
            f1 = open("data/iter/gauss_seidel_norm_"+ matrix+".txt", "w")
            for item in norm:
                f1.write("%s\n" % item)
            f1.close()

            f2 = open("data/iter/gauss_seidel_solve_"+ matrix+".txt", "w")
            f2.write(f"Gauss_seidel method for N: {N} and matrix {matrix};\t")
            f2.write("iters: " + str(iters) + ";\t" + "time: " + str(round(calc_time, 5)) + "\n")
            for item in x:
                f2.write("%s\n" % item)
            f2.close()
        plotresults.append([norm, calc_time, iters])


    if method == 3:
        if nametag == "_library":
            x, norm, calc_time = lu_from_library(N, Asim, bsim) # test wbudowanego lu dla porównania
            if write:
                print_things(x, norm, calc_time, N, 0, strr="funkcja scipy.linalg.lu()", matrix=matrix)

                f1 = open("data/direct/lu_solve" + nametag + "_" + matrix + ".txt", "w")
                f1.write(f"[Reference only] Built-in scipy.linalg.lu() method for N: {N} and matrix {matrix};\t")
                f1.write("norm: " + str(norm) + ";\t" + "time: " + str(round(calc_time, 5)) + "\n")
                for item in x:
                    f1.write("%s\n" % item)
                f1.close()
            plotresults.append([norm, calc_time, 0])

        elif nametag == "_optimized":
            x, norm, calc_time = lu_method(N, Asim, bsim, banded_matrix_optimalization=True)
            if write:
                print_things(x, norm, calc_time, N, 0, strr="zoptymalizowany lu", matrix=matrix)

                f2 = open("data/direct/lu_solve" + nametag + "_" + matrix + ".txt", "w")
                f2.write(f"Optimized LU method for N: {N} and matrix {matrix};\t")
                f2.write("norm: " + str(norm) + ";\t" + "time: " + str(round(calc_time, 5)) + "\n")
                for item in x:
                    f2.write("%s\n" % item)
                f2.close()
            plotresults.append([norm, calc_time, 0])

        else: #full
            x, norm, calc_time = lu_method(N, Asim, bsim)
            if write:
                print_things(x, norm, calc_time, N, 0, strr="pelny lu", matrix=matrix)

                f3 = open("data/direct/lu_solve" + nametag + "_" + matrix + ".txt", "w")
                f3.write(f"Full LU method for N: {N} and matrix {matrix};\t")
                f3.write("norm: " + str(norm) + ";\t" + "time: " + str(round(calc_time, 5)) + "\n")
                for item in x:
                    f3.write("%s\n" % item)
                f3.close()
            plotresults.append([norm, calc_time, 0])

    return plotresults # norma, czas, liczba iteracji


def generate_simulation():

    steps = [100, 300, 500, 800, 1000, 1300, 1500, 1800, 2000, 2500, 3000, 3500, 4000]
    print("jacobi")
    for size in steps:
        values = simulate(size, 1)
        norm, calc_time, iters = values[0]
        print(norm[-1], calc_time, iters, size)
    print("gauss-seidel")
    for size in steps:
        values = simulate(size, 2)
        norm, calc_time, iters = values[0]
        print(norm[-1], calc_time, iters, size)
    print("lu")
    for size in steps:
        values = simulate(size, 3, nametag="_optimized")
        norm, calc_time, iters = values[0]
        print(norm, calc_time, iters, size)
    return None




# print("\n######################### ZAD A ###########################\n")
# print(generate_matrix(N)) #defaultowo "A"

# b = np.array( [np.sin(n*(np.float64(f+1))) for n in range(1, N+1)] )
# print(f"b = {b}")


print("\n######################## ZAD B ###########################\n")
# simulate(1299, method=1,matrix="A", write=True)
# simulate(1299, method=2,matrix="A", write=True)
#
# simulate(1299, method=3,matrix="A", write=True, nametag="_optimized")
# simulate(1299, method=3,matrix="A", write=True, nametag="_library")
# simulate(1299, method=3,matrix="A", write=True, nametag="_full")
#
#
# solve_direct(1299, matrix="A", write=True)


print("\n###################### ZAD C i D #########################\n")
# simulate(1299, method=1,matrix="C", write=True)
# simulate(1299, method=2,matrix="C", write=True)
#
# simulate(1299, method=3,matrix="C", write=True, nametag="_optimized")
# simulate(1299, method=3,matrix="C", write=True, nametag="_library")
# simulate(1299, method=3,matrix="C", write=True, nametag="_full")
#
#
# solve_direct(1299, matrix="C", write=True)

print("\n######################### ZAD E ###########################\n")

generate_simulation()



