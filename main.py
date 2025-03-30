import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp

N= 1299
a1=5
a2=-1
a3=-1
c=9 # przedostatnia cyfra
d=9 # ostatnia cyfra
e=0 # czwarta cyfra
f=8 # trzecia cyfra


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
main_diag = np.diag(np.array([a1]*N), k=0)
A = main_diag

#a2
a2_diag_upper =np.diag(np.array([a2]*(N-1)), k=1)
A += a2_diag_upper
a2_diag_lower = np.diag(np.array([a2]*(N-1)), k=-1)
A += a2_diag_lower

#a3
a3_diag_upper =np.diag(np.array([a3]*(N-2)), k=2)
A += a3_diag_upper
a3_diag_lower = np.diag(np.array([a3]*(N-2)), k=-2)
A += a3_diag_lower

# print(A)

# tworzenie wektora b
b = np.array( [np.sin(n*(f+1)) for n in range(1, N+1)] )

# print(b)



