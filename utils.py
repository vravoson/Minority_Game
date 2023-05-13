import random as rd
import numpy as np
from numba import njit

def bin2dec(num):
        return int(''.join(str(x) for x in num),2)

@njit
def strategy(M):
    return np.random.binomial(1, 0.5, 2**M)
