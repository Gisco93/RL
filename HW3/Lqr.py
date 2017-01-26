import numpy as np
import matplotlib.pyplot as plt
def  lqr():
    A = np.array([[1, 0.1][0, 1]])
    B = np.array([[0][0.1]])
    b = np.array([[5][0]])
    Sig_t = 0.01
    K_t = np.array([5, 0.3])
    k_t  =0.3
    H_t = 1


def getR_t(t):
    if t is 14 or 40 :
       return  np.array([[100000, 0][0, 0.1]])
    else :
        return np.array([[0.01, 0][0, 0.1]])

def getr_t(t):
    if t < 15 :
       return  np.array([[10][0]])
    else :
        return np.array([[20][0]])