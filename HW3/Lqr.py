import numpy as np
import matplotlib.pyplot as plta

def  lqr(startstate):
    #LQR parameter
    A_t = np.array([[1, 0.1],[0, 1]])
    B_t = np.array([[0],[0.1]])
    b_t = np.array([[5],[0]])
    Sig_t = 0.01
    K_t = np.array([5, 0.3])
    k_t  =0.3
    H_t = 1
    T = 50
    
    states = np.zeros((2,T))
    states[:,0] = startstate
    actions = np.zeros(T)
    for i in range(1,T): 
        w_t = np.random.normal(b_t,Sig_t)
        actions[i-1] = -1.0 * np.dot(K_t,states[:,i-1]) + k_t
        states[:,i] = np.dot(A_t,states[:,i-1]) + B_t * actions[i-1] + w_t

    return actions, states
def compute_rt(s_t,a_t,H_t,t,T): 
    r_t = getr_t(t)
    R_t = getR_t(t)
    diff = s_t - r_t
    rslt = -1.0 *np.dot(np.dot(np.transpose(diff),R_t),diff)    
    if (t == T): 
        return rslt
    else: 
        return rslt - np.dot(np.dot(np.transpose(a_t),H_t),a_t)


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
