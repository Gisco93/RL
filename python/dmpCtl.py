# DMP-based controller.
#
# DMPPARAMS is the struct containing all the parameters of the DMP.
#
# PSI is the vector of basis functions.
#
# Q and QD are the current position and velocity, respectively.

import numpy as np

def dmpCtl (dmpParams, psi_i, q, qd):
    alpha = dmpParams.alpha
    beta = dmpParams.beta
    tau = dmpParams.tau
    goal = dmpParams.goal
    w = dmpParams.w

    KD = tau * tau * alpha * beta
    KP = alpha * tau
    qdd = KD * (goal - q) - KP * qd + (tau * tau) * np.matmul(psi_i, w)
    return qdd
