# Learns the weights for the basis functions.
#
# Q_IM, QD_IM, QDD_IM are vectors containing positions, velocities and
# accelerations of the two joints obtained from the trajectory that we want
# to imitate.
#
# DT is the time step.
#
# NSTEPS are the total. number of steps

from getDMPBasis import *
import numpy as np


class dmpParams():
    def __init__(self):
        self.alphaz = 0.0
        self.alpha = 0.0
        self.beta = 0.0
        self.Ts = 0.0
        self.tau = 0.0
        self.nBasis = 0.0
        self.goal = 0.0
        self.w = 0.0


def dmpTrain(q, qd, qdd, dt, nSteps):

    params = dmpParams()
    # Set dynamic system parameters
    params.alphaz = 3 / (dt*nSteps)
    params.alpha = 25
    params.beta = 6.25
    params.Ts = (dt*nSteps)
    params.tau = 1
    params.nBasis = 50
    params.goal = np.transpose(q[:, -1])

    Phi = getDMPBasis(params, dt, nSteps)

    # shorthand for parameters
    tau = params.tau 
    alpha = params.alpha 
    beta = params.beta 
    goal = params.goal
    # Compute the forcing function
    ft = np.zeros((nSteps, 2))
    ydd = np.zeros((2, 1))
    yd = np.zeros((2, 1))
    y = np.zeros((2, 1))
    for z in range(0, nSteps):
        ydd[0] = qdd[0][z]
        ydd[1] = qdd[1][z]
        yd[0] = qd[0][z]
        yd[1] = qd[1][z]
        y[0] = q[0][z]
        y[1] = q[1][z]
        temp = ydd/(tau*tau) - np.transpose(alpha * (beta * np.subtract(goal, np.transpose(y)))) - (np.divide(yd, tau))
        ft[z, 0] = temp[0]
        ft[z, 1] = temp[1]

    # Learn the weights
    params.w = np.matmul(np.linalg.inv(np.matmul(Phi.transpose(), Phi) + np.eye(50)), np.matmul(Phi.transpose(), ft))

    return params
