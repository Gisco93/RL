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

def dmpTrain (q, qd, qdd, dt, nSteps):

    params = dmpParams()
    #Set dynamic system parameters
    params.alphaz = 3 / (dt*nSteps)
    params.alpha  = 25
    params.beta	 = 6.25
    params.Ts     = (dt*nSteps)
    params.tau    = 1
    params.nBasis = 50
    params.goal   = q[-1,:]

    Phi = getDMPBasis(params, dt, nSteps)

    #shorthand f o r parameters
    tau = params.tau 
    alpha = params.alpha 
    beta = params.beta 
    goal = params.goal 

    #Compute the forcing function
    ft = np.zeros(nSteps,2)
    for z in q :
        ydd = qdd[z,:] 
        yd = qd[z,:] 
        y = q[z,:] 
        ft[z,:] = ydd/(tau^2) - alpha * ( beta * ( no.transpose(goal) - y) - ( yd / tau ))     

    #Learn the weights
    params.w = (np.transpose(Phi) * Phi )/ (np.transpose(Phi) * ft) 

    return params
