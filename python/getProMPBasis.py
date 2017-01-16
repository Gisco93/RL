import numpy as np
import matplotlib.pyplot as plt

def getProMPBasis(dt, nSteps, n_of_basis, bandwidth):

    time = np.arange(dt,nSteps*dt,dt)

    Phi =

    plt.figure()
    plt.plot(time, Phi.transpose())
    plt.title('Basis Functions')

    return Phi
