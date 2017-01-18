import numpy as np
import matplotlib.pyplot as plt

def getProMPBasis(dt, nSteps, n_of_basis, bandwidth):

    time = np.arange(dt, nSteps*dt, dt)
    Phi = np.zeros((len(time), n_of_basis))

    for i in range(len(time)):
        currentTime = time[i]
        phi_i = np.zeros(n_of_basis)
        centers = np.linspace(dt, nSteps * dt - dt, n_of_basis)
        for j in range(n_of_basis):
            phi_i[j] = np.exp(-((centers[j] - currentTime)*(centers[j] - currentTime))/(bandwidth*bandwidth))
        sum_phi_i = np.sum(phi_i)
        Phi[i, :] = phi_i / sum_phi_i

    plt.figure()
    plt.plot(time, Phi)
    plt.title('Basis Functions')

    return Phi.transpose()
