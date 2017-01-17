import numpy as np
import matplotlib.pyplot as plt
from getImitationData import *
from getProMPBasis import *

def proMP (nBasis):

    dt = 0.002
    time = np.arange(dt,3,dt)
    nSteps = len(time);
    data = getImitationData(dt, time, multiple_demos=True)
    q = data[0]
    qd = data[1]
    qdd = data[2]

    bandwidth = 0.2
    Phi = getProMPBasis(dt, nSteps, nBasis, bandwidth)
    print Phi.shape
    print q.shape
    w = np.matmul((np.matmul(Phi, Phi.transpose())), np.linalg.pinv((np.matmul(Phi, q.transpose()))).transpose())
    mean_w = np.mean(w.transpose(), axis=0)
    cov_w = np.cov(w)
    print w.shape
    print mean_w.shape
    print cov_w.shape
    plt.figure()
    plt.hold('on')
    plt.fill_between(time, np.dot(Phi.transpose(), mean_w) - 2*np.sqrt(np.diag(np.dot(Phi.transpose(), np.dot(cov_w, Phi)))), np.dot(Phi.transpose(), mean_w) + 2*np.sqrt(np.diag(np.dot(Phi.transpose(), np.dot(cov_w, Phi)))), alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')
    plt.plot(time, np.dot(Phi.transpose(), mean_w), color='#1B2ACC')
    plt.plot(time, q.transpose())
    plt.title('ProMP learned from several trajectories')

    #Conditioning
    y_d = 3
    Sig_d = 0.0002
    t_point = np.round(2300/2)
    x = np.divide(y_d - np.matmul( Phi.transpose(), w), Sig_d)
    w_new = np.exp(- x*x) / (np.sqrt(np.pi) * Sig_d)
    mean_w_new = np.mean(w_new.transpose(), axis=0)
    cov_w_new = np.cov(w_new)
    print w_new.shape
    print mean_w_new.shape
    print cov_w_new.shape

    plt.figure()
    plt.hold('on')
    plt.fill_between(time, np.dot(Phi.transpose(), mean_w) - 2*np.sqrt(np.diag(np.dot(Phi.transpose(), np.dot(cov_w, Phi)))), np.dot(Phi.transpose(), mean_w) + 2*np.sqrt(np.diag(np.dot(Phi.transpose(), np.dot(cov_w, Phi)))), alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')
    # plt.fill_between(time, np.dot(Phi.transpose(), mean_w) - 2*np.sqrt(np.diag(np.dot(Phi.transpose(), np.dot(cov_w, Phi)))), np.dot(Phi.transpose(), mean_w) + 2*np.sqrt(np.diag(np.dot(Phi.transpose(), np.dot(cov_w, Phi)))), alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')
    plt.plot(time, np.dot(Phi.transpose(), mean_w), color='#1B2ACC')
    #plt.fill_between(time, np.dot(Phi.transpose(), mean_w_new) - 2*np.sqrt(np.diag(np.dot(Phi.transpose(), np.dot(cov_w_new, Phi)))), np.dot(Phi.transpose(), mean_w_new) + 2*np.sqrt(np.diag(np.dot(Phi.transpose(), np.dot(cov_w_new, Phi)))), alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    # plt.fill_between(time, np.dot(Phi.transpose(), mean_w_new) - 2*np.sqrt(np.diag(np.dot(Phi.transpose(), np.dot(cov_w_new, Phi)))), np.dot(Phi.transpose(), mean_w_new) + 2*np.sqrt(np.diag(np.dot(Phi.transpose(), np.dot(cov_w_new, Phi)))), alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    #plt.plot(time, np.dot(Phi.transpose(), mean_w_new), color='#CC4F1B')

    #sample_traj = np.dot(Phi.transpose(),np.random.multivariate_normal(mean_w_new,cov_w_new,10).transpose())
    #plt.plot(time,sample_traj)
    plt.title('ProMP after contidioning with new sampled trajectories')
















