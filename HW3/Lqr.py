import numpy as np
import matplotlib.pyplot as plt
import scipy as sp 
import scipy.stats

def  lqr(mode,startstate,A_t,B_t,b_t,H_t,K_t,k_t,Sig_t,T):
    states = np.zeros((2,T+1))
    states[:,0] = np.reshape(startstate,2)
    actions = np.zeros(T)
    rewards = np.zeros(T+1)
    for i in range(1,T+1): 
        w_t = np.random.normal(b_t,Sig_t)
        actions[i-1] = geta_t(mode,i,states,K_t,k_t)
        rewards[i-1] = compute_rt(states[:,i-1],actions[i-1],H_t,i-1,T)
        states[:,i] = np.reshape(np.reshape(np.dot(A_t,states[:,i-1]),(2,1)) + B_t * actions[i-1] + w_t, 2)
    
    rewards[T] = compute_rt(states[:,T],actions[T-1],H_t,T,T)#action doesn't matter here
    return actions, states, rewards
def compute_rt(s_t,a_t,H_t,t,T):
    s_t = np.reshape(s_t,(2,1))
    r_t = getr_t(t)
    R_t = getR_t(t)
    diff = s_t - r_t
    rslt = -1.0 * np.dot(np.dot(np.transpose(diff),R_t),diff)    
    if (t == T): 
        return rslt
    else: 
        return rslt - np.dot(np.dot(np.transpose(a_t),H_t),a_t)
def geta_t(mode, t,states, K_t,k_t): 
    s_pt = states[:,t-1]
    if mode == 1:
        s_pt = getr_t(t-1) - s_pt
    a_t = -1.0 * np.dot(K_t,s_pt) + k_t#t or t-1
    return a_t

def getR_t(t):
    if t is 14 or 40 :
       return  np.array([[100000, 0],[0, 0.1]])
    else :
        return np.array([[0.01, 0],[0, 0.1]])

def getr_t(t):
    if t < 15 :
       return  np.array([[10],[0]])
    else :
        return np.array([[20],[0]])

def mean_confidence_interval(data, confidence=0.95): 
    a = 1.0 * np.array(data) 
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h 

def calcLQR(mode,startstate,A_t,B_t,b_t,H_t,K_t,k_t,Sig_t,T):

    states = np.zeros((40,T+1))
    actions = np.zeros((20,T))
    rewards = np.zeros(20)
    #do 20 experiments
    for i in range(20): 
        s = np.random.normal([[0],[0]],1)
        (a,s,r) = lqr(0,s,A_t,B_t,b_t,H_t,K_t,k_t,Sig_t,T)# calculate all actions, states and rewards perceding a given start state
        rewards[i] = r.sum()# sum up the rewards
        actions[i,:] = a # store action vector
        states[2*i:2*i+2,:]  = s # store every state of the 51 timestep, the state matrix 2x51, in global state matrix

    #mean and std of rewards
    r_mean = rewards.mean()
    r_std = rewards.std()
    rvalues = (r_mean,r_std)
    (S_x, S_y) = [states[i:40:2] for i in range(2)]#20x51 matrices, storing the state vector elements separate

    #confidence matrices for the States and actions
    (conf_x, conf_y, conf_a) = [np.zeros((51,2)) for i in range(3)]
    for i in range(51) : 
        ((m_x,h_x),(m_y,h_y))  = [mean_confidence_interval(x) for x in [S_x[:,i],S_y[:,i]]]#calculate mean and stds with 95% confidence
    #there are only 50 actions
        if i is not 50:
            (m_a,h_a) = mean_confidence_interval(actions[:,i])
        conf_x[i][0] = m_x 
        conf_x[i,1] = h_x
        conf_y[i,0] = m_y
        conf_y[i,1] = h_y
        conf_a[i,0] = m_a
        conf_a[i,1] = h_a

    #plot
    x = conf_x[:,0]
    y = conf_y[:,0]
    x_std = conf_x[:,1]
    y_std = conf_y[:,1]
    plt.plot(conf_x[:,0],conf_y[:,0],'k-')
    #plt.fill_between(y,x+x_std,x-x_std)
    plt.fill_between(x,y+y_std,y-y_std)
    plt.show()
    return rvalues, conf_x, conf_y, conf_a

#LQR parameter
A_t = np.array([[1, 0.1],[0, 1]])
B_t = np.array([[0],[0.1]])
b_t = np.array([[5],[0]])
Sig_t = 0.01
K_t = np.array([5, 0.3])
k_t  =0.3
H_t = 1    
T = 50
startstate = np.random.normal([[0],[0]],1)
