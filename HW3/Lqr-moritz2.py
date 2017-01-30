import numpy as np
import matplotlib.pyplot as plt



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
    
    states = np.zeros((2, T+1))
    states[:, 0] = startstate
    actions = np.zeros(T)
    rewards = np.zeros(T+1)
    for i in range(1, T+1):
        w_t = np.random.normal(b_t,Sig_t)
        actions[i-1] = -1.0 * np.dot(K_t,states[:,i-1]) + k_t
        rewards[i] = compute_rt(states[:,i-1],actions[i-1],H_t,i-1,T)
        states[:,i] = np.reshape(np.reshape(np.dot(A_t, states[:, i-1]), (2, 1)) + B_t * actions[i-1] + w_t, 2)

    return actions, states , rewards
def compute_rt(s_t,a_t,H_t,t,T): 
    r_t = getr_t(t)
    R_t = getR_t(t)
    diff = np.reshape(s_t,(2,1)) - r_t
    rslt = -1.0 *np.dot(np.dot(np.transpose(diff),R_t),diff)
    if (t == T):
        return rslt
    else: 
        return rslt - np.dot(np.dot(np.transpose(a_t),H_t),a_t)


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
Actions = np.zeros((20,50))
States = np.zeros((20, 51, 2))
dev = np.zeros((51, 2))
Rewards = np.zeros((20,51))
for i in range(20):
    s = np.random.normal([0,0],1)
    (a,st,r) = lqr(s)
    Actions[i] = a
    States[i] = st.transpose()
    Rewards[i] = r

mean = np.mean(States, axis=0)
for i in range(2):
    for j in range(51):
        for k in range(20):
            dev[j,i] = dev[j,i] + (States[k,j,i] - mean[j,i]) * (States[k,j,i] - mean[j,i])
        dev[j, i] = dev[j, i] / 20

the_mean = mean.transpose()
plt.plot(the_mean[0] + 2*dev[:, 0], the_mean[1] + 2*dev[:, 1], 'y')
plt.plot(the_mean[0] - 2*dev[:, 0], the_mean[1] - 2*dev[:, 1], 'y')
plt.plot(the_mean[0], the_mean[1], 'r')
plt.fill_between(the_mean[0] + 2*dev[:, 0], the_mean[1] - 2*dev[:, 1], the_mean[1] + 2*dev[:, 1], alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')
plt.fill_between(the_mean[0] - 2*dev[:, 0], the_mean[1] - 2*dev[:, 1], the_mean[1] + 2*dev[:, 1], alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')
plt.show()