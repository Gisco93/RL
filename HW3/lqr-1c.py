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
        s_des = 0
        w_t = np.random.normal(b_t, Sig_t)
        K_t = getK_t(i, T)
        k_t = getk_t(i, T)
        actions[i-1] = np.dot(K_t,s_des - states[:, i-1]) + k_t
        rewards[i] = compute_rt(states[:, i-1], actions[i-1], H_t, i-1, T)
        states[:, i] = np.reshape(np.reshape(np.dot(A_t, states[:, i-1]), (2, 1)) + B_t * actions[i-1] + w_t, 2)

    return actions, states, rewards


def lqr_standart(startstate):
    # LQR parameter
    A_t = np.array([[1, 0.1], [0, 1]])
    B_t = np.array([[0], [0.1]])
    b_t = np.array([[5], [0]])
    Sig_t = 0.01
    K_t = np.array([5, 0.3])
    k_t = 0.3
    H_t = 1
    T = 50

    states = np.zeros((2, T + 1))
    states[:, 0] = startstate
    actions = np.zeros(T)
    rewards = np.zeros(T + 1)
    for i in range(1, T + 1):
        w_t = np.random.normal(b_t, Sig_t)
        actions[i - 1] = -1.0 * np.dot(K_t, states[:, i - 1]) + k_t
        rewards[i] = compute_rt(states[:, i - 1], actions[i - 1], H_t, i - 1, T)
        states[:, i] = np.reshape(np.reshape(np.dot(A_t, states[:, i - 1]), (2, 1)) + B_t * actions[i - 1] + w_t, 2)

    return actions, states, rewards


def lqr_sdes(startstate):
    # LQR parameter
    A_t = np.array([[1, 0.1], [0, 1]])
    B_t = np.array([[0], [0.1]])
    b_t = np.array([[5], [0]])
    Sig_t = 0.01
    K_t = np.array([5, 0.3])
    k_t = 0.3
    H_t = 1
    T = 50

    states = np.zeros((2, T + 1))
    states[:, 0] = startstate
    actions = np.zeros(T)
    rewards = np.zeros(T + 1)
    for i in range(1, T + 1):
        s_des = np.array([0])
        w_t = np.random.normal(b_t, Sig_t)
        actions[i - 1] = np.dot(K_t, (s_des.transpose() - states[:, i - 1]).transpose()) + k_t
        rewards[i] = compute_rt(states[:, i - 1], actions[i - 1], H_t, i - 1, T)
        states[:, i] = np.reshape(np.reshape(np.dot(A_t, states[:, i - 1]), (2, 1)) + B_t * actions[i - 1] + w_t, 2)

    return actions, states, rewards


def compute_rt(s_t,a_t,H_t,t,T): 
    r_t = getr_t(t)
    R_t = getR_t(t)
    diff = np.reshape(s_t,(2,1)) - r_t
    rslt = -1.0 *np.dot(np.dot(np.transpose(diff), R_t), diff)
    if (t == T):
        return rslt
    else: 
        return rslt - np.dot(np.dot(np.transpose(a_t),H_t),a_t)


def getR_t(t):
    if t is 14 or 40:
        return np.array([[100000, 0],[0, 0.1]])
    else:
        return np.array([[0.01, 0],[0, 0.1]])


def getr_t(t):
    if t < 15:
        return np.array([[10],[0]])
    else:
        return np.array([[20],[0]])


def getK_t(t,T):
    A_t = np.array([[1, 0.1],[0, 1]])
    B_t = np.array([[0],[0.1]])
    H_t = 1
    inv = -np.linalg.inv(H_t + np.dot(B_t.transpose(), np.dot(getV_t(t+1, T), B_t)))
    return np.dot(inv, np.dot(B_t.transpose(), np.dot(getV_t(t+1, T), A_t)))


def getk_t(t,T):
    B_t = np.array([[0],[0.1]])
    H_t = 1
    b_t = np.array([[5],[0]])
    inv = -np.linalg.inv(H_t + np.dot(B_t.transpose(), np.dot(getV_t(t+1, T), B_t)))
    return np.dot(inv, np.dot(B_t.transpose(), np.dot(getV_t(t+1, T), b_t) - getv_t(t, T)))


def getv_t(t, T):
    A_t = np.array([[1, 0.1],[0, 1]])
    B_t = np.array([[0],[0.1]])
    b_t = np.array([[5], [0]])
    H_t = 1
    if t is T:
        return np.dot(getR_t(t), getr_t(t))
    else:
        M_t = getM_t(A_t, B_t, H_t, getV_t(t+1, T))
        return np.dot(getR_t(t), getr_t(t)) + np.dot((A_t - M_t).transpose(), getv_t(t+1, T) - np.dot(getV_t(t+1, T), b_t))


def getV_tt(t, T):
    A_t = np.array([[1, 0.1],[0, 1]])
    B_t = np.array([[0],[0.1]])
    b_t = np.array([[5], [0]])
    H_t = 1
    if t is T:
        return getR_t(t)
    else:
        M_t = getM_t(A_t, B_t, H_t, getV_tt(t+1, T))
        return getR_t(t) + np.dot((A_t - M_t).transpose(), np.dot(getV_tt(t+1, T), A_t))


def getV_t(t, T):
    A_t = np.array([[1, 0.1],[0, 1]])
    B_t = np.array([[0],[0.1]])
    H_t = 1
    vsave = getR_t(50)
    for i in range(T-t, -1, -1):
        M_t = getM_t(A_t, B_t, H_t, vsave)
        vsave = getR_t(i) + np.dot((A_t - M_t).transpose(), np.dot(vsave, A_t))
    return vsave


def getM_t(A_t, B_t, H_t, V_t1):
    inv = np.linalg.inv(H_t + np.dot(B_t.transpose(), np.dot(V_t1, B_t)))
    return np.dot(B_t, np.dot(inv, np.dot(B_t.transpose(), np.dot(V_t1, A_t))))

Actions = np.zeros((20,50))
States = np.zeros((20, 51, 2))
dev = np.zeros((2, 2))
Rewards = np.zeros((20,51))
Actionssdes = np.zeros((20,50))
Statessdes = np.zeros((20, 51, 2))
devsdes = np.zeros((2, 2))
Rewardssdes = np.zeros((20,51))
Actionskt = np.zeros((20,50))
Stateskt = np.zeros((20, 51, 2))
devkt = np.zeros((2, 2))
Rewardskt = np.zeros((20,51))
for i in range(20):
    s = np.random.normal([0, 0], 1)
    (a, st, r) = lqr_standart(s)
    (asdes, stsdes, rsdes) = lqr_sdes(s)
    (akt, stkt, rkt) = lqr(s)
    Actions[i] = a
    States[i] = st.transpose()
    Rewards[i] = r
    Actionssdes[i] = asdes
    Statessdes[i] = stsdes.transpose()
    Rewardssdes[i] = rsdes
    Actionskt[i] = akt
    Stateskt[i] = stkt.transpose()
    Rewardskt[i] = rkt

mean = np.mean(States[:,:2,:], axis=0)
meansdes = np.mean(Statessdes[:,:2,:], axis=0)
meankt = np.mean(Stateskt[:,:2,:], axis=0)
for i in range(2):
    for j in range(2):
        for k in range(20):
            dev[j,i] = dev[j,i] + (States[k,j,i] - mean[j,i]) * (States[k,j,i] - mean[j,i])
            devsdes[j,i] = devsdes[j,i] + (Statessdes[k,j,i] - meansdes[j,i]) * (Statessdes[k,j,i] - meansdes[j,i])
            devkt[j,i] = devkt[j,i] + (Stateskt[k,j,i] - meankt[j,i]) * (Stateskt[k,j,i] - meankt[j,i])
        dev[j, i] = np.sqrt(dev[j, i] / 20)
        devsdes[j, i] = np.sqrt(devsdes[j, i] / 20)
        devkt[j, i] = np.sqrt(devkt[j, i] / 20)

the_mean = mean.transpose()
the_meansdes = meansdes.transpose()
the_meankt = meankt.transpose()
plt.plot(the_meankt[0], the_meankt[1] + 2*devkt[:, 1], 'r')
plt.plot(the_meankt[0], the_meankt[1] - 2*devkt[:, 1], 'r')
plt.plot(the_meankt[0], the_meankt[1], 'r', label='(1c) Optimal Control with s_des = 0')
plt.fill_between(the_meankt[0], the_meankt[1] - 2*devkt[:, 1], the_meankt[1] + 2*devkt[:, 1], alpha=0.5, edgecolor='r', facecolor='r')
plt.fill_between(the_meankt[0], the_meankt[1] - 2*devkt[:, 1], the_meankt[1] + 2*devkt[:, 1], alpha=0.5, edgecolor='r', facecolor='r')

plt.plot(the_mean[0], the_mean[1] + 2*dev[:, 1], 'b')
plt.plot(the_mean[0], the_mean[1] - 2*dev[:, 1], 'b')
plt.plot(the_mean[0], the_mean[1], 'b', label='(1a) Standart')
plt.fill_between(the_mean[0], the_mean[1] - 2*dev[:, 1], the_mean[1] + 2*dev[:, 1], alpha=0.5, edgecolor='b', facecolor='b')
plt.fill_between(the_mean[0], the_mean[1] - 2*dev[:, 1], the_mean[1] + 2*dev[:, 1], alpha=0.5, edgecolor='b', facecolor='b')

plt.plot(the_meansdes[0], the_meansdes[1] + 2*devsdes[:, 1], 'g')
plt.plot(the_meansdes[0], the_meansdes[1] - 2*devsdes[:, 1], 'g')
plt.plot(the_meansdes[0], the_meansdes[1], 'g', label='(1b) s_des=r_t')
plt.fill_between(the_meansdes[0], the_meansdes[1] - 2*devsdes[:, 1], the_meansdes[1] + 2*devsdes[:, 1], alpha=0.5, edgecolor='g', facecolor='g')
plt.fill_between(the_meansdes[0], the_meansdes[1] - 2*devsdes[:, 1], the_meansdes[1] + 2*devsdes[:, 1], alpha=0.5, edgecolor='g', facecolor='g')

plt.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)

plt.show()
Reward = np.zeros((20, 51))
Rewardsdes = np.zeros((20, 51))
Rewardkt = np.zeros((20, 51))
r_dev = np.zeros(51)
r_devsdes = np.zeros(51)
r_devkt = np.zeros(51)
Reward[:, 0] = - Rewards[:, 0]
Rewardsdes[:, 0] = - Rewardssdes[:, 0]
Rewardkt[:, 0] = - Rewardskt[:, 0]
for i in range(1, 51):
    Reward[:, i] = Reward[:, i-1] - Rewards[:, i]
    Rewardsdes[:, i] = Rewardsdes[:, i-1] - Rewardssdes[:, i]
    Rewardkt[:, i] = Rewardkt[:, i-1] - Rewardskt[:, i]

r_mean = np.mean(Reward, axis=0)
r_meansdes = np.mean(Rewardsdes, axis=0)
r_meankt = np.mean(Rewardkt, axis=0)
for j in range(51):
    for k in range(20):
        r_dev[j] = r_dev[j] + (Reward[k, j] - r_mean[j]) * (Reward[k, j] - r_mean[j])
        r_devsdes[j] = r_devsdes[j] + (Rewardsdes[k, j] - r_meansdes[j]) * (Rewardsdes[k, j] - r_meansdes[j])
        r_devkt[j] = r_devkt[j] + (Rewardkt[k, j] - r_meankt[j]) * (Rewardkt[k, j] - r_meankt[j])
    r_dev[j] = np.sqrt(r_dev[j] / 20)
    r_devsdes[j] = np.sqrt(r_devsdes[j] / 20)
    r_devkt[j] = np.sqrt(r_devkt[j] / 20)

plt.plot(r_mean, 'r', label='the_mean 1a)')
plt.fill_between(range(51), r_mean + 2 * r_dev, r_mean - 2 * r_dev, alpha=0.5, edgecolor='r', facecolor='r')

plt.plot(r_meansdes, 'b', label='the_mean for 1b) with s_des = 0')
plt.fill_between(range(51), r_meansdes + 2 * r_devsdes, r_meansdes - 2 * r_devsdes, alpha=0.5, edgecolor='b', facecolor='b')

#plt.plot(r_meankt, 'g', label='the_mean for 1c)')
#plt.fill_between(range(51), r_meankt + 2 * r_devkt, r_meankt - 2 * r_devkt, alpha=0.5, edgecolor='g', facecolor='g')
plt.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)
plt.show()
