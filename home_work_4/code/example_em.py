from matplotlib import pyplot as plt
import numpy as np
from Pend2dBallThrowDMP import *

# matplotlib inline

env = Pend2dBallThrowDMP()

numDim = 10
numSamples = 25
maxIter = 100
numTrials = 10
i = 0
# Do your learning
print np.array([[2], [1]])
(q, u, b, bd) = env.simulateSystem(np.array([[2], [1]]))
print q
print u
print b
print bd
# For example, let initialize the distribution...
Mu_w = np.zeros(numDim)
Sigma_w = np.eye(numDim) * 1e6

# ... then draw a sample and simulate an episode
reward = np.zeros(numSamples)
for j in range(numSamples):
    sample = np.random.multivariate_normal(Mu_w, Sigma_w)
    reward[j] = env.getReward(sample)
    print reward
    beta = 7 / (np.max(reward) - np.min(reward))
    w_i = np.exp((reward[j] - np.max(reward)) * beta)
    print w_i
  #  while(i < 100):



# reward_new = env.getReward(np.random.multivariate_normal(Mu_w_new, Sigma_w_new))
# if np.mean(reward) - np.mean(reward_new) < 0.001:
# break
# Sigma_w = Sigma_w_new + np.eye(Sigma_w.shape)

# Save animation
env.animate_fig(np.random.multivariate_normal(Mu_w, Sigma_w))
plt.show()
# plt.savefig('EM-Ex2.pdf')
