from matplotlib import pyplot as plt
import numpy as np
from Pend2dBallThrowDMP import *
%matplotlib inline

env = Pend2dBallThrowDMP()

numDim = 10
numSamples = 25
maxIter = 100
numTrials = 10

# Do your learning


# For example, let initialize the distribution...
Mu_w = np.zeros(numDim)
Sigma_w = np.eye(numDim) * 1e6

# ... then draw a sample and simulate an episode
sample = np.random.multivariate_normal(Mu_w,Sigma_w)
reward = env.getReward(sample)

# Save animation
env.animate_fig ( np.random.multivariate_normal(Mu_w,Sigma_w) )
plt.savefig('EM-Ex2.pdf')

def getSigma(self, mu, theta, w):
    pdb.set_trace()
    variancesum = 0 
    weightsum   = 0 
    for i in range(len(w)): 
        diff = theta[i] - mu 
        variancesum += np.dot(np.dot(w[i],diff), diff)
        weightsum   += w[i]
    return variancesum / weightsum 

def getW(self, R, theta, beta): 
    weights = np.ones(len(R))
    rewards = np.array([self.getReward(theta[i]) for i in range(len(theta))])
    max_r = rewards.max()
    for i in range(len(weights)): 
        x = (rewards[i] - max_r) * beta 
        weights[i] = np.exp(x)
    return weights 

