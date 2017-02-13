from matplotlib import pyplot as plt
import numpy as np
from Pend2dBallThrowDMP import *
# matplotlib inline


numDim = 10
numSamples = 30
maxIter = 300
numTrials = 10
lambda_var = 7

env = Pend2dBallThrowDMP()

def main():
	# Do your learning
	# For example, let initialize the distribution...
	Mu_w = np.zeros(numDim)
	Sigma_w = np.eye(numDim) * 1e5
	# ... then draw a sample and simulate an episode
	sample = mkSample(Mu_w, Sigma_w)
	reward = rewardSample(sample)
	beta = mkBeta(reward)
	weights = mkWeight(reward, beta)
	Mu_w = mkWeightedMean(sample, weights, reward)
	Sigma_w = mkWeightedCovariance(sample, weights ,Mu_w)
	mu = np.copy(Mu_w)
	sig = np.copy(Sigma_w)
	i = 1
	convergence = False
	while(i < maxIter and not convergence):
		i += 1
		if i is maxIter: print "convergence not reached"
		old_reward = reward
		sample = mkSample(mu, sig)
		reward = rewardSample(sample)
		beta = mkBeta(reward)
		weights = mkWeight(reward, beta)
		mu = mkWeightedMean(sample, weights, reward)
		sig = mkWeightedCovariance(sample, weights ,mu) + np.eye(numDim)
		convergence = True if meanRewardDiff(reward, old_reward) < 1e-3 else False
      
	# Save animation
	env.animate_fig(np.random.multivariate_normal(mu, sig))
	plt.show()
	# plt.savefig('EM-Ex2.pdf')


def mkBeta(reward):
	return lambda_var / (np.amax(reward) - np.amin(reward))

def mkWeight(reward, beta):
	weight = np.zeros(numSamples)
	for current in range(numSamples):	
		weight[current] = np.exp((reward[current] - np.max(reward)) * beta)
	return weight

def mkSample(mu, sig):
    sample = np.zeros((numDim, numSamples))
    for i in range(numSamples):
        sample[:,i] = np.random.multivariate_normal(mu, sig)
    return sample

def rewardSample(sample):
    reward = np.zeros(numSamples)
    for i in range(numSamples):
        reward[i] = env.getReward(sample[:,i])
    return reward

def meanRewardDiff(rew1, rew2):
    return np.absolute(np.mean(rew1) - np.mean(rew2));

def mkWeightedMean(sample, weight,reward):
    sumOfWeights = 0
    sumOfWeightedSamples = 0    
    for i in range(numSamples):
     	sumOfWeights += weight[i]
        sumOfWeightedSamples += weight[i] * sample[:,i]
    return sumOfWeightedSamples / sumOfWeights
    
def mkWeightedCovariance(sample, weights,mean):
    sumOfWeights = 0
    sumOfWeightedSample = 0
    for i in range(sample.shape[0]):
        sumOfWeights += weights[i]
        sumOfWeightedSample += np.dot(weights[i], np.dot((sample[:,i] - mean), np.transpose(sample[:,i] - mean)))
    return np.dot((sumOfWeightedSample / sumOfWeights), np.eye(numDim))

if __name__ == '__main__':
    main()











































