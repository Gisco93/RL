from matplotlib import pyplot as plt
import numpy as np
from Pend2dBallThrowDMP import *
# matplotlib inline


numDim = 10
numSamples = 25
maxIter = 100
numTrials = 10
lambda_var = 7
i = 0
Mu_w = np.zeros(numDim)
Sigma_w = np.eye(numDim) * 1e6

env = Pend2dBallThrowDMP()

def main():
	# Do your learning
	(q, u, b, bd) = env.simulateSystem(np.array([[2], [1]]))

	# For example, let initialize the distribution...
	Mu_w = np.zeros(numDim)
	Sigma_w = np.eye(numDim) * 1e6

	# ... then draw a sample and simulate an episode
	sample = mkSample()
	reward = rewardSample(sample)
	beta = mkBeta(reward)
	weights = mkWeight(reward, beta)
	Mu_w = mkWeightedMean(sample, weights, reward)
	Sigma_w = mkWeightedCovariance(sample, weights ,Mu_w)
	
	i = 1
	convergence = False
	while(i < 100 and not convergence):
		i += 1
		if i is 100: print "convergence not reached"
		old_reward = reward

		sample = mkSample()
		reward = rewardSample(sample)
		beta = mkBeta(reward)
		weights = mkWeight(reward, beta)
		Mu_w = mkWeightedMean(sample, weights, reward)
		Sigma_w = mkWeightedCovariance(sample, weights ,Mu_w)
		
		convergence = True if meanRewardDiff(reward, old_reward) < 1e3 else False

	# Save animation
	print Mu_w
	env.animate_fig(np.random.multivariate_normal(Mu_w, Sigma_w))
	plt.show()
	# plt.savefig('EM-Ex2.pdf')


def mkBeta(reward):
	return lambda_var / (np.max(reward) + 1 - np.min(reward))

def mkWeight(reward, beta):
	weight = np.zeros(numSamples)
	for current in weight:	
		weight[current] = np.exp((reward[current] - np.max(reward)) * beta)
	return weight

def mkSample():
    sample = np.zeros(numSamples)
    for i in range(numSamples):
        sample = np.random.multivariate_normal(Mu_w, Sigma_w)
    return sample

def rewardSample(sample):
    reward = np.zeros(numSamples)
    for i in range(numSamples):
        reward[i] = env.getReward(sample)
    return reward

def meanRewardDiff(rew1, rew2):
    return np.absolute(np.mean(rew1) - np.mean(rew2));

def mkWeightedMean(sample, weight,reward):
    sumOfWeights = 0
    sumOfWeightedSamples = 0
    for i in range(sample.shape[0]):
     	sumOfWeights += weight[i]
        sumOfWeightedSamples += weight[i] * sample[i]
    return sumOfWeights / sumOfWeightedSamples

def mkWeightedCovariance(sample, weights,mean):
    sumOfWeights = 0
    sumOfWeightedSample = 0
    for i in range(sample.shape[0]):
        sumOfWeights += weights[i]
        sumOfWeightedSample += weights[i] * (sample[i] - mean) * np.transpose(sample[i] - mean)
    return sumOfWeightedSample / sumOfWeights

if __name__ == '__main__':
    main()











































