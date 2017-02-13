from matplotlib import pyplot as plt
import numpy as np
from Pend2dBallThrowDMP import *
numDim = 10
numSamples = 30
maxIter = 300
numTrials = 10
lambda_var = 7
env = Pend2dBallThrowDMP()

def main():
	step = 0.125
	mu = np.ones(numDim) * 3.255
	sig = np.eye(numDim) * 1.001
	reward = rewardSample(mkSample(mu, sig))
	max_reward = reward
	while(np.amax(sig) < 1e5):
		sig += step * np.eye(numDim)
		while(np.amax(mu) < 1e6):
			mu += step * np.ones(numDim)
			old_reward = reward
			reward = rewardSample(mkSample(mu, sig))
			if np.amax(reward) > np.amax(max_reward): 
				step = step /8
				print np.amax(mu); print np.amax(sig); max_reward = reward
				print step
				print
			else: step = step * -1.1
	


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

if __name__ == '__main__':
    main()
