from matplotlib import pyplot as plt
import numpy as np
from Pend2dBallThrowDMP import *
#a matplotlib inline
import pdb
import scipy as sp 
import scipy.stats
numDim = 10
numSamples = 25#changed for test by temi
maxIter = 100
#numTrials = 10
lambda_var = 7

env = Pend2dBallThrowDMP()

def learn(lambda_var):
	# Do your learning
	# For example, let initialize the distribution...
	Mu_w = np.zeros(numDim)
	Sigma_w = np.eye(numDim) * 1e6
	# ... then draw a sample and simulate an episode
	sample = mkSample(Mu_w, Sigma_w)
	reward = rewardSample(sample)
	beta = mkBeta(reward)
	weights = mkWeight(reward, beta)
	Mu_w = mkWeightedMean(sample, weights, reward)
	Sigma_w = mkWeightedCovariance(sample, weights ,Mu_w)
	mu = np.copy(Mu_w)
	sig = np.copy(Sigma_w)
	i = 0
	convergence = False
        av_rewards = np.zeros(maxIter)
	while(i < maxIter and not convergence):
		if i is maxIter-1: print "convergence not reached"
		old_reward = reward
		sample = mkSample(mu, sig)
		reward = rewardSample(sample)
		beta = mkBeta(reward)
		weights = mkWeight(reward, beta)
		mu = mkWeightedMean(sample, weights, reward)
		sig = mkWeightedCovariance(sample, weights ,mu) + np.eye(numDim)
                print np.mean(reward)
                av_rewards[i]=np.mean(reward)
		convergence= True if meanRewardDiff(reward, old_reward) < 1e-3 else False
                i += 1
        return av_rewards

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
    for i in range(numSamples):
        sumOfWeights += weights[i]
        sumOfWeightedSample += weights[i] * (np.outer((sample[:,i] - mean), np.transpose(sample[:,i] - mean)))
    return sumOfWeightedSample / sumOfWeights

average_rewards7 = np.zeros((10,100))
for i in range(10): 
    average_rewards7[i,:] = learn(7)

average_rewards3 = np.zeros((10,100))
for i in range(10): 
    average_rewards3[i,:] = learn(3)

average_rewards25 = np.zeros((10,100))
for i in range(10): 
    average_rewards25[i,:] = learn(25)

def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m,se = np.mean(a), scipy.stats.sem(a)
        h = se * sp.stats.t._ppf((1+confidence)/2.,n-1)
        return m, h

a3, a7, a25 = [np.ones((10,2)) for i in range(3)]
for i in range(10): 
    a = average_rewards3[i]
    a3[i,:] = np.array(mean_confidence_interval(a[a != 0]))

for i in range(10): 
    b = average_rewards7[i]
    a7[i,:] = np.array(mean_confidence_interval(b[b != 0]))

for i in range(10): 
    c = average_rewards25[i]
    a25[i,:] = np.array(mean_confidence_interval(c[c != 0]))

a3[:,0] = -1 * a3[:,0]
a7[:,0] = -1 * a7[:,0]
a25[:,0] = -1 * a25[:,0]

sub7 = a7[:,0]-a7[:,1]
sub7 = sub7.clip(min=0)
sub3 = a3[:,0]-a3[:,1]
sub3 = sub3.clip(min=0)
sub25 = a25[:,0]-a25[:,1]
sub25 = sub25.clip(min=0)
x = np.array(range(10))
plt.plot(x,a3[:,0],'g-')
plt.fill_between(x,a3[:,0]+a3[:,1],sub3)
plt.plot(x,a7[:,0],'r-')
plt.fill_between(x,a7[:,0]+a7[:,1],sub7)
plt.plot(x,a25[:,0],'y-')
plt.fill_between(x,a25[:,0]+a25[:,1],sub25)
#plt.plot(x,a3[:,0]+a3[:,1],'g--')
#plt.plot(x,a3[:,0]-a3[:,1],'g--')
#plt.plot(x,a7[:,0]+a7[:,1],'r--')
#plt.plot(x,sub,'r--')
#plt.plot(x,a25[:,0]+a25[:,1],'y--')
#plt.plot(x,a25[:,0]-a25[:,1],'y--')
plt.yscale('log')
#names = ('lambda=3','lambda=7','lambda=25')
#plt.legend(tuple(names))
plt.show()

