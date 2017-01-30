import numpy as np
import matplotlib.pyplot as plt
import warnings
import pdb
warnings.filterwarnings("ignore")

def gridworld():
    saveFigures = True

    data = genGridWorld()
    grid_world = data[0]
    grid_list = data[1]

    probModel = np.copy(data[0])

    ax = showWorld(grid_world, 'Environment')
    showTextState(grid_world, grid_list, ax)
    
    if saveFigures:
        plt.savefig('gridworld.pdf')

    # Finite Horizon
    R = np.copy(probModel)
    V = ValIter(R, 0.5, 15, False, probModel)
   
    
    showWorld(np.maximum(V, 0), 'Value Function - Finite Horizon\n 15 steps')
    if saveFigures:
        plt.savefig('value_Fin_test.pdf')

    policy = findPolicy(V, probModel=np.array([]))
    ax = showWorld(grid_world, 'Policy - Finite Horizon\n 15 steps')
    showPolicy(policy, ax)
    if saveFigures:
        plt.savefig('policy_Fin_test.pdf')

'''#Infinite Horizon
    V = ValIter(R, 1, 15, True, probModel)
    showWorld(np.maximum(V, 0), 'Value Function - Infinite Horizon')
    if saveFigures:
        plt.savefig('value_Inf_08.pdf')

    policy = findPolicy(...);
    ax = showWorld(grid_world, 'Policy - Infinite Horizon')
    showPolicy(policy, ax)
    if saveFigures:
        plt.savefig('policy_Inf_08.pdf')

     # Finite Horizon with Probabilistic Transition
    V = ValIter(...)
    V = V[:,:,0];
    showWorld(np.maximum(V, 0), 'Value Function - Finite Horizon with Probabilistic Transition')
    if saveFigures:
        plt.savefig('value_Fin_15_prob.pdf')

    policy = findPolicy(...)
    ax = showWorld(grid_world, 'Policy - Finite Horizon with Probabilistic Transition')
    showPolicy(policy, ax)
    if saveFigures:
        plt.savefig('policy_Fin_15_prob.pdf')'''


##
def genGridWorld():
    O = -1e5  # Dangerous places to avoid
    D = 35    # Dirt
    W = -100  # Water
    C = -3000 # Cat
    T = 1000  # Toy
    grid_list = {0:'', O:'O', D:'D', W:'W', C:'C', T:'T'}
    grid_world = np.array([[0, O, O, 0, 0, O, O, 0, 0, 0],
        [0, 0, 0, 0, D, O, 0, 0, D, 0],
        [0, D, 0, 0, 0, O, 0, 0, O, 0],
        [O, O, O, O, 0, O, 0, O, O, O],
        [D, 0, 0, D, 0, O, T, D, 0, 0],
        [0, O, D, D, 0, O, W, 0, 0, 0],
        [W, O, 0, O, 0, O, D, O, O, 0],
        [W, 0, 0, O, D, 0, 0, O, D, 0],
        [0, 0, 0, D, C, O, 0, 0, D, 0]])
    return grid_world, grid_list


##
def showWorld(grid_world, tlt):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title(tlt)
    ax.set_xticks(np.arange(0.5,10.5,1))
    ax.set_yticks(np.arange(0.5,9.5,1))
    ax.grid(color='b', linestyle='-', linewidth=1)
    ax.imshow(grid_world, interpolation='nearest', cmap='copper')
    return ax


##
def showTextState(grid_world, grid_list, ax):
    for x in xrange(grid_world.shape[0]):
        for y in xrange(grid_world.shape[1]):
            if grid_world[x,y] >= -3000:
                ax.annotate(grid_list.get(grid_world[x,y]), xy=(y,x), horizontalalignment='center')


##
def showPolicy(policy, ax):
    for x in xrange(policy.shape[0]):
        for y in xrange(policy.shape[1]):
            if policy[x,y] == 0:
                ax.annotate('$\downarrow$', xy=(y,x), horizontalalignment='center')
            elif policy[x,y] == 1:
                ax.annotate(r'$\rightarrow$', xy=(y,x), horizontalalignment='center')
            elif policy[x,y] == 2:
                ax.annotate('$\uparrow$', xy=(y,x), horizontalalignment='center')
            elif policy[x,y] == 3:
                ax.annotate('$\leftarrow$', xy=(y,x), horizontalalignment='center')
            elif policy[x,y] == 4:
                ax.annotate('$\perp$', xy=(y,x), horizontalalignment='center')


##
def ValIter(R, discount, maxSteps, infHor, probModel=np.array([])):
	if maxSteps == 0:
		V=np.copy(R)
		return V
	else:
		V = np.copy(maxAction(ValIter(R, discount, maxSteps-1, infHor, probModel), R, discount, probModel))
	return V
				

##
def maxAction(V, R, discount, probModel=np.array([])):
    V_append = np.copy(R)
    for x in range(R.shape[0]):
        for y in range(R.shape[1]): #for each starting point
				current_max = R[x,y] #staying case
				Act = doAction(R,x,y)	#
				for i in range(4):		# translates the booleans to numbers
					if Act[i]:			#
						Act[i] = 1		#
					else:				#
						Act[i] = 0		#
				if V[x-Act[0],y] > current_max: current_max = V[x-Act[0],y];
				if V[x,y+Act[1]] > current_max: current_max = V[x,y+Act[1]];
				if V[x+Act[2],y] > current_max: current_max = V[x+Act[2],y];
				if V[x,y-Act[3]] > current_max: current_max = V[x,y-Act[3]];				
				V_append[x,y] = R[x,y] + discount * current_max;
    return V_append
					

def findPolicy(V, probModel=np.array([])):
    P = np.copy(V)
    for x in range(P.shape[0]):
        for y in range(P.shape[1]): #for each starting point
            P[x,y] = 4
            current_max = V[x,y]    #stay case
            Act = doAction(P,x,y)	#
            for i in range(4):		# translates the booleans to numbers
                if Act[i]:			#
                    Act[i] = 1		#
                else:				#
                    Act[i] = 0		#
            if V[x+Act[2],y] > current_max: P[x,y] = 0;current_max = V[x+Act[2],y]
            if V[x,y+Act[1]] > current_max: P[x,y] = 1;current_max = V[x,y+Act[1]]
            if V[x-Act[0],y] > current_max: P[x,y] = 2;current_max = V[x-Act[0],y]
            if V[x,y-Act[3]] > current_max: P[x,y] = 3;current_max = V[x,y-Act[3]]
    return P

def doAction(R, x, y):#returns wether action can be performed
	return [((x-1) >= 0), (y+1) < R.shape[1], (x+1) < R.shape[0], (y-1) >= 0] 
	
def printplot(V):
	for value in V:
		print value
	print "\n"





















