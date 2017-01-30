import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def gridworld():
    saveFigures = True

    data = genGridWorld()
    grid_world = data[0]
    grid_list = data[1]

    probModel = [data[0]]

    ax = showWorld(grid_world, 'Environment')
    showTextState(grid_world, grid_list, ax)
    #if saveFigures:
        #savefig('gridworld.pdf')

    # Finite Horizon
    V = ValIter(probModel, 1, 15, 15, probModel)
    V = V[:,:,0];
    showWorld(np.maximum(V, 0), 'Value Function - Finite Horizon')
    if saveFigures:
        savefig('value_Fin_15.pdf')

''' policy = findPolicy(...)
    ax = showWorld(grid_world, 'Policy - Finite Horizon')
    showPolicy(policy, ax)
    if saveFigures:
        savefig('policy_Fin_15.pdf')

# Infinite Horizon
    V = ValIter(...)
    showWorld(np.maximum(V, 0), 'Value Function - Infinite Horizon')
    if saveFigures:
        savefig('value_Inf_08.pdf')

    policy = findPolicy(...);
    ax = showWorld(grid_world, 'Policy - Infinite Horizon')
    showPolicy(policy, ax)
    if saveFigures:
        savefig('policy_Inf_08.pdf')

    # Finite Horizon with Probabilistic Transition
    V = ValIter(...)
    V = V[:,:,0];
    showWorld(np.maximum(V, 0), 'Value Function - Finite Horizon with Probabilistic Transition')
    if saveFigures:
        savefig('value_Fin_15_prob.pdf')

    policy = findPolicy(...)
    ax = showWorld(grid_world, 'Policy - Finite Horizon with Probabilistic Transition')
    showPolicy(policy, ax)
    if saveFigures:
        savefig('policy_Fin_15_prob.pdf')'''


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
		for x in range(probModel.shape[0]):
			for y in range(probModel.shape[1]): #for each starting point
				V[x,y] = R[x,y]
		return V
	else:
		V = maxAction(ValIter(R, discount, maxSteps-1, infHor, probModel), R, discount, probModel=np.array([]))
	return V #V[:,:,0] is the ideal path
				

##
def maxAction(V, R, discount, probModel=np.array([])):
	for x in range(probModel.shape[0]):
			for y in range(probModel.shape[1]): #for each starting point
				current_max = R[x,y] #staying case
				Act = doAction(R,x,y)	#
				for i in range(4):		# translates the booleans to numbers
					if Act[i]:			#
						Act[i] = 1		#
					else:				#
						Act[i] = 0		#
				if R[x-Act[0],y,0] > current_max: current_max = R[x-Act[0],y,0]	#Down
				if R[x,y+Act[1],0] > current_max: current_max = R[x,y+Act[1],0]	#Right
				if R[x+Act[2],y,0] > current_max: current_max = R[x+Act[2],y,0]	#Up
				if R[x,y-Act[3],0] > current_max: current_max = R[x,y-Act[3],0]	#Left
				V_append[x,y] = V[x,y,0] + current_max  
	return np.c_[V_append, V]	
					
##
#def findPolicy(V, probModel=np.array([])):

def doAction(R, x, y):
	return [((x-1) > 0), (y+1) < R.shape[1], (x+1) < R.shape[0], (y-1) > 0] #returns if action can be performed
	
	





















