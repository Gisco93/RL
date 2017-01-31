import numpy as np
import matplotlib.pyplot as plt
import warnings
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

#Finite Horizon
    R = np.copy(probModel)
    V = ValIter(R, 1, 15, False)
    
    showWorld(np.maximum(V, 0), 'Value Function - Finite Horizon\n 15 steps')
    if saveFigures:
        plt.savefig('value_Fin_15.pdf')

    policy = findPolicy(V)
    ax = showWorld(grid_world, 'Policy - Finite Horizon\n 15 steps')
    showPolicy(policy, ax)
    if saveFigures:
        plt.savefig('policy_Fin_15.pdf')

#Infinite Horizon
    V = ValIter(R, 0.8, 15, True)
    showWorld(np.maximum(V, 0), 'Value Function - Infinite Horizon')
    if saveFigures:
        plt.savefig('value_Inf_08.pdf')

    policy = findPolicy(V);
    ax = showWorld(grid_world, 'Policy - Infinite Horizon')
    showPolicy(policy, ax)
    if saveFigures:
        plt.savefig('policy_Inf_08.pdf')

    # Finite Horizon with Probabilistic Transition
    V = ValIter(R,1,15,False,probModel)
    showWorld(np.maximum(V, 0), 'Value Function - Finite Horizon with Probabilistic Transition 15 steps')
    if saveFigures:
        plt.savefig('value_Fin_15_prob.pdf')

    policy = findPolicy(V,probModel)
    ax = showWorld(grid_world, 'Policy - Finite Horizon with Probabilistic Transition 15 steps')
    showPolicy(policy, ax)
    if saveFigures:
        plt.savefig('policy_Fin_15_prob.pdf')
       
    # PROOF: Finite Horizon with Probabilistic Transition
    V = ValIter(R,1,250,False,probModel)
    showWorld(np.maximum(V, 0), 'Value Function - Finite Horizon with Probabilistic Transition 250 steps')
    if saveFigures:
        plt.savefig('value_Fin_15_proof.pdf')

    policy = findPolicy(V,probModel)
    ax = showWorld(grid_world, 'Policy - Finite Horizon with Probabilistic Transition 250 steps')
    showPolicy(policy, ax)
    if saveFigures:
        plt.savefig('policy_Fin_15_proof.pdf')
    # Finite Horizon with Probabilistic Transition
    V = ValIter(R,1,15,False,probModel)
    #V = V[:,:,0]
    #printplot(V)
    showWorld(np.maximum(V, 0), 'Value Function - Finite Horizon with Probabilistic Transition')
    if saveFigures:
        plt.savefig('value_Fin_15_prob.pdf')

    policy = findPolicy(V,probModel)
    ax = showWorld(grid_world, 'Policy - Finite Horizon with Probabilistic Transition')
    showPolicy(policy, ax)
    if saveFigures:
        plt.savefig('policy_Fin_15_prob.pdf')

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
			
def ValIter(R, discount, maxSteps, infHor, probModel=np.array([])):
	V = np.copy(R)
        if(np.array_equal(probModel,np.array([]))):
            if not infHor:
                for i in range(maxSteps):
                    V = np.copy(maxAction(V, R, discount, probModel))
            else:
                V_old = np.copy(V) 
                V = np.copy(maxAction(V_old, R, discount, probModel))
                while(not np.array_equal(V,V_old) ): 
                    V_old = np.copy(V)
                    V = np.copy(maxAction(V_old, R, discount, probModel))
        else:
            for i in range(maxSteps):
                V = np.copy(maxAction(V, R, discount, probModel))
	return V
				
def maxAction(V, R, discount, probModel=np.array([])):
    V_append = np.copy(R)
    if(np.array_equal(probModel, np.array([]))):
        for x in range(R.shape[0]):
            for y in range(R.shape[1]): #for each starting point
                current_max = V[x,y] #staying case
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

                V_append[x,y] = R[x,y]+ discount * current_max;
    else: 
        for x in range(R.shape[0]):
            for y in range(R.shape[1]): #for each starting point
                current_max = V[x,y] #staying case
                Act = doAction(R,x,y)	#
                for i in range(4):		# translates the booleans to numbers
                    if Act[i]:			#
                        Act[i] = 1		#
                    else:				#
                        Act[i] = 0		#
                if  probActSum(Act,0,x,y,V)>current_max: 
                    current_max=probActSum(Act,0,x,y,V)  
                if  probActSum(Act,1,x,y,V)>current_max: 
                    current_max=probActSum(Act,1,x,y,V) 
                if  probActSum(Act,2,x,y,V)>current_max: 
                    current_max=probActSum(Act,2,x,y,V) 
                if  probActSum(Act,3,x,y,V)>current_max: 
                    current_max=probActSum(Act,3,x,y,V) 

                V_append[x,y] = R[x,y]+ discount * current_max;
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
            if V[x+Act[2],y] > current_max: P[x,y] = 0;current_max = V[x+Act[2],y]#Down
            if V[x,y+Act[1]] > current_max: P[x,y] = 1;current_max = V[x,y+Act[1]]#Right
            if V[x-Act[0],y] > current_max: P[x,y] = 2;current_max = V[x-Act[0],y]#Up
            if V[x,y-Act[3]] > current_max: P[x,y] = 3;current_max = V[x,y-Act[3]]#Left
    return P

def doAction(R, x, y):#returns wether action can be performed:
	return [((x-1) >= 0), (y+1) < R.shape[1], (x+1) < R.shape[0], (y-1) >= 0] 

def probActSum(Act, action,x,y ,V):
    if action == 0 : 
        if Act[0]: 
            return (0.7 * V[x-Act[0],y] + 0.1 * Act[1] * V[x,y+Act[1]]+ 
                    0.1 * Act[3] * V[x,y-Act[3]] + 0.1 * V[x,y])
        else: 
            return 0 
    
    if action == 1 : 
        if Act[1]: 
            return (0.7 * V[x,y+Act[1]] + 0.1 * Act[0] * V[x-Act[0],y]+ 
                    0.1 * Act[2] * V[x+Act[2],y] + 0.1 * V[x,y])
        else: 
            return 0 
    
    if action == 2 : 
        if Act[2]: 
            return (0.7 * V[x+Act[2],y] + 0.1 * Act[1] * V[x,y+Act[1]]+ 
                    0.1 * Act[3] * V[x,y-Act[3]] + 0.1 * V[x,y])
        else: 
            return 0 
    
    if action == 3 : 
        if Act[3]: 
            return (0.7 * V[x,y-Act[3]] + 0.1 * Act[0] * V[x-Act[0],y]+ 
                    0.1 * Act[2] * V[x+Act[2],y] + 0.1 * V[x,y])
        else: 
            return 0 
    return 0 
def printplot(V):
	for value in V:
		print value
	print "\n"
