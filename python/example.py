import time as t
from dmpComparison import *
from proMP import *
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#plt.ion()

#Reproduction the desired trajectory with a DMP and save a plot
#dmpComparison([], [], 'dmp')
#Reproduce the trajectory and condition on the goal position
#dmpComparison([[0,0.2],[0.8,0.5]], [], 'goalCond')

#Reproduce the trajectory and condition on the time
#dmpComparison([], [0.5,1.5], 'timeCond')

#Learn a ProMP with 10 radial basis functions
proMP(30)
#plt.figure()
plt.show()
t.sleep(40)
