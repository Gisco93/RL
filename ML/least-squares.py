import numpy as np 
import matplotlib.pyplot as plt

#calculate W Matrix (Parameters), LLS Solution is 
# W = ((X*X^T)^-1)*X*Y
def calcW(F,Y):
    W = np.dot(F,np.transpose(F))
    W = np.linalg.inv(W) 
    W = np.dot(W,F)
    W = np.dot(W,Y)
    return W

# linear regression equation is L(x) = W * phi(x)
def predict(W,F):
    F = np.transpose(F)
    return np.dot(F,W)

def preprocess(filename):
    data = np.loadtxt("data/"+filename)
    X = data[0:9,0:]
    Y = data[9:,0:]
    return X,Y
#calculate featue matrix representation of single data point
def calcFeat(x):
    F = np.zeros((3,3))
    q3   = x[2]
    q2_1 = x[4]
    q3_1 = x[5]
    q1_2 = x[6]
    q2_2 = x[7]
    q3_2 = x[8]
    F[0][0] = q1_2
    F[1][0] = q1_2
    F[1][1] = 2*q3_1*q2_1*q3+(q3**2)*q2_2
    F[1][2] = q3_2-q3*q2_1**2
    F[2][0] = 1
    return F
#calculate feature matrix of data points
def calcFeatMat(X):
    first = X[0:,0]
    F = calcFeat(first)
    for i in range(1,100): 
        x = X[0:,i]
        FT = calcFeat(x)
        F  = np.concatenate((F,FT),1)
    return F; 
X,Y = preprocess("spinbotdata.txt")
F = calcFeatMat(X)
Y = Y.flatten(order = 'F')
W = calcW(F,Y)
Y_pred = predict(W,F)
u1, u2, u3, u_pred, u_pred2, u_pred3 = (np.zeros((100)) for i in range(6))
for i in range (100): 
    u1[i] = Y[3*i]
    u2[i] = Y[3*i+1]
    u3[i] = Y[3*i+2]
    u_pred[i] = Y_pred[3*i]
    u_pred2[i] = Y_pred[3*i+1]
    u_pred3[i] = Y_pred[3*i+2]

#############plot##############
plt.ylabel("u1")
plt.xlabel("t")
plt.axis([0,100,-50,60])
plt.plot(np.arange(0,100,1.0), u1, 'b')
plt.plot(np.arange(0,100,1.0),u_pred,'r')
names = ('u1','prediction of u1')
plt.legend(tuple(names))
plt.show()
############u2#############
plt.ylabel("u2")
plt.xlabel("t")
plt.axis([0,100,-50,60])
plt.plot(np.arange(0,100,1.0), u2, 'b')
plt.plot(np.arange(0,100,1.0),u_pred2,'r')
names = ('u2','prediction of u2')
plt.legend(tuple(names))
plt.show()
############u3##############
plt.ylabel("u3")
plt.xlabel("t")
plt.axis([0,100,-50,60])
plt.plot(np.arange(0,100,1.0), u3, 'b')
plt.plot(np.arange(0,100,1.0),u_pred3,'r')
names = ('u3','prediction of u3')
plt.legend(tuple(names))
plt.show()
