import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from statistics import mode
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import multiprocessing
import matplotlib
import matplotlib.patches as mpatches
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import seaborn as sn

def KNN(X_train,Y_train,k,X):
    predicted=[]
    for p in range(X.shape[0]):
        test=X[p]

        le=np.sum((test-X_train)**2,axis=1)
        #distances=list()
        distances=defaultdict(list)
        for i in range(len(Y_train)):
            distances[Y_train[i]].append(le[i])

        #distances.sort(key=lambda tup: tup[0])
        neighbors = list()
        for l, v in distances.items():  #.iteritems for lower python versions
            distances[l].sort()
            neighbors.append(distances[l][k-1])
        #if p==0:
        #    print(distances)
        predicted.append(np.argmin(neighbors,axis=0))


    return predicted


def KNN_single(X_train,Y_train,k,X):
    predicted=[]
    for p in range(1):
        test=X

        le=np.sum((test-X_train)**2,axis=1)
        #distances=list()
        distances=defaultdict(list)
        for i in range(len(Y_train)):
            distances[Y_train[i]].append(le[i])

        #distances.sort(key=lambda tup: tup[0])
        neighbors = list()
        for l, v in distances.items():  #.iteritems for lower python versions
            distances[l].sort()
            neighbors.append(distances[l][k-1])
        #if p==0:
        #    print(distances)
        predicted.append(np.argmin(neighbors,axis=0))


    return predicted[0]



#Taking input from csv file and taking x and y out
data=pd.read_csv("19/train.csv",header=None)

data=data.to_numpy()


X_train=data[:,0:2]
Y_train=data[:,2]


data=pd.read_csv("19/dev.csv",header=None)
X_test=data.loc[np.r_[15:30, 45:60, 75:90],:]
X_test=X_test.to_numpy()
Y_test=X_test[:,2]
X_test=X_test[:,0:2]
#X_valid=data[0:int(data.shape[0]/2),0:2]
#Y_valid=data[0:int(data.shape[0]/2),2]

X_valid=data.loc[np.r_[0:15, 30:45, 60:75],:]
X_valid=X_valid.to_numpy()
Y_valid=X_valid[:,2]
X_valid=X_valid[:,0:2]


K=[10,20]

#KNN classifier
for k in K:
    predicted=KNN(X_train,Y_train,k,X_valid)
    print("accuracy for k="+str(k)+" on validation set is "+str(accuracy_score(Y_valid,predicted)*100))

for k in K:
    predicted=KNN(X_train,Y_train,k,X_train)
    print("accuracy for k="+str(k)+" on training set is "+str(accuracy_score(Y_train,predicted)*100))

k=10

predicted=KNN(X_train,Y_train,k,X_test)
print("accuracy for k="+str(k)+" on testing set is "+str(accuracy_score(Y_test,predicted)*100))


confuse=confusion_matrix(Y_test,predicted)
#print(confuse)
sn.heatmap(confuse/np.sum(confuse,axis=0), annot=True,
    fmt='.2%', cmap='Blues',cbar=False)
plt.xlabel('Predicted Class')
plt.ylabel("Actual Class")
plt.title('Confusion Matrix for Bayes KNN with k=10 on Testing data')
plt.savefig('Confusion_test_Bayes_KNN_4.png')

plt.show()

predicted=KNN(X_train,Y_train,k,X_train)

confuse=confusion_matrix(Y_train,predicted)
#print(confuse)
sn.heatmap(confuse/np.sum(confuse,axis=0), annot=True,
    fmt='.2%', cmap='Blues',cbar=False)
plt.xlabel('Predicted Class')
plt.ylabel("Actual Class")
plt.title('Confusion Matrix for Bayes KNN with k=10 on Training data')
plt.savefig('Confusion_train_Bayes_KNN_4.png')

plt.show()


x1=np.linspace(-4,4,num=200)
x2=np.linspace(-3,3,num=200)
xx1, xx2 = np.meshgrid(x1, x2)
r1, r2 = xx1.flatten(), xx2.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
grid = np.hstack((r1,r2))
#print(grid)
predicted.clear()

num_cores = multiprocessing.cpu_count()
#print(X_valid[0].shape[1])
predicted = Parallel(n_jobs=num_cores)(delayed(KNN_single)(X_train,Y_train,10,grid[i]) for i in range(grid.shape[0]))
#predicted = Parallel(n_jobs=num_cores)(delayed(knn)(grid[i],means,covdif,counts) for i in range(grid.shape[0]))
#predicted=KNN(X_train,Y_train,15,grid)
#print(predicted)
predicted=np.array(predicted)


predicted=predicted.reshape(xx1.shape)
fig = plt.figure(figsize=(8,8))
plt.contourf(xx1, xx2, predicted, cmap='RdBu')
colors = ['green','red','blue','purple']
plt.scatter(X_train[:,0], X_train[:,1], c=Y_train, cmap=matplotlib.colors.ListedColormap(colors))
#plt.scatter(X_train[:,0], X_train[:,1], c=Y_train, cmap='RdBu')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Decision Region plot for k=10 for each class')


plt.savefig('plot_KNN_Bayes_4.png')
plt.show()
