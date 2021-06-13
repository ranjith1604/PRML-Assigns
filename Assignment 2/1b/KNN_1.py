import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from statistics import mode
from sklearn.metrics import accuracy_score



#Taking input from csv file and taking x and y out
data=pd.read_csv("19/train.csv",header=None)

data=data.to_numpy()


X_train=data[:,0:2]
Y_train=data[:,2]


data=pd.read_csv("19/dev.csv",header=None)
data=data.to_numpy()
#print(data)


X_valid=data[0:int(data.shape[0]/2),0:2]
Y_valid=data[0:int(data.shape[0]/2),2]

X_test=data[int(data.shape[0]/2):,0:2]
Y_test=data[int(data.shape[0]/2):,2]

#print(Y_test.shape)


K=[1,7,15]

#KNN classifier
for k in K:
	
	predicted=[]
	for p in range(len(Y_valid)):
		test=X_valid[p]

		le=np.sum((test-X_train)**2,axis=1)
		distances=list()

		for i in range(len(Y_train)):
			distances.append((Y_train[i],le[i]))

		distances.sort(key=lambda tup: tup[1])
		neighbors = list()
		for i in range(k):
			neighbors.append(distances[i][0])
		predicted.append(mode(neighbors))
	
	print("accuracy for k="+str(k)+" on validation set is "+str(accuracy_score(Y_valid,predicted)*100))

for k in K:
	
	predicted=[]
	for p in range(len(Y_train)):
		test=X_train[p]

		le=np.sum((test-X_train)**2,axis=1)
		distances=list()

		for i in range(len(Y_train)):
			distances.append((Y_train[i],le[i]))

		distances.sort(key=lambda tup: tup[1])
		neighbors = list()
		for i in range(k):
			neighbors.append(distances[i][0])
		predicted.append(mode(neighbors))
	
	print("accuracy for k="+str(k)+" on training set is "+str(accuracy_score(Y_train,predicted)*100))
