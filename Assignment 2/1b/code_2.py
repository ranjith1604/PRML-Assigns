import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from statistics import mode
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

	
#Taking input from csv file and taking x and y out
data=pd.read_csv("19/train.csv",header=None)

data=data.to_numpy()


X_train=data[:,0:2]
Y_train=data[:,2]


data=pd.read_csv("19/dev.csv",header=None)
data=data.to_numpy()

X_valid=data[0:int(data.shape[0]/2),0:2]
Y_valid=data[0:int(data.shape[0]/2),2]

X_test=data[int(data.shape[0]/2):,0:2]
Y_test=data[int(data.shape[0]/2):,2]


#print(type(Y_train))
(unique, counts) = np.unique(Y_train, return_counts=True)
#print(unique.shape[0])
means=np.zeros((unique.shape[0],X_train.shape[1]))
#print(means)

for i in range(X_train.shape[0]):
	means[np.where(unique==Y_train[i]),:]+=X_train[i]
#print(means)
means=means/(np.tile(counts,(means.shape[1],1)).T)
#print(means)
sum=0
#print(X_train[0]-means[np.where(unique==Y_train[0]),:])
for i in range(X_train.shape[0]):
	le=(X_train[i]-means[np.where(unique==Y_train[i]),:])
	sum=sum+np.dot(le[0],le[0].T)

#print(sum)
sigma2=sum[0][0]/(X_train.shape[0]*X_train.shape[1])
predicted=[]
for i in range(len(Y_train)):
	predicted.append(predict(X_train[i],means,sigma2,counts))

print("accuracy on training set with covariance matrix as sigma2 is "+str(accuracy_score(Y_train,predicted)*100))

predicted.clear()

for i in range(len(Y_valid)):
	predicted.append(predict(X_valid[i],means,sigma2,counts))
	
print("accuracy on validation set with covariance matrix as sigma2 is "+str(accuracy_score(Y_valid,predicted)*100))

cov=np.zeros((X_train.shape[1],X_train.shape[1]))
#print(cov)
for i in range(X_train.shape[0]):
	tp=(X_train[i]-means[np.where(unique==Y_train[i]),:])
	le=np.dot(tp[0].T,tp[0])
	cov=cov+le
#print(X_train[0]-means[np.where(unique==Y_train[0]),:])
#tp=(X_train[0]-means[np.where(unique==Y_train[0]),:])
#print(cov)
cov=cov/(X_train.shape[0])

#print((X_train-means).reshape(len(X_train[0]),1))
#print(predict_cov(X_train[5],means,cov,counts))
predicted.clear()

for i in range(len(Y_train)):
	predicted.append(predict_cov(X_train[i],means,cov,counts))

print("accuracy on training set with covariance matrix as cov is "+str(accuracy_score(Y_train,predicted)*100))

predicted.clear()

for i in range(len(Y_valid)):
	predicted.append(predict_cov(X_valid[i],means,cov,counts))
	
print("accuracy on validation set with covariance matrix as cov is "+str(accuracy_score(Y_valid,predicted)*100))

covdif=np.zeros((len(unique),X_train.shape[1],X_train.shape[1]))
#print(covdif)
for i in range(X_train.shape[0]):
	tp=(X_train[i]-means[np.where(unique==Y_train[i]),:])
	le=np.dot(tp[0].T,tp[0])
	covdif[np.where(unique==Y_train[i]),:,:]=covdif[np.where(unique==Y_train[i]),:,:]+le


for i in range(len(unique)):
	covdif[i,:,:]=covdif[i,:,:]/counts[i]


predicted.clear()

for i in range(len(Y_train)):
	predicted.append(predict_covdif(X_train[i],means,covdif,counts))

print("accuracy on training set with covariance matrix as difcov is "+str(accuracy_score(Y_train,predicted)*100))
print("Confusion Matrix for training data is: ")
print(confusion_matrix(Y_train, predicted))

predicted.clear()

for i in range(len(Y_valid)):
	predicted.append(predict_covdif(X_valid[i],means,covdif,counts))
	
print("accuracy on validation set with covariance matrix as difcov is "+str(accuracy_score(Y_valid,predicted)*100))
print("Confusion Matrix for validation data is: ")
print(confusion_matrix(Y_valid, predicted))

