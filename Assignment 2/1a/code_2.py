import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from statistics import mode
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.patches as mpatches
from joblib import Parallel, delayed
import multiprocessing
from scipy.stats import multivariate_normal
import seaborn as sn

def predict(x,means,sigma2,counts):  #S
	tp=np.sum((x-means)**2,axis=1)
	tp=tp/(-2*sigma2)
	tp=tp+np.log(counts)
	return np.argmax(tp)

def predict_cov(x,means,cov,counts):
	tp=x-means
	tp=np.dot(np.dot(tp,np.linalg.inv(cov)),tp.T)
	value=[]
	for i in range(tp.shape[0]):
		value.append(tp[i][i])
	value=np.array(value)
	value=value/-2
	return np.argmax(value+np.log(counts))

def predict_covdif(x,means,covdif,counts):
	tp=x-means
	value=[]
	for i in range(tp.shape[0]):
		le=tp[i,:].reshape(len(tp[i,:]),1)
		value.append((np.dot(np.dot(le.T,np.linalg.inv(covdif[i,:,:])),le))[0][0]+np.log(np.linalg.det(covdif[i,:,:])))
	value=np.array(value)
	value=value/-2
	return np.argmax(value+np.log(counts))





#Taking input from csv file and taking x and y out
data=pd.read_csv("19/train.csv",header=None)

data=data.to_numpy()


X_train=data[:,0:2]
Y_train=data[:,2]


data=pd.read_csv("19/dev.csv",header=None)
data=data.to_numpy()

X_valid=data[0:60,0:2]
Y_valid=data[0:60,2]

X_test=data[60:120,0:2]
Y_test=data[60:120,2]


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

print("accuracy on training set with covariance matrix as same sigma2 is "+str(accuracy_score(Y_train,predicted)*100))

predicted.clear()

for i in range(len(Y_valid)):
	predicted.append(predict(X_valid[i],means,sigma2,counts))

print("accuracy on validation set with covariance matrix as same sigma2 is "+str(accuracy_score(Y_valid,predicted)*100))

cov=np.zeros((X_train.shape[1],X_train.shape[1]))
#print(cov)
for i in range(X_train.shape[0]):
	tp=(X_train[i]-means[np.where(unique==Y_train[i]),:])
	le=np.dot(tp[0].T,tp[0])
	cov=cov+le

cov=cov/(X_train.shape[0])
d= np.diag(cov)
cov=np.diag(d)

#print((X_train-means).reshape(len(X_train[0]),1))
#print(predict_cov(X_train[5],means,cov,counts))
predicted.clear()

for i in range(len(Y_train)):
	predicted.append(predict_cov(X_train[i],means,cov,counts))

print("accuracy on training set with covariance matrix as same cov is "+str(accuracy_score(Y_train,predicted)*100))

predicted.clear()

for i in range(len(Y_valid)):
	predicted.append(predict_cov(X_valid[i],means,cov,counts))

print("accuracy on validation set with covariance matrix as same cov is "+str(accuracy_score(Y_valid,predicted)*100))

covdif=np.zeros((len(unique),X_train.shape[1],X_train.shape[1]))
#print(covdif)
for i in range(X_train.shape[0]):
	tp=(X_train[i]-means[np.where(unique==Y_train[i]),:])
	le=np.dot(tp[0].T,tp[0])
	covdif[np.where(unique==Y_train[i]),:,:]=covdif[np.where(unique==Y_train[i]),:,:]+le


for i in range(len(unique)):
	covdif[i,:,:]=covdif[i,:,:]/counts[i]
	d= np.diag(covdif[i,:,:])
	covdif[i,:,:]=np.diag(d)


predicted.clear()

for i in range(len(Y_train)):
	predicted.append(predict_covdif(X_train[i],means,covdif,counts))

print("accuracy on training set with covariance matrix as difcov is "+str(accuracy_score(Y_train,predicted)*100))
#print("Confusion Matrix for training data is: ")
#print(confusion_matrix(Y_train, predicted))

predicted.clear()

for i in range(len(Y_valid)):
	predicted.append(predict_covdif(X_valid[i],means,covdif,counts))

print("accuracy on validation set with covariance matrix as difcov is "+str(accuracy_score(Y_valid,predicted)*100))
#print("Confusion Matrix for validation data is: ")
#print(confusion_matrix(Y_valid, predicted))

predicted.clear()

for i in range(len(Y_test)):
	predicted.append(predict_covdif(X_test[i],means,covdif,counts))

print("accuracy on test set with different covariance matrix(Best Model) is "+str(accuracy_score(Y_test,predicted)*100))


confuse=confusion_matrix(Y_test,predicted)

sn.heatmap(confuse/np.sum(confuse,axis=0), annot=True,
    fmt='.2%', cmap='Blues',cbar=False)
plt.xlabel('Predicted Class')
plt.ylabel("Actual Class")
plt.title('Confusion Matrix for Guassian Distribution with different covariance matrix on Testing data')
plt.savefig('Confusion_test_2.png')

plt.show()

predicted.clear()

for i in range(len(Y_train)):
	predicted.append(predict_covdif(X_train[i],means,covdif,counts))

confuse=confusion_matrix(Y_train,predicted)

sn.heatmap(confuse/np.sum(confuse,axis=0), annot=True,
    fmt='.2%', cmap='Blues',cbar=False)
plt.xlabel('Predicted Class')
plt.ylabel("Actual Class")
plt.title('Confusion Matrix for Guassian Distribution with different covariance matrix on Training data')
plt.savefig('Confusion_train_2.png')

plt.show()


x1=np.linspace(-15,15,num=400)
x2=np.linspace(-3,15,num=400)
xx1, xx2 = np.meshgrid(x1, x2)
r1, r2 = xx1.flatten(), xx2.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
grid = np.hstack((r1,r2))
#print(grid)
predicted.clear()
num_cores = multiprocessing.cpu_count()

predicted = Parallel(n_jobs=num_cores)(delayed(predict_covdif)(grid[i],means,covdif,counts) for i in range(grid.shape[0]))
pos=np.empty(xx1.shape+(2,))
pos[:,:,0]=xx1
pos[:,:,1]=xx2


predicted=np.array(predicted)
predicted=predicted.reshape(xx1.shape)
fig = plt.figure(figsize=(8,8))
plt.contourf(xx1, xx2, predicted, cmap='RdBu')
colors = ['green','red','blue','purple']
plt.scatter(X_train[:,0], X_train[:,1], c=Y_train, cmap=matplotlib.colors.ListedColormap(colors))
#plt.scatter(X_train[:,0], X_train[:,1], c=Y_train, cmap='RdBu')
for i in range(covdif.shape[0]):
    mid=multivariate_normal(mean=means[i],cov=covdif[i])
    plt.contour(xx1,xx2,mid.pdf(pos),[0.0075,0.05,0.1,0.15,0.2,0.23])

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Decision Region plot with different covariance matrix for each class')

plt.savefig('plot_Gaussian_2.png')
plt.show()
