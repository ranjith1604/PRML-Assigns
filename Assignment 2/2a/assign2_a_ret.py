import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans


def gaussian(X,mean,cov,K):
	n_samples, n_features = np.shape(X)
	Diff = np.reshape(X, (n_samples, 1, n_features) ) - np.reshape(mean, (1, K, n_features) )
	L = np.linalg.inv(cov)
	exponent = np.einsum("nkj,nkj->nk", np.einsum("nki,kij->nkj", Diff, L), Diff)
	#Nmat = np.exp(-0.5*exponent)/np.sqrt(np.linalg.det(cov))   / (2*np.pi)**(n_features/2)
	print(Diff.shape)
	return 1
	

#Taking input from csv file and taking x and y out
data=pd.read_csv("dataset/coast/train.csv")
#data=data.dropna()
data=data.to_numpy()


data=data[:,1:]
#print(data[0][0])
df = pd.DataFrame(data)
df=df.astype(float)
print(df.corr().iloc[0:,0:])
print(max((df.corr()).iloc[1:,1:]))
K=4 # number of clusters
#print(data.shape)
n_samples, n_features = np.shape(data)
rnd = np.random.RandomState(seed=None)
#Pi = np.ones(K)/K
means = data[rnd.choice(n_samples, size=K, replace=False)]
#Sigma = np.tile(np.diag(np.var(data, axis=0)), (K, 1, 1))

#print(Sigma.shape)

gam=np.zeros([data.shape[0],K])

convergence=True
count=0
while(convergence):
	tp=np.zeros([data.shape[0],K])
	for i in range(data.shape[0]):
		list=np.empty([K,1])
		for p in range(K):
			list[p]=(np.sum((data[i,:]-means[p])**2))
		tp[i][np.argmin(list,axis=0)]=1
	#print(tp)
	#print(means)
	
	for i in range(K):
		b=np.where(tp[:,i]==1)
		#print(np.sum(np.sum(data[b,:],axis=0),axis=0).shape)
		#print(len(b[0]))
		#print(data[b,:].shape)
		#print(means[i])
		means[i]=np.sum(np.sum(data[b,:],axis=0),axis=0)
		#print(means[i])
		#print(means[i]/len(b[0]))
		means[i]=means[i]/len(b[0])
		
	comparison= tp==gam
	if comparison.all():
		break
	else:
		count+=1
		gam=tp.copy()

N=np.sum(gam)		
cov2=np.zeros([K,data.shape[1],data.shape[1]])
#print(data.shape)
Wq=np.zeros([K,1])

for i in range(K):
	Nq=np.sum(gam[:,i])
	Wq[i]=Nq/N
	tp=np.zeros([data.shape[1],data.shape[1]])
	for p in range(data.shape[0]):
		le=data[p,:]-means[i]
		le=np.reshape(le,[le.shape[0],1])
		tp=tp+gam[p,i]*(np.dot(le,le.T))
	tp=tp/Nq
	cov2[i,:,:]=tp.copy()
	
#print(Wq.shape)
#print(cov2[0][0])
#print(gaussian(data,means,cov2,K))

		

