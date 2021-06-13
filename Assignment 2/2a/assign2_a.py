import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
def gaussian(x,mean,cov):
	#print(x.shape)
	#print(mean.shape)
	#x=np.reshape(x,[x.shape[0],1])
	#mean=np.reshape(mean,[mean.shape[0],1])
	#print(x.shape)
	#print(np.linalg.inv(cov))
	#print(mean)
	tp=np.matmul(np.matmul((x-mean).T,np.linalg.inv(cov)),(x-mean))

	#print(tp)
	tp=np.exp(-tp[0][0]/2)

	mid=((np.linalg.det(cov))**0.5)
	#print(np.linalg.inv(cov))
	#if np.isnan(mid):
	#	mid=1e-7

	tp=tp/mid
	#print(((2*math.pi)**(x.shape[0]/2)))
	tp=tp/((2*math.pi)**(x.shape[0]/2))
	#print(tp)
	return tp


"""def gaussian(x,mean,cov):
	return ((1/(((2*math.pi)**(x.shape[0]/2))*((np.linalg.det(cov))**0.5)))*math.exp(-0.5*np.matmul(np.matmul((x-mean).T,np.linalg.inv(cov)),(x-mean))))
"""
#Taking input from csv file and taking x and y out
data=pd.read_csv("dataset/coast/train.csv")
data=data.to_numpy()


data=data[:,1:]
#print(data)

k=4 # number of clusters


#print(data.shape[0])
means = data[np.random.choice(range(data.shape[0]), k, replace=False),:]
#print(means[0])

z_prev=np.zeros([data.shape[0],k])

convergence=True
count=0
while(convergence):
	tp=np.zeros([data.shape[0],k])
	for i in range(data.shape[0]):
		list=np.empty([k,1])
		for p in range(k):
			list[p]=(np.sum((data[i,:]-means[p])**2))
		tp[i][np.argmin(list,axis=0)]=1
	#print(tp)
	#print(means)

	for i in range(k):
		b=np.where(tp[:,i]==1)
		#print(np.sum(np.sum(data[b,:],axis=0),axis=0).shape)
		#print(len(b[0]))
		#print(data[b,:].shape)
		#print(means[i])
		means[i]=np.sum(np.sum(data[b,:],axis=0),axis=0)
		#print(means[i])
		#print(means[i]/len(b[0]))
		means[i]=means[i]/len(b[0])

	comparison= tp==z_prev
	if comparison.all():
		break
	else:
		count+=1
		z_prev=tp.copy()

#print(means)
#print(count)
"""chk=[]
for i in range(data.shape[0]):
	vale=[]
	for p in range(k):
		vale.append(np.sum((data[i,:]-means[p])**2))
	chk.append(vale.index(min(vale)))

print(chk)
"""

#kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
#print(kmeans.fit_predict(data))

Wq=np.zeros([k,1])
#cov=np.zeros([k,data.shape[1],data.shape[1]])

N=np.sum(z_prev)

cov2=np.zeros([k,data.shape[1],data.shape[1]])
#print(data.shape)
for i in range(k):
	Wq[i]=np.sum(z_prev[:,i])/N
	tp=np.zeros([data.shape[1],data.shape[1]])
	for p in range(data.shape[0]):
		le=data[p,:]-means[i]
		le=np.reshape(le,[le.shape[0],1])
		tp=tp+z_prev[p,i]*(np.dot(le,le.T))
	tp=tp/np.sum(z_prev[:,i])
	cov2[i,:,:]=tp.copy()


#print(le.shape)
print(cov2[0])
#print(z_prev)

likeli_prev=0

for i in range(data.shape[0]):
		sumv=0
		data1=np.reshape(data[i,:],[data[i,:].shape[0],1])
		for p in range(k):
			sumv=sumv+(Wq[p,0]*gaussian(data1,np.reshape(means[p],[means[p].shape[0],1]),cov2[p]))
		#print(means[p].shape)
		sumv=np.log(sumv)

		likeli_prev=likeli_prev+sumv

while(True):
	for i in range(data.shape[0]):
		sumv=0
		data1=np.reshape(data[i,:],[data[i,:].shape[0],1])
		for p in range(k):
			sumv=sumv+(Wq[p,0]*gaussian(data1,np.reshape(means[p],[means[p].shape[0],1]),cov2[p]))
		for p in range(k):
			z_prev[i][p]=(Wq[p,0]*gaussian(data1,np.reshape(means[p],[means[p].shape[0],1]),cov2[p]))/sumv


	#print(np.sum(z_prev[3,:]))
	N=np.sum(z_prev)
	#means[i]=z_prev[:,0]*data
	#le=z_prev[:,0]
	#le=np.reshape(le,[le.shape[0],1])
	#print(le[0])
	#print(data[0])
	#print(((le*data).sum(axis=0)).shape)
	for i in range(k):
		Wq[i]=np.sum(z_prev[:,i])/N
		le=z_prev[:,i]
		le=np.reshape(le,[le.shape[0],1])
		le=((le*data).sum(axis=0))
		le=le/np.sum(z_prev[i])
		le=np.reshape(le,[1,le.shape[0]])
		means[i]=le
		tp=np.zeros([data.shape[1],data.shape[1]])
		for p in range(data.shape[0]):
			le=data[p,:]-means[i]
			le=np.reshape(le,[le.shape[0],1])
			tp=tp+z_prev[p,i]*(np.dot(le,le.T))
		tp=tp/np.sum(z_prev[:,i])
		cov2[i,:,:]=tp.copy()
	likeli=0
	for i in range(data.shape[0]):
		sumv=0
		data1=np.reshape(data[i,:],[data[i,:].shape[0],1])
		for p in range(k):
			sumv=sumv+(Wq[p,0]*gaussian(data1,np.reshape(means[p],[means[p].shape[0],1]),cov2[p]))
		#print(means[p].shape)
		sumv=np.log(sumv)

		likeli=likeli+sumv

	if abs(likeli-likeli_prev) < 0.0001:
		break
	else:
		likeli_prev=likeli
	#print(likeli)
