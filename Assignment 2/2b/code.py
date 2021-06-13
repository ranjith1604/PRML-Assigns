import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.patches as mpatches
from joblib import Parallel, delayed
import multiprocessing
from scipy.stats import multivariate_normal
import seaborn as sn
from matplotlib.collections import QuadMesh
import matplotlib.font_manager as fm


def gauss(X, mean_vector, covariance_matrix):
    if (np.abs(np.linalg.det(covariance_matrix))==0):
        print("ERROR")
    # a= (2*np.pi)**(-len(X)/2)*np.abs(np.prod((np.linalg.eigvals(covariance_matrix))))**(-1/2)*np.exp(-np.dot(np.dot((X-mean_vector).T, np.linalg.pinv(covariance_matrix)), (X-mean_vector))/2)
    b= (2*np.pi)**(-len(X)/2)*(np.linalg.det(covariance_matrix))**(-1/2)*np.exp(-np.dot(np.dot((X-mean_vector).T, np.linalg.inv(covariance_matrix)), (X-mean_vector))/2)
    # c= ((1/(((2*math.pi)**(X.shape[0]/2))*((np.linalg.det(covariance_matrix))**0.5)))*math.exp(-0.5*np.matmul(np.matmul((X-mean_vector).T,np.linalg.pinv(covariance_matrix)),(X-mean_vector))))
    # return (2*np.pi)**(-len(X)/2)*np.linalg.det(covariance_matrix)**(-1/2)*np.exp(-np.dot(np.dot((X-mean_vector).T, np.linalg.inv(covariance_matrix)), (X-mean_vector))/2)

    return b

def KNN_class(data,k):
    #k=4 # number of clusters
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

    	for i in range(k):
    		b=np.where(tp[:,i]==1)
    		means[i]=np.sum(np.sum(data[b,:],axis=0),axis=0)
    		means[i]=means[i]/len(b[0])

    	comparison= tp==z_prev
    	if comparison.all():
    		break
    	else:
    		count+=1
    		z_prev=tp.copy()
    return means,z_prev

def GMM_classifier(X,means,weights,cov,k):
    ll_n=[]
    for i in range(3):
#             ll= np.log(sum([weights[i][j]*gauss(X_valid[n], means[i][j], cov[i][j])  for j in range(k)])) + np.log(prior_class[i])
        ll= np.log(sum([weights[i][j]*gauss(X, means[i][j], cov[i][j])  for j in range(k)]))
        ll_n.append(ll)
    ll_n=np.array(ll_n)

    return np.argmax(ll_n)


labels=['coast','forest','opencountry','street','tallbuilding']


#for label in labels:
arr = os.listdir('./'+labels[0]+'/train')

print(arr)
coast_train=pd.read_csv('coast/train/'+arr[0],header=None,delim_whitespace=True)
for i in range(1,len(arr)):
    data=pd.read_csv('coast/train/'+arr[i],header=None,delim_whitespace=True)
    #print(data.shape)
    #coast_train.concat(data)
    coast_train=pd.concat([coast_train,data],ignore_index=True)

print(coast_train)
