import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal



def gauss(X, mean_vector, covariance_matrix):
    '''
        This function implements the multivariat normal derivation formula,
        the normal distribution for vectors it requires the following parameters
            :param X: 1-d numpy array
                The row-vector for which we want to calculate the distribution
            :param mean_vector: 1-d numpy array
                The row-vector that contains the means for each column
            :param covariance_matrix: 2-d numpy array (matrix)
                The 2-d matrix that contain the covariances for the features
    '''
    if (np.abs(np.linalg.det(covariance_matrix))==0):
        print("ERROR")
#     #         a= (2*np.pi)**(-len(X)/2)*np.abs(np.prod((np.linalg.eigvals(covariance_matrix))))**(-1/2)*np.exp(-np.dot(np.dot((X-mean_vector).T, np.linalg.pinv(covariance_matrix)), (X-mean_vector))/2)
    b= (2*np.pi)**(-len(X)/2)*(np.linalg.det(covariance_matrix))**(-1/2)*np.exp(-np.dot(np.dot((X-mean_vector).T, np.linalg.inv(covariance_matrix)), (X-mean_vector))/2)
#     #         c= ((1/(((2*math.pi)**(X.shape[0]/2))*((np.linalg.det(covariance_matrix))**0.5)))*math.exp(-0.5*np.matmul(np.matmul((X-mean_vector).T,np.linalg.pinv(covariance_matrix)),(X-mean_vector))))
#     return (2*np.pi)**(-len(X)/2)*np.linalg.det(covariance_matrix)**(-1/2)*np.exp(-np.dot(np.dot((X-mean_vector).T, np.linalg.inv(covariance_matrix)), (X-mean_vector))/2)
    return b
def gaussian(x,mean,cov):
	return ((1/(((2*math.pi)**(x.shape[0]/2))*((np.linalg.det(cov))**0.5)))*math.exp(-0.5*np.matmul(np.matmul((x-mean).T,np.linalg.inv(cov)),(x-mean))))

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
    return means,z_prev

data=pd.read_csv("19/train.csv",header=None)

#data=data.dropna()
data=data.to_numpy()
X=data[:,1:]
#print(data)
k=2

means_old,r_old=KNN_class(X,k)

N=len(X)

#print(means_old.shape)

Nq_old=np.sum(r_old,axis=0) # sum conatins the number of elements belonging
                             # to each cluster

#print(r_old)

# Initialization

#cov2 is a 3-d array containing the covariance matrix of each cluster
cov_old=np.zeros([k,X.shape[1],X.shape[1]])
Wq_old =np.zeros([k,1]) ## weight of each cluster
#print(Nq_old)
for i in range(k):
    Nq=Nq_old[i]
    Wq_old[i]= Nq/N
    tp=np.zeros([X.shape[1],X.shape[1]])

    for p in range(X.shape[0]):
        le=X[p,:]-means_old[i]
        le=np.reshape(le,[le.shape[0],1])
        tp=tp+r_old[p,i]*(np.dot(le,le.T))
    tp=tp/Nq

    d= np.diag(tp)
    tp=np.diag(d)
    cov_old[i,:,:]=tp.copy()

ll_old= 0.0
for n in range(len(X)):
    ll_old = ll_old + np.log(sum([Wq_old[j]*gaussian(X[n],means_old[j],cov_old[j]) for j in range(k)]))
print(ll_old)

convergence=False
iter_convergence=0
run=0
runs=1000
epsilon=100

r_new = np.zeros((len(X), k))  # responsibilty matrix

# Calculating the r matrix
"""
for n in range(len(X)):
    for i in range(k):
        r_new[n][i] = Wq_old[i] * gaussian(X[n],means_old[i],cov_old[i])
        r_new[n][i] /= sum([Wq_old[j]*gaussian(X[n],means_old[j],cov_old[j]) for j in range(k)])

# Calculating the N effective elemts fro each component
Nq_new = np.sum(r_new, axis=0)
print(Nq_new)
"""
while (convergence == False and run<runs):

    # ''' --------------------------   E - STEP   -------------------------- '''

    # Initiating the r matrix, every row contains the probabilities
    # for every cluster for this row

    r_new = np.zeros((len(X), k))  # responsibilty matrix

    # Calculating the r matrix
    for n in range(len(X)):
        for i in range(k):
            r_new[n][i] = Wq_old[i] * gaussian(X[n],means_old[i],cov_old[i])
            r_new[n][i] /= sum([Wq_old[j]*gaussian(X[n],means_old[j],cov_old[j]) for j in range(k)])

    # Calculating the N effective elemts fro each component
    Nq_new = np.sum(r_new, axis=0)
    #print(r_new)

    # ''' --------------------------   M - STEP   -------------------------- '''


    # Updating the weights list
    Wq_new =np.zeros([k,1]) ## weight of each cluster
    for i in range(k):
        Wq_new[i]= Nq_new[i]/ N


    # Initializing the mean vector as a zero vector
    means_new = np.zeros((k, len(X[0])))

    # Updating the mean vector
    for i in range(k):
        for n in range(len(X)):
            means_new[i] = means_new[i] + r_new[n][i] * X[n]
        means_new[i] = means_new [i]/Nq_new[i]



    # Initiating the list of the covariance matrixes
    cov_new =np.zeros([k,X.shape[1],X.shape[1]])

    # Updating the covariance matrices
    for i in range(k):
        Nq=Nq_new[i]
        tp=np.zeros([X.shape[1],X.shape[1]])

        for p in range(X.shape[0]):
            le=X[p,:]-means_new[i]
            le=np.reshape(le,[le.shape[0],1])
            tp=tp+r_new[p,i]*(np.dot(le,le.T))

        tp=tp/Nq
        d= np.diag(tp)
        tp=np.diag(d)
        cov_new[i,:,:]=tp.copy()


#     print(f"\nRun= {run}\n")
#     print(np.sum(Nq_new))
#     print("\nWeights\n")
#     print(np.sum(Wq_new))
#     print(Wq_new)
#     print(np.sum(r_new))
#     print("\n------------------")

    # Calculating log-likelhood
    ll_new=0
    for n in range(len(X)):
        ll_new = ll_new +  np.log(sum([Wq_new[j]*gaussian(X[n],means_old[j],cov_old[j]) for j in range(k)]))

    #print(ll_new)
    diff=ll_new-ll_old

    print(diff)

    #Convergence condition
    if diff < 1e-5:
        iter_convergence=run
        convergence=True
        break

    else:
        ll_old=ll_new.copy()
        Wq_old= Wq_new.copy()
        means_old=means_new.copy()
        cov_old=cov_new.copy()

    run= run +1

if convergence==True and run!=runs:
    print("Iterations for convergence=",iter_convergence)
else:
    print("Estimate has not converged yet, more runs needed")

"""

valid=pd.read_csv("dev_1.csv")
#data=data.dropna()
valid=valid.to_numpy()
X_v=valid[:,1:]

ll1=0
for n in range(len(X_v)):
    ll1= ll1+ (sum([Wq_old[j]*gaussian(X[n],means_old[j],cov_old[j]) for j in range(k)]))


"""
#
# valid=pd.read_csv("dev_2.csv")
# #data=data.dropna()
# valid=valid.to_numpy()
# X_v=valid[:,1:]
#
# ll2=0
# for n in range(len(X_v)):
#     ll2= ll2+ (sum([Wq_old[j]*gauss(X_v[n], means_old[j], cov_old[j]) for j in range(k)]))
#
#
# valid=pd.read_csv("dev_3.csv")
# #data=data.dropna()
# valid=valid.to_numpy()
# X_v=valid[:,1:]
#
# ll3=0
# for n in range(len(X_v)):
#     ll3= ll3+ (sum([Wq_old[j]*gauss(X_v[n], means_old[j], cov_old[j]) for j in range(k)]))
#
#
# valid=pd.read_csv("dev_4.csv")
# #data=data.dropna()
# valid=valid.to_numpy()
# X_v=valid[:,1:]
#
# ll4=0
# for n in range(len(X_v)):
#     ll4= ll4+ (sum([Wq_old[j]*gauss(X_v[n], means_old[j], cov_old[j]) for j in range(k)]))
#
# valid=pd.read_csv("dev_5.csv")
# #data=data.dropna()
# valid=valid.to_numpy()
# X_v=valid[:,1:]
#
# ll5=0
# for n in range(len(X_v)):
#     ll5= ll5+ (sum([Wq_old[j]*gauss(X_v[n], means_old[j], cov_old[j]) for j in range(k)]))
