import numpy as np
import pandas as pd
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
    #np.random.seed(0)
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

def GMM_classifier(X,means,weights,cov,k):
    ll_n=[]
    for i in range(3):
#             ll= np.log(sum([weights[i][j]*gauss(X_valid[n], means[i][j], cov[i][j])  for j in range(k)])) + np.log(prior_class[i])
        ll= np.log(sum([weights[i][j]*gauss(X, means[i][j], cov[i][j])  for j in range(k)]))
        ll_n.append(ll)
    ll_n=np.array(ll_n)
#         print(ll_n)

    return np.argmax(ll_n)



K=[2,3,4]

size_best=[]
weights_best=[]
cov_best=[]
means_best=[]


for k in K:

    data=pd.read_csv("19/train.csv",header=None)

    data.columns =['x1', 'x2','Class']

    data1= data[data['Class']==0]
    data2= data[data['Class']==1]
    data3= data[data['Class']==2]

    X_data=[data1, data2, data3]



    size=[]
    weights=[]
    cov=[]
    means=[]


    # The only hyperparameter is k ( no.of components for each class)


    for c, X in enumerate (X_data):
        X= X.to_numpy()
        X= X[:,:-1]
    #     print(X)
        size.append(len(X))
        print(f"\nClass {c}\n")
    #     print(size)

        means_old,r_old=KNN_class(X,k)

        N=len(X)

        Nq_old=np.sum(r_old,axis=0) # sum conatins the number of elements belonging
                                     # to each cluster

        # Initialization

        #cov2 is a 3-d array containing the covariance matrix of each cluster
        cov_old=np.zeros([k,X.shape[1],X.shape[1]])
        Wq_old =np.zeros([k,1]) ## weight of each cluster

        for i in range(k):
            Nq=Nq_old[i]
            Wq_old[i]= Nq/N
            tp=np.zeros([X.shape[1],X.shape[1]])

            for p in range(X.shape[0]):
                le=X[p,:]-means_old[i]
                le=np.reshape(le,[le.shape[0],1])
                tp=tp+r_old[p,i]*(np.dot(le,le.T))
            tp=tp/Nq

    #         d= np.diag(tp)
    #         tp=np.diag(d)
            cov_old[i,:,:]=tp.copy()

        ll_old= 0.0
        for n in range(len(X)):
            ll_old = ll_old + np.log(sum([Wq_old[j]*gauss(X[n], means_old[j], cov_old[j]) for j in range(k)]))

        #print(ll_old)

        convergence=False
        iter_convergence=0
        run=0
        runs=1000
        epsilon=100

        while (convergence == False and run<runs):

            # ''' --------------------------   E - STEP   -------------------------- '''

            # Initiating the r matrix, every row contains the probabilities
            # for every cluster for this row

            r_new = np.zeros((len(X), k))  # responsibilty matrix

            # Calculating the r matrix
            for n in range(len(X)):
                for i in range(k):
                    r_new[n][i] = Wq_old[i] * gauss(X[n], means_old[i], cov_old[i])
                    r_new[n][i] /= sum([Wq_old[j]*gauss(X[n], means_old[j], cov_old[j]) for j in range(k)])

            # Calculating the N effective elemts fro each component
            Nq_new = np.sum(r_new, axis=0)


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
                #d= np.diag(tp)
                #tp=np.diag(d)
                cov_new[i,:,:]=tp.copy()




            # Calculating log-likelhood
            ll_new=0
            for n in range(len(X)):
                ll_new = ll_new + np.log(sum([Wq_new[j]*gauss(X[n], means_new[j], cov_new[j]) for j in range(k)]))

        #     print(ll_new)
            diff=ll_new-ll_old

            #print(diff)

            #Convergence condition
            if diff < 1e-2:
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

        #print(ll_new)


        weights.append(Wq_new)
        means.append(means_new)
        cov.append(cov_new)

    print("##############################################################################")


    data=pd.read_csv("19/train.csv",header=None)


    data.columns =['x1', 'x2','Class']



    data1= data[data['Class']==0]
    data2= data[data['Class']==1]
    data3= data[data['Class']==2]



    X_data=[data1, data2, data3]

    prob=0
    tot=len(data)
    predicted=[]
    real=[]
    for ind, X_valid in enumerate(X_data):

        X_valid= X_valid.to_numpy()
        X_valid= X_valid[:,:-1]



        index=[]

        for n in range(len(X_valid)):
            ll_n=[]
            for i in range(3):
    #             ll= np.log(sum([weights[i][j]*gauss(X_valid[n], means[i][j], cov[i][j])  for j in range(k)])) + np.log(prior_class[i])
                ll= np.log(sum([weights[i][j]*gauss(X_valid[n], means[i][j], cov[i][j])  for j in range(k)]))
                ll_n.append(ll)
            ll_n=np.array(ll_n)
    #         print(ll_n)
            index.append(np.argmax(ll_n))
            predicted.append(np.argmax(ll_n))
            real.append(ind)
    #     print(len(index))
    #     print(index)
        p=index.count(ind)
        prob+=p
        #print(prob)

    #print(X_valid)
    print("accuracy for k="+str(k)+" using GMM with full covariance matrix on Training set is "+str(prob/tot*100))

    if k==4:
        confuse=confusion_matrix(real,predicted)

        sn.heatmap(confuse/np.sum(confuse,axis=0), annot=True,
            fmt='.2%', cmap='Blues',cbar=False)
        plt.xlabel('Predicted Class')
        plt.ylabel("Actual Class")
        plt.title('Confusion Matrix for GMM with k=4 on Training data')
        plt.savefig('Confusion_train_2.png')

        plt.show()




    data=pd.read_csv("19/dev.csv",header=None)
    data.columns =['x1', 'x2','Class']
    #print(len(data))

    X_valid=data.loc[np.r_[0:15, 30:45, 60:75],:]

    data1= X_valid[X_valid['Class']==0]
    data2= X_valid[X_valid['Class']==1]
    data3= X_valid[X_valid['Class']==2]

    X_data=[data1, data2, data3]
    prob=0
    tot=len(X_valid)
    for ind, X_valid in enumerate(X_data):

        X_valid= X_valid.to_numpy()
        X_valid= X_valid[:,:-1]



        index=[]

        for n in range(len(X_valid)):
            ll_n=[]
            for i in range(3):
    #             ll= np.log(sum([weights[i][j]*gauss(X_valid[n], means[i][j], cov[i][j])  for j in range(k)])) + np.log(prior_class[i])
                ll= np.log(sum([weights[i][j]*gauss(X_valid[n], means[i][j], cov[i][j])  for j in range(k)]))
                ll_n.append(ll)
            ll_n=np.array(ll_n)
    #         print(ll_n)
            index.append(np.argmax(ll_n))
    #     print(len(index))
    #     print(index)
        p=index.count(ind)
        prob+=p
        #print(prob)

    #print(X_valid)
    print("accuracy for k="+str(k)+" using GMM with full covariance matrix on validation set is "+str(prob/tot*100))

    size_best=size.copy()
    weights_best=weights.copy()
    cov_best=cov.copy()
    means_best=means.copy()

data=pd.read_csv("19/dev.csv",header=None)
data.columns =['x1', 'x2','Class']
#print(len(data))
k=4
X_valid=data.loc[np.r_[15:30, 45:60, 75:90],:]

data1= X_valid[X_valid['Class']==0]
data2= X_valid[X_valid['Class']==1]
data3= X_valid[X_valid['Class']==2]

X_data=[data1, data2, data3]
prob=0
tot=len(X_valid)
predicted=[]
real=[]
for ind, X_valid in enumerate(X_data):

    X_valid= X_valid.to_numpy()
    X_valid= X_valid[:,:-1]



    index=[]

    for n in range(len(X_valid)):
        ll_n=[]
        for i in range(3):
#             ll= np.log(sum([weights[i][j]*gauss(X_valid[n], means[i][j], cov[i][j])  for j in range(k)])) + np.log(prior_class[i])
            ll= np.log(sum([weights[i][j]*gauss(X_valid[n], means_best[i][j], cov_best[i][j])  for j in range(k)]))
            #print(X_valid[n])
            ll_n.append(ll)                                                                                  #k
        ll_n=np.array(ll_n)
        index.append(np.argmax(ll_n))
        predicted.append(np.argmax(ll_n))
        real.append(ind)

    p=index.count(ind)
    prob+=p
#     print(len(index))
    #print(prob)


print("accuracy for k="+str(4)+" using GMM with full covariance matrix on Testing set is "+str(prob/tot*100))
confuse=confusion_matrix(real,predicted)

sn.heatmap(confuse/np.sum(confuse,axis=0), annot=True,
    fmt='.2%', cmap='Blues',cbar=False)
plt.xlabel('Predicted Class')
plt.ylabel("Actual Class")
plt.title('Confusion Matrix for GMM with k=4 on Testing data')
plt.savefig('Confusion_test_2.png')

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
#GMM_classifier(X,means,cov,k)
predicted = Parallel(n_jobs=num_cores)(delayed(GMM_classifier)(grid[i],means_best,weights_best,cov_best,k) for i in range(grid.shape[0]))
pos=np.empty(xx1.shape+(2,))
pos[:,:,0]=xx1
pos[:,:,1]=xx2


predicted=np.array(predicted)
predicted=predicted.reshape(xx1.shape)
fig = plt.figure(figsize=(8,8))
plt.contourf(xx1, xx2, predicted, cmap='RdBu')
colors = ['green','red','blue','purple']

data=pd.read_csv("19/train.csv",header=None)
X_train= data.to_numpy()
#data.columns =['x1', 'x2','Class']


plt.scatter(X_train[:,0], X_train[:,1], c=X_train[:,2], cmap=matplotlib.colors.ListedColormap(colors))
#plt.scatter(X_train[:,0], X_train[:,1], c=Y_train, cmap='RdBu')

for i in range(3):
    for j in range(k):
        mid=multivariate_normal(mean=means_best[i][j],cov=cov_best[i][j])
        plt.contour(xx1,xx2,mid.pdf(pos),[0.3,0.5,0.7,0.8,0.85,0.9,0.95,1,1.1,1.15])

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Decision Region plot for GMM with k=4 for each class using full covariance matrix')

plt.savefig('plot_GMM_Full_2.png')
plt.show()
