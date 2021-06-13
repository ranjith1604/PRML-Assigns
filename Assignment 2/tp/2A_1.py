import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

size=[]
weights=[]
cov=[]
means=[]

# def gauss(X, mean_vector, covariance_matrix):
#     if (np.abs(np.linalg.det(covariance_matrix))==0):
#         print("ERROR")
#     # a= (2*np.pi)**(-len(X)/2)*np.abs(np.prod((np.linalg.eigvals(covariance_matrix))))**(-1/2)*np.exp(-np.dot(np.dot((X-mean_vector).T, np.linalg.pinv(covariance_matrix)), (X-mean_vector))/2)
#     b= (2*np.pi)**(-len(X)/2)*(np.linalg.det(covariance_matrix))**(-1/2)*np.exp(-np.dot(np.dot((X-mean_vector).T, np.linalg.inv(covariance_matrix)), (X-mean_vector))/2)
#     # c= ((1/(((2*math.pi)**(X.shape[0]/2))*((np.linalg.det(covariance_matrix))**0.5)))*math.exp(-0.5*np.matmul(np.matmul((X-mean_vector).T,np.linalg.pinv(covariance_matrix)),(X-mean_vector))))
#     # return (2*np.pi)**(-len(X)/2)*np.linalg.det(covariance_matrix)**(-1/2)*np.exp(-np.dot(np.dot((X-mean_vector).T, np.linalg.inv(covariance_matrix)), (X-mean_vector))/2)

#     return b

# The only hyperparameter is k ( no.of components for each class)
k=3
train=['coast','forest','opencountry','street','tallbuilding']
#train=['coast']
for c, train_file in enumerate(train):
    data=pd.read_csv('dataset/'+train_file+'/train.csv')
    data=data.to_numpy()
    X=data[:,1:]
    size.append(len(X))
    print(f"\n\n\nClass {c}")
#     print(size)
    kmeans=KMeans(n_clusters=k,random_state=0).fit(X)
    # kmeans=KMeans(n_clusters=k).fit(X)
    means_old=kmeans.cluster_centers_
    labels=kmeans.labels_



    N=len(X)
    r_old=np.zeros((len(X),k)) # form a Z ( indicator ) matrix

    for i in range(len(X)):
        r_old[i,labels[i]]=1

    Nq_old=np.sum(r_old,axis=0) # sum conatins the number of elements belonging
                                 # to each cluster

    print("\nOriginal effective number of elements in each cluster")
    print(Nq_old)
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
        ll_old = ll_old + np.log(sum([Wq_old[j]*multivariate_normal.pdf(X[n], means_old[j], cov_old[j],allow_singular=True) for j in range(k)]))

    print(f"\nInitial log-likehood = {ll_old}")

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
                r_new[n][i] = Wq_old[i] * multivariate_normal.pdf(X[n], means_old[i], cov_old[i],allow_singular=True)
                r_new[n][i] /= sum([Wq_old[j]*multivariate_normal.pdf(X[n], means_old[j], cov_old[j],allow_singular=True) for j in range(k)])

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
#             d= np.diag(tp)
#             tp=np.diag(d)
            cov_new[i,:,:]=tp.copy()


        # print(f"\nRun= {run}\n")
#         print(np.sum(Nq_new))
#         print("\nWeights\n")
#         print(np.sum(Wq_new))
#         print(Wq_new)
#         print(np.sum(r_new))
#         print("\n------------------")

        # Calculating log-likelhood
        ll_new=0
        for n in range(len(X)):
            ll_new = ll_new + np.log(sum([Wq_new[j]*multivariate_normal.pdf(X[n], means_new[j], cov_new[j],allow_singular=True) for j in range(k)]))

    #     print(ll_new)
        diff=ll_new-ll_old

    #     print(diff)

        #Convergence condition
        if diff < 1e-1:
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
    print(f"Final log-likehood = {ll_new}")

    print("\nEffective number of elements in each cluster is")
    print(Nq_new)
#     ass=np.sum(Nq_new)
#     print(ass)
    weights.append(Wq_new)
    means.append(means_new)
    cov.append(cov_new)

print("\n##############################################################################")


# prior_class=size/sum(size)
# print(prior_class)
# print(len(wieghts))
# print(weights[0])


# validation_set=["dev_1.csv","dev_2.csv","dev_3.csv","dev_4.csv","dev_5.csv"]

# for ind, valid_file in enumerate(validation_set):

#     X_valid=pd.read_csv(valid_file)
#     X_valid=X_valid.to_numpy()
#     X_valid=X_valid[:,1:]

#     ll_n=[]
#     index=[]

#     for n in range(len(X_valid)):
#         for i in range(len(validation_set)):
#             ll= np.log(sum([weights[i][j]*gauss(X_valid[n], means[i][j], cov[i][j])  for j in range(k)])) + np.log(prior_class[i])
#             ll_n.append(ll)
#         ll_n=np.array(ll_n)
#         index.append(argmax(ll_n))

#     p=index.count(ind)
#     prob=p/len(index)

#     print(probab)







# In[ ]:







size=np.array(size)
prior_class=size/np.sum(size)

validation_set=['coast','forest','opencountry','street','tallbuilding']
# validation_set=['train_1.csv','train_2.csv','train_3.csv','train_4.csv','train_5.csv']

for ind, valid_file in enumerate(validation_set):

    X_valid=pd.read_csv('dataset/'+valid_file+'/dev.csv')
    X_valid=X_valid.to_numpy()
    X_valid=X_valid[:,1:]

    index=[]

    for n in range(len(X_valid)):
        ll_n=[]
        for i in range(len(validation_set)):
            ll= np.log(sum([weights[i][j]*multivariate_normal.pdf(X_valid[n], means[i][j], cov[i][j],allow_singular=True)  for j in range(k)])) + np.log(prior_class[i])
            ll_n.append(ll)
        ll_n=np.array(ll_n)
        index.append(np.argmax(ll_n))

    p=index.count(ind)
    prob=p/len(index)

    print(prob)


# In[6]:


# for j in range(5):
#     for i in range(k):
#         print(np.diag(cov[2][i]))
#     print("\n###############################################################################\n")





# In[ ]:
