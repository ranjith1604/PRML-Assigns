import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Taking input from csv file and taking x and y out
data=pd.read_csv("function3.csv")
data=data.to_numpy()
data=np.delete(data,0,axis=1)
x=data[:200,0] #The size is set here in x and y
y=data[:200,1]
power=2 #Setting the degree

#Calculating the Weights
A=np.empty((power+1,power+1))

for i in range(power+1):
	for p in range(power+1):
		A[i][p]=np.sum(np.power(x,i+p))

C=np.empty((power+1,1))

for i in range(power+1):
		C[i]=np.dot(y.T,np.power(x,i))
		

W=np.linalg.solve(A,C)

#X values for plotting
X=np.linspace(-2,2,num=100)

#finding the outputs for differnt X in trained model
fx=[]
print(x)
for i in range(len(X)):
	tp=0
	for p in range(power+1):
		tp+=W[p]*(X[i]**p)
	fx.append(tp)
	
#Plotting the graph
plt.plot(X,fx,color='red',label='Polynomial Curve')
plt.scatter(x,y,marker='o',label='Dataset Points')
plt.xlabel('x')
plt.ylabel('t')
plt.text(2,34,'M=2',verticalalignment='top',horizontalalignment='right',fontsize=15)
plt.legend()
plt.savefig('M=2_N=200.png')
plt.show()

