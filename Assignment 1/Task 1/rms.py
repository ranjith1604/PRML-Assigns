import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


#Taking input from csv file and taking x and y out
data=pd.read_csv("function3.csv")
data=data.to_numpy()
data=np.delete(data,0,axis=1)

size=200 #Size of dataset

x=data[:size,0]
y=data[:size,1]

totsize=math.floor(size*10/7) #Finding the total size

#Validation data
x_valid=data[size+1:size+math.floor(totsize/5)+1,0]
y_valid=data[size+1:size+math.floor(totsize/5)+1,1]

#Testing data
x_test=data[size+math.floor(totsize/5)+1:size+math.floor(totsize/5)+math.floor(totsize/10)+1,0]
y_test=data[size+math.floor(totsize/5)+1:size+math.floor(totsize/5)+math.floor(totsize/10)+1,1]


powers=[2,3,6,9] #Powers to find rms for

finvals1=[]
finvals2=[]
finvals3=[]

#The below for loop calculates rms for training data
for power in powers:
	
	#Calculating the Weights
	A=np.empty((power+1,power+1))

	for i in range(power+1):
		for p in range(power+1):
			A[i][p]=np.sum(np.power(x,i+p))

	C=np.empty((power+1,1))

	for i in range(power+1):
		C[i]=np.dot(y.T,np.power(x,i))

	W=np.linalg.solve(A,C)
	fx=[]
	for i in range(len(x)):
		tp=0
		for p in range(power+1):
			tp+=W[p]*(x[i]**p)
		fx.append(tp)
	fx=np.array(fx)
	fx=fx.reshape(fx.shape[0],)
	rms=np.sqrt(np.mean((fx-y)**2,axis=0))
	finvals1.append(rms)
	
#The below for loop calculates rms for validation data	
for power in powers:
	A=np.empty((power+1,power+1))

	for i in range(power+1):
		for p in range(power+1):
			A[i][p]=np.sum(np.power(x,i+p))

	C=np.empty((power+1,1))

	for i in range(power+1):
		C[i]=np.dot(y.T,np.power(x,i))

	W=np.linalg.solve(A,C)
	fx=[]
	for i in range(len(x_valid)):
		tp=0
		for p in range(power+1):
			tp+=W[p]*(x_valid[i]**p)
		fx.append(tp)
	fx=np.array(fx)
	fx=fx.reshape(fx.shape[0],)
	rms=np.sqrt(np.mean((fx-y_valid)**2,axis=0))
	finvals2.append(rms)
	
	
#The below for loop calculates rms for testing data	
for power in powers:
	A=np.empty((power+1,power+1))

	for i in range(power+1):
		for p in range(power+1):
			A[i][p]=np.sum(np.power(x,i+p))

	C=np.empty((power+1,1))

	for i in range(power+1):
		C[i]=np.dot(y.T,np.power(x,i))

	W=np.linalg.solve(A,C)
	fx=[]
	for i in range(len(x_test)):
		tp=0
		for p in range(power+1):
			tp+=W[p]*(x_test[i]**p)
		fx.append(tp)
	fx=np.array(fx)
	fx=fx.reshape(fx.shape[0],)
	rms=np.sqrt(np.mean((fx-y_test)**2,axis=0))
	finvals3.append(rms)
	

#Plotting the graph	
plt.plot(powers,finvals1,color='red',marker='o',label='Training Data')
plt.plot(powers,finvals2,color='blue',marker='o',label='Validation Data')
plt.plot(powers,finvals3,color='green',marker='o',label='Testing Data')
plt.xlabel('M')
plt.ylabel('$E_{rms}$')
plt.legend()
plt.title('$E_{rms}$ value for N=200')
plt.savefig('Task1_N=200.png')
plt.show()
	