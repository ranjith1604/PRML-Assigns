import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def calci(X1,X2,W,power): #function to find the output for an input
	ct=0
	val=0
	for p in range(power+1):
		tp=p
		while tp>=0:
			val=val+W[ct]*(X1**tp)*(X2**(p-tp))
			tp=tp-1
			ct=ct+1
	
	return val
	
	

if __name__ == "__main__":
	
	#The below lines take the csv file as input and make changes to the input dat accordingly
	data=pd.read_csv("function3_2d.csv")
	data=data.to_numpy()
	data=np.delete(data,0,axis=1)

	size=50 #Setting size of data to be trained on
	x1=data[:size,0]
	x2=data[:size,1]
	y=data[:size,2]
	
	totsize=math.floor(size*10/7) #Since 50 is the size of only the training data here we calculate the size of the total data(70% of total data)
	
	#Validation data
	x1_valid=data[size+1:size+math.floor(totsize/5)+1,0]
	x2_valid=data[size+1:size+math.floor(totsize/5)+1,1]
	y_valid=data[size+1:size+math.floor(totsize/5)+1,2]
	#Testing data
	x1_test=data[size+math.floor(totsize/5)+1:size+math.floor(totsize/5)+math.floor(totsize/10)+1,0]
	x2_test=data[size+math.floor(totsize/5)+1:size+math.floor(totsize/5)+math.floor(totsize/10)+1,1]
	y_test=data[size+math.floor(totsize/5)+1:size+math.floor(totsize/5)+math.floor(totsize/10)+1,2]

	powers=[2,3,6] #degree of the polynomial 

	finvals1=[]
	finvals2=[]
	finvals3=[]
	
	for power in powers: #Traversing the powers
		des_mat=np.empty((size,int(((power+2)*(power+1))/2)))

		for i in range(size):
			ct=0
			for p in range(power+1):
				tp=p
				while tp>=0:
					des_mat[i][ct]=(x1[i]**tp)*(x2[i]**(p-tp))
					tp=tp-1
					ct=ct+1

		#Obtaining the W Matrix
		A=np.dot(des_mat.T,des_mat)
		B=np.dot(des_mat.T,y)

		W=np.linalg.solve(A,B)
		
		
		fx=[]

		#The below loop is rms on training data
		for i in range(x1.shape[0]):
			fx.append(calci(x1[i],x2[i],W,power))
		
		fx=np.array(fx)
		rms=np.sqrt(np.mean((fx-y)**2,axis=0)) #Calcuating the rms value using model and target output
		finvals1.append(rms)
		
		
		
		fx=[]
		#The below loop is rms on validation data
		for i in range(x1_valid.shape[0]):
			fx.append(calci(x1_valid[i],x2_valid[i],W,power))
			
		
		fx=np.array(fx)
		fx=fx.reshape(fx.shape[0],)
		rms=np.sqrt(np.mean((fx-y_valid)**2,axis=0))
		finvals2.append(rms)
		
		fx=[]
		#The below loop is rms on testing data
		for i in range(x1_test.shape[0]):
			fx.append(calci(x1_test[i],x2_test[i],W,power))
		
		fx=np.array(fx)
		fx=fx.reshape(fx.shape[0],)
		rms=np.sqrt(np.mean((fx-y_test)**2,axis=0))
		finvals3.append(rms)
		
	#Plotting the power and rms values
	plt.plot(powers,finvals1,color='red',marker='o',label='Training Data')
	plt.plot(powers,finvals2,color='blue',marker='o',label='Validation Data')
	plt.plot(powers,finvals3,color='green',marker='o',label='Testing Data')
	plt.xlabel('M')
	plt.ylabel('$E_{rms}$')
	plt.legend()
	plt.title('$E_{rms}$ value for N=50')
	plt.savefig('Task2_N=50.png')
	plt.show()	
		
		
	


	


	