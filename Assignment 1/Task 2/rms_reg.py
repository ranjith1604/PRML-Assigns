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
	
	data=pd.read_csv("function3_2d.csv")
	data=data.to_numpy()
	data=np.delete(data,0,axis=1)

	size=500
	x1=data[:size,0]
	x2=data[:size,1]
	y=data[:size,2]
	
	totsize=math.floor(size*10/7)

	x1_valid=data[size+1:size+math.floor(totsize/5)+1,0]
	x2_valid=data[size+1:size+math.floor(totsize/5)+1,1]
	y_valid=data[size+1:size+math.floor(totsize/5)+1,2]

	x1_test=data[size+math.floor(totsize/5)+1:size+math.floor(totsize/5)+math.floor(totsize/10)+1,0]
	x2_test=data[size+math.floor(totsize/5)+1:size+math.floor(totsize/5)+math.floor(totsize/10)+1,1]
	y_test=data[size+math.floor(totsize/5)+1:size+math.floor(totsize/5)+math.floor(totsize/10)+1,2]

	
	power=3
	finvals1=[]
	finvals2=[]
	finvals3=[]
	hyper=np.logspace(5,8,10)
	
	#Here instead of varying power we vary the hyperparameter
	for hyp in hyper:
		
		des_mat=np.empty((size,int(((power+2)*(power+1))/2)))
		for i in range(size):
			ct=0
			for p in range(power+1):
				tp=p
				while tp>=0:
					des_mat[i][ct]=(x1[i]**tp)*(x2[i]**(p-tp))
					tp=tp-1
					ct=ct+1

		

		A=np.dot(des_mat.T,des_mat)
		
		tp=A.shape[0]
	
		for i in range(tp):
			for p in range(tp):
				if i==p:
					A[i][p]+=hyp
		
		B=np.dot(des_mat.T,y)

		W=np.linalg.solve(A,B)
		
		fx=[]

		
		for i in range(x1.shape[0]):
			fx.append(calci(x1[i],x2[i],W,power))
		
		fx=np.array(fx)
		rms=np.sqrt(np.mean((fx-y)**2,axis=0))
		finvals1.append(rms)
		
		
		
		fx=[]
		
		for i in range(x1_valid.shape[0]):
			fx.append(calci(x1_valid[i],x2_valid[i],W,power))
			
		
		fx=np.array(fx)
		fx=fx.reshape(fx.shape[0],)
		rms=np.sqrt(np.mean((fx-y_valid)**2,axis=0))
		finvals2.append(rms)
		
		fx=[]
		
		for i in range(x1_test.shape[0]):
			fx.append(calci(x1_test[i],x2_test[i],W,power))
		
		fx=np.array(fx)
		fx=fx.reshape(fx.shape[0],)
		rms=np.sqrt(np.mean((fx-y_test)**2,axis=0))
		finvals3.append(rms)
		
		
	#Plotting the power and rms values	
	hyper=np.log10(hyper)
	plt.plot(hyper,finvals1,color='red',marker='o',label='Training Data')
	plt.plot(hyper,finvals2,color='blue',marker='o',label='Validation Data')
	plt.plot(hyper,finvals3,color='green',marker='o',label='Testing Data')
	plt.xlabel('log$\lambda$')
	plt.ylabel('$E_{rms}$')
	plt.legend()
	plt.title('$E_{rms}$ value for N=500 and M=3 with varying $\lambda$')
	plt.savefig('Task2_reg_N=500_M=3.png')
	plt.show()
		
		
	


	


	