import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def calci(X1,X2,W,power):  #function to find the output for an input
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
	
	#The below code is same as rms
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

	
	power=6
	finvals1=[]
	finvals2=[]
	finvals3=[]
	hyp=0  #Vary this according to best model
		
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

		
	for i in range(x1.shape[0]): #The x1 could be varied to x1_test for testing data 
		fx.append(calci(x1[i],x2[i],W,power))
		
	fx=np.array(fx)
	
	rms=np.sqrt(np.mean((fx-y)**2,axis=0))
	finvals1.append(rms)
		
	#Plotting the scatter values and the straight line
	plt.scatter(y,fx,marker='o',color='red',label='Estimated Outputs')
	plt.plot(y,y,label='Actual Outputs')
	
	plt.xlabel('t$_n$')
	plt.ylabel('Model Output')
	plt.legend()
	plt.title('Training Data with M=6 N=500 and $\lambda$=0')
	plt.text(170,12,'$E_{rms}$=%f'%(rms),fontsize=11)
	plt.savefig('Scatter_3_train.png')
	plt.show()
		
		
	


	


	