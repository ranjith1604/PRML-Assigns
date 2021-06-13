import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
	
	#The below lines take the csv file as input and make changes to the input dat accordingly
	data=pd.read_csv("function3_2d.csv")
	data=data.to_numpy()
	data=np.delete(data,0,axis=1)

	size=500 #Setting size of data to be trained on
	x1=data[:size,0]
	x2=data[:size,1]
	y=data[:size,2]

	power=6 #degree of the polynomial 

	des_mat=np.empty((size,int(((power+2)*(power+1))/2)))

	


	for i in range(size):
		ct=0
		for p in range(power+1):
			tp=p
			while tp>=0:
				des_mat[i][ct]=(x1[i]**tp)*(x2[i]**(p-tp)) #Constructing the design matrix
				tp=tp-1
				ct=ct+1

	#Obtaining the W Matrix
	A=np.dot(des_mat.T,des_mat)
	B=np.dot(des_mat.T,y)

	W=np.linalg.solve(A,B)

	
	#Setting X1 and X2 values for 3-D graph plotting
	X1=np.linspace(-18,18,200)
	X2=np.linspace(-18,18,200)
	
	X1,X2 = np.meshgrid(X1,X2)
	
	fx=[]

	
	#The below for loop generates the output for different X1's and X2's and stores them in fx
	for i in range(200):
		tp=[]
		for p in range(200):
			tp.append(calci(X1[i][p],X2[i][p],W,power))
		fx.append(tp)
			
	
	#Plotting is done below
	fig=plt.figure()
	fx=np.array(fx)
	ax=plt.axes(projection='3d')

	
	ax.plot_surface(X1,X2,fx)
	ax.scatter3D(x1,x2,y,color="red",label='Dataset Points')
	ax.set_title('Surface plot with M=6 N=500')
	ax.set_xlabel('x1')
	ax.set_ylabel('x2')
	ax.set_zlabel('y')
	ax.legend(bbox_to_anchor=(1.1, 0.97), bbox_transform=ax.transAxes)
	plt.savefig('M=6_N=500.png')
	plt.show()
	


	