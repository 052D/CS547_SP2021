import numpy as np
import pandas as pd  
import os
import time
from sklearn import linear_model

def load_data(folder):
	df = pd.read_csv(folder+'qsar_fish_toxicity.csv',header=None,sep=';')
	X = np.zeros((df.shape[0],4))
	X[:,0] = df.loc[:,0].to_numpy()
	X[:,1] = df.loc[:,1].to_numpy()
	X[:,2]= df.loc[:,2].to_numpy()
	X[:,3] = df.loc[:,5].to_numpy()
	y	= df.loc[:,6].to_numpy()
	return X,y


def question3_1():
	X,y = load_data('./')
	reg = linear_model.LinearRegression().fit(X,y)
	coef = reg.coef_
	intercept = reg.intercept_
	print("Result for Question 3.1")
	print('Y = %f*X0+%f*X1+%f*X3+%f*X6+%f'%(coef[0],coef[1],coef[2],coef[3],intercept))
	return

def question3_2():
	X,y = load_data('./')
	X = np.append(X,np.ones((X.shape[0],1)),axis=1)
	X_t = np.transpose(X)
	# b = (X^TX)^-1X^Ty
	b =np.matmul(np.matmul(np.linalg.inv(np.matmul(X_t,X)),X_t),y)
	print('Y = %f*X0+%f*X1+%f*X3+%f*X6+%f'%(b[0],b[1],b[2],b[3],b[4]))
	return

def question3_3(initial_point, learning_rate):
	X,y = load_data('./')
	X = np.append(X,np.ones((X.shape[0],1)),axis=1)
	X_t = np.transpose(X)
	b = initial_point
	for i in range(100000):
		b = b- learning_rate*(-2*np.matmul(X_t,y)+2*np.matmul(np.matmul(X_t,X),b))
	print('Y = %f*X0+%f*X1+%f*X3+%f*X6+%f'%(b[0],b[1],b[2],b[3],b[4]))
	return


if __name__ == '__main__':
	question3_1()
	question3_2()
	initial_point = np.zeros(5)
	learning_rate = 0.00001
	question3_3(initial_point,learning_rate)