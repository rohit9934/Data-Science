import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import StandardScaler
#from sklearn import Decomposition
from scipy.linalg import eigh
#This is data visualisation using PCA technique
#MNIST data_set is used
#Basically, i have manually coded the sklearn PCA method to understand Principal Component Analysis deeply
# 3 steps are used
#1. Column Standardization
#2. Finding the covarient matrix
#3. Finding Eigen values and Eigen vectors
#4. The last two Eigen values are need to convert the 784 dimensional(features) dataset into 2D
dataset= pd.read_csv('./mnist_train.csv')
#print(dataset.head(0))
d= dataset.drop('label',axis='columns')
l= dataset["label"]
plt.figure(figsize=(7,7))
idx=100
grid= d.iloc[idx].values.reshape(28,28)
#plt.imshow(grid,interpolation="nearest",cmap="gray")
#plt.show()
data= d.head(15000)
labels= l.head(15000)
print(labels.shape)
#Standardizing data
standard_data= StandardScaler().fit_transform(data)
#print(standard_data.shape)
#Finding covariance = X^T*X
cov_matrix= np.matmul(standard_data.T,standard_data)
#print(cov_matrix.shape)
#Finding Eigen values and eigen vectors
values, vectors = eigh(cov_matrix,eigvals=(782,783))
vectors= vectors.T
#print(values)
#print(vectors.shape)
#print(standard_data.T.shape)
#Finding optimal value by multiplying eigen vector to the matrix
optimal_vector= np.matmul(vectors,standard_data.T)
optimal_vector= np.vstack((optimal_vector,labels)).T
dataframe= pd.DataFrame(optimal_vector,columns=("1st","2nd","labels"))
#print(dataframe.head())
sn.FacetGrid(dataframe,hue="labels",size=6).map(plt.scatter,"1st","2nd").add_legend()
plt.show()
