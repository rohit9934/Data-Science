import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import StandardScaler
#from sklearn import Decomposition
from scipy.linalg import eigh
from sklearn.manifold import TSNE
dataset= pd.read_csv("./mnist_train.csv")
d= dataset.drop('label',axis='columns')
l= dataset['label']
data= d.head(15000)
labels= l.head(15000)
standard_data= StandardScaler().fit_transform(data)
small_data= standard_data[0:1000]
small_labels= labels[0:1000]
#print(small_data.shape)
model= TSNE(n_components=2,random_state=0,perplexity= 50,n_iter= 5000)
tsne_data= model.fit_transform(small_data)
tsne_data= np.vstack((tsne_data.T,small_labels)).T
tsne_df= pd.DataFrame(data= tsne_data,columns=("1st","2nd","label"))
sn.FacetGrid(tsne_df,hue="label",size=6).map(plt.scatter,'1st','2nd','label').add_legend()
plt.show()
