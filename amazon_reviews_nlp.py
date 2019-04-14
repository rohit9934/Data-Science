import sqlite3
import string
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from nltk.stem.porter import PorterStemmer
con= sqlite3.connect('./database.sqlite')
data= pd.read_sql_query("""SELECT * FROM Reviews WHERE Score !=3  """, con)
def seperate(x):
    if x>3:
        return "positive"
    else:
        return "negative"
data['Score']=data['Score'].map(seperate)
#Data Cleaning, removing all the reviews by same person of same product with different types
sorted_data= data.sort_values('ProductId',axis=0,ascending=True,kind='quicksort')
final_data= sorted_data.drop_duplicates(subset={'UserId','ProfileName','Time','Text'},keep="first",inplace=False)
#HelpfulnessNumerator is present in the dataset and is the number of how much people found the review helpful
#print(final_data['Id'].size)
final_data= final_data[final_data.HelpfulnessNumerator<=final_data.HelpfulnessDenominator]
#------------------------Bag Of Words--------------------------
#Convert a text into a data structure like dictionary which instead of storing 364K(reviews)* 115K(vectors),storevectorsefficiently
#CountVectorizer store count of words in vectors efficiently
count_vect= CountVectorizer(ngram_range=(2,2))
BOW= count_vect.fit_transform(final_data['Text'].values)
print(BOW.shape)
# type of BOW vector is 'scipy.sparse.csr.csr_matrix'
