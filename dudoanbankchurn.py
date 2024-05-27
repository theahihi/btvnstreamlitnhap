import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle as pickle
import streamlit as st
import numpy as np
data= pd.read_csv('train.csv')
feature=['id','Age','CreditScore','Balance','EstimatedSalary']
target=['Exited']
X=data[feature]
y=data[target]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 1/4, random_state=42,)
clf = LogisticRegression()
clf.fit(X_train, y_train)
pickle.dump(clf, open("bank_churn.pkl", 'wb'))
