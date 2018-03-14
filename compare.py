from __future__ import print_function
import pandas as pd
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import preprocessing,cross_validation,neighbors
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Data input
df=pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999,inplace=True)

print(df)
df.drop(labels=['id'], axis=1, inplace=True)

X=np.array(df.drop(['class'],1))
Y=np.array(df['class'])

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X,Y,test_size=0.2)

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', neighbors.KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = "accuracy"
seed=7
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	# print(cv_results)
	for accx in cv_results:
		print ("%.2f" % (accx), end = ', ') 
	print("")
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print (msg,end = "\n\n")

