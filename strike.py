import numpy as np
from sklearn import preprocessing,cross_validation,neighbors
import pandas as pd  

# Through compare.py, we know that the best model to be used is KNN


df=pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999,inplace=True)

print(df)
df.drop(labels=['id'], axis=1, inplace=True)

X=np.array(df.drop(['class'],1))
Y=np.array(df['class'])

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X,Y,test_size=0.2)
# print(X_test)
# print(Y_test)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,Y_train)

accuracy=clf.score(X_test,Y_test)
print("accuracy is ")
print(accuracy)

example_measures=np.array([[4,2,1,1,1,2,3,2,1]])
prediction = clf.predict(example_measures)
print(prediction)

