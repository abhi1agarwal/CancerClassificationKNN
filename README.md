# CancerClassificationKNN
Using model comparison and ultimately using KNN for Breast Cancer classification into two classes namely benign and malignant.

Compared 5 models with accuracy as percentage score, and 10 fold to find the best fitting model among all.
The models tried were - 
LogisticRegression, LinearDiscriminantAnalysis, KNeighborsClassifier, DecisionTreeClassifier, GaussianNB, SVC 

Accuracy using 10 fold : (stating indivial accuracy of the 10 folds and then mean and standard deviation)
0.98, 1.00, 0.98, 0.86, 0.96, 0.98, 0.95, 0.96, 0.96, 0.96, 
LR: 0.960649 (0.037281)

0.98, 0.98, 0.96, 0.84, 1.00, 0.95, 0.93, 0.93, 0.96, 0.95, 
LDA: 0.948117 (0.042601)

0.95, 0.98, 1.00, 0.89, 0.96, 0.98, 0.98, 0.96, 1.00, 0.96, 
KNN: 0.967792 (0.029675)

0.91, 0.96, 0.96, 0.91, 0.95, 0.93, 0.91, 0.95, 0.95, 0.95, 
CART: 0.937403 (0.019923)

0.95, 0.98, 0.98, 0.84, 0.96, 0.98, 0.96, 0.95, 0.98, 0.95, 
NB: 0.953474 (0.040895)

0.93, 0.98, 1.00, 0.89, 0.98, 0.98, 0.98, 0.91, 0.96, 0.93, 
SVM: 0.955227 (0.035140)

Best model found is KNN.
Implemented KNN using scikit-learn in script file named "strike.py"

Data set Taken from : https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
