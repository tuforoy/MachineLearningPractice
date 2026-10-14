#%% import package

import pandas as pd 
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.preprocessing import StandardScaler

#%% import and visualise data

data = pd.read_csv('C:/data/Glass.csv')
data1=pd.read_csv('C:/data/Glass.csv')
data = data.iloc[:,1:]
data1 = data.iloc[:,1:]
scaler = StandardScaler()
data1.iloc[:,:-1] = scaler.fit_transform(data1.iloc[:,:-1])
pd.plotting.scatter_matrix(data1.iloc[:,:-1])


#%% knn(Data)

X,y = data.iloc[:,:-1],data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=0)
print('=====KNN (Data)=====')
for k in range(1,10):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    print('Correct rate with ' + str(k) + '-NN: ' + str(metrics.accuracy_score(y_test, y_pred)))
    
#%% knn(Data 1)

X,y = data1.iloc[:,:-1],data1.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=0)
print('=====KNN (Data1)=====')
for k in range(1,10):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    print('Correct rate with ' + str(k) + '-NN: ' + str(metrics.accuracy_score(y_test, y_pred)))
    
    
#%% hyper parameter tuning of knn (data)

X,y = data.iloc[:,:-1],data.iloc[:,-1]
max_k = 10; no_runs = 10; possible_k = range(1,max_k)
results = []
for ii in range(no_runs):
    grid_search_results = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=ii)
    X_model, X_validate, y_model, y_validate = train_test_split(X_train, y_train, test_size=0.10,random_state=ii)
    for k in possible_k:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_model,y_model)
        y_pred = knn.predict(X_validate)
        grid_search_results.append(metrics.accuracy_score(y_validate, y_pred))
    optimal_k = possible_k[np.argmax(grid_search_results)]
    knn = KNeighborsClassifier(n_neighbors=optimal_k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    results.append(metrics.accuracy_score(y_test, y_pred))
print('=====KNN hyper parameter tuning (Data)=====')
print('Average classificaiton rate: ' + str(np.average(results)) )
print("Optimal value of k: %s" %optimal_k)

#%% hyper parameter tuning of knn (data1)

X,y = data1.iloc[:,:-1],data1.iloc[:,-1]
max_k = 10; no_runs = 10; possible_k = range(1,max_k)
results = []
for ii in range(no_runs):
    grid_search_results = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=ii)
    X_model, X_validate, y_model, y_validate = train_test_split(X_train, y_train, test_size=0.10,random_state=ii)
    for k in possible_k:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_model,y_model)
        y_pred = knn.predict(X_validate)
        grid_search_results.append(metrics.accuracy_score(y_validate, y_pred))
    optimal_k = possible_k[np.argmax(grid_search_results)]
    knn = KNeighborsClassifier(n_neighbors=optimal_k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    results.append(metrics.accuracy_score(y_test, y_pred))
print('=====KNN hyper parameter tuning (Data1)=====')
print('Average classificaiton rate: ' + str(np.average(results)) )
print("Optimal value of k: %s" %optimal_k)

#%% Grid Search CV (SVM) (data)

X,y = data.iloc[:,1:-1],data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=ii)

parameters = {'kernel':('linear', 'rbf','poly','sigmoid'), 'C':[0.001,0.01,0.1,1, 10,100]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(X_train, y_train)
print('=====Grid Search CV (SVM) (Data)=====')
print("The best parameters are %s with a score of %.2f"  %(clf.best_params_,clf.best_score_))
print('Test accuracy: %.3f' % clf.score(X_test, y_test))
clfSVC = clf.best_estimator_
print('Test accuracy: %.3f' % clfSVC.score(X_test, y_test))

#%% Grid Search CV (SVM) (data1)

X,y = data1.iloc[:,1:-1],data1.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=ii)

parameters = {'kernel':('linear', 'rbf','poly','sigmoid'), 'C':[0.001,0.01,0.1,1, 10,100]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(X_train, y_train)
print('=====Grid Search CV (SVM) (Data1)=====')
print("The best parameters are %s with a score of %.2f"  %(clf.best_params_,clf.best_score_))
print('Test accuracy: %.3f' % clf.score(X_test, y_test))
clfSVC = clf.best_estimator_
print('Test accuracy: %.3f' % clfSVC.score(X_test, y_test))

#%% Random Search CV (SVM) (data)

X,y = data.iloc[:,1:-1],data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=ii)

parameters = {'kernel':('linear', 'rbf','poly','sigmoid'), 'C': [0.001,0.01,0.1,1, 10,100]}
svc = svm.SVC()
clf = RandomizedSearchCV(svc, parameters)
clf.fit(X_train, y_train)
print('===== Random Search CV (SVM) (Data)=====')
print("The best parameters are %s with a score of %.2f"  %(clf.best_params_,clf.best_score_))
print('Test accuracy: %.3f' % clf.score(X_test, y_test))
clfSVC = clf.best_estimator_
print('Test accuracy: %.3f' % clfSVC.score(X_test, y_test))



#%%  Random Search CV (SVM) (data1)

X,y = data1.iloc[:,1:-1],data1.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=ii)

parameters = {'kernel':('linear', 'rbf','poly','sigmoid'), 'C': [0.001,0.01,0.1,1, 10,100]}
svc = svm.SVC()
clf = RandomizedSearchCV(svc, parameters)
clf.fit(X_train, y_train)
print('===== Random Search CV (SVM) (Data1)=====')
print("The best parameters are %s with a score of %.2f"  %(clf.best_params_,clf.best_score_))
print('Test accuracy: %.3f' % clf.score(X_test, y_test))
clfSVC = clf.best_estimator_
print('Test accuracy: %.3f' % clfSVC.score(X_test, y_test))
