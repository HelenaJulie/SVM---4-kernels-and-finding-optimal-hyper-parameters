# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 07:20:14 2018

@author: Helena J Arpudaraj
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 16:03:58 2018

@author: Helena J Arpudaraj
"""
import numpy as np
import pandas as pd
from sklearn import svm, grid_search
from sklearn.model_selection import ShuffleSplit, train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
import time

row=['Id','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Type']
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data',header=None,names=row)

df=df.drop(columns=['Id'])
print(df)

train= np.zeros((214,10))
train=np.array(df)[:,:]

trainX=np.zeros((214,9))
trainY=np.zeros((214,1))
trainX=np.array(train)[:,:-1]
trainY=np.array(train)[:,9]

#shuffle data set-validation set 20%
X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size = 0.20,random_state=42)

#finding optimal parameters for every kernel

accuracies_linear=[]
C_linear=[]
gamma_linear=[]

#k-fold validation to find optimal hyperparameters C and Gamma-linear kernel
for i in range (-2,3):
    for j in range (-14,15):
        svc = svm.SVC(kernel='linear', C=(2**i), gamma=(2**j))
        #5fold validation
        scores = cross_val_score(svc, X_train, y_train, cv=5)
        accuracies_linear.append(scores.mean())
        C_linear.append(i)
        gamma_linear.append(j)
        
index=accuracies_linear.index(max(accuracies_linear))
OptimalC_linear=C_linear[index]
Optimalgamma_linear=gamma_linear[index]
print(accuracies_linear)
print("Optimal hyperparamteres for linear kernel: C=2^",OptimalC_linear," gamma=2^",Optimalgamma_linear)

accuracies_rbf=[]
C_rbf=[]
gamma_rbf=[]

#k-fold validation to find optimal hyperparameters C and Gamma-rbf kernel
for i in range (-2,3):
    for j in range (-14,15):
        svc = svm.SVC(kernel='rbf', C=(2**i), gamma=(2**j))
        #5fold validation
        scores = cross_val_score(svc, X_train, y_train, cv=5)
        accuracies_rbf.append(scores.mean())
        C_rbf.append(i)
        gamma_rbf.append(j)

index=accuracies_rbf.index(max(accuracies_rbf))
OptimalC_rbf=C_rbf[index]
Optimalgamma_rbf=gamma_rbf[index]
print(accuracies_rbf)
print("Optimal hyperparamteres for rbf kernel: C=2^",OptimalC_rbf," gamma=2^",Optimalgamma_rbf)

accuracies_poly=[]
C_poly=[]
gamma_poly=[]
degree_poly=[]
#k-fold validation to find optimal hyperparameters C and Gamma-poly kernel
for i in range (-2,3):
    for j in range (-14,15):
        for k in range (1,4):
            svc = svm.SVC(kernel='poly', C=(2**i), gamma=(2**j),degree=k)
            #5fold validation
            scores = cross_val_score(svc, X_train, y_train, cv=5)
            accuracies_poly.append(scores.mean())
            C_poly.append(i)
            gamma_poly.append(j)
            degree_poly.append(k)
index=accuracies_poly.index(max(accuracies_poly))
OptimalC_poly=C_poly[index]
Optimalgamma_poly=gamma_poly[index]
Optimaldegree_poly=degree_poly[index]
print(accuracies_poly)
print("Optimal hyperparamteres for poly kernel: C=2^",OptimalC_poly," gamma=2^",Optimalgamma_poly," degree=",Optimaldegree_poly)


accuracies_sigmoid=[]
C_sigmoid=[]
gamma_sigmoid=[]

#k-fold validation to find optimal hyperparameters C and Gamma-sigmoid kernel
for i in range (-2,3):
    for j in range (-14,15):
        svc = svm.SVC(kernel='sigmoid', C=(2**i), gamma=(2**j))
        #5fold validation
        scores = cross_val_score(svc, X_train, y_train, cv=5)
        accuracies_sigmoid.append(scores.mean())
        C_sigmoid.append(i)
        gamma_sigmoid.append(j)

index=accuracies_sigmoid.index(max(accuracies_sigmoid))
OptimalC_sigmoid=C_sigmoid[index]
Optimalgamma_sigmoid=gamma_sigmoid[index]
print(accuracies_sigmoid)
print("Optimal hyperparamteres for sigmoid kernel: C=2^",OptimalC_sigmoid," gamma=2^",Optimalgamma_sigmoid)


#applying above optimal hyperparameters to the dataset
start_time = time.time()

svc = svm.SVC(kernel='linear', C=(2**OptimalC_linear), gamma=(2**Optimalgamma_linear)).fit(X_train,y_train)
predicted = svc.predict(X_test)

# get the accuracy
print("Accuracy for linear kernel on validation set = ",(accuracy_score(y_test, predicted))*100)
print("training time=",time.time()-start_time)


start_time = time.time()

svc = svm.SVC(kernel='rbf', C=(2**OptimalC_rbf), gamma=(2**Optimalgamma_rbf)).fit(X_train,y_train)
predicted = svc.predict(X_test)

# get the accuracy
print("Accuracy for rbf kernel on validation set = ",(accuracy_score(y_test, predicted))*100)
print("training time=",time.time()-start_time)

start_time = time.time()

svc = svm.SVC(kernel='poly', C=(2**OptimalC_poly), gamma=(2**Optimalgamma_poly),degree=Optimaldegree_poly).fit(X_train,y_train)
predicted = svc.predict(X_test)

# get the accuracy
print("Accuracy for poly kernel on validation set = ",(accuracy_score(y_test, predicted))*100)
print("training time=",time.time()-start_time)


start_time = time.time()

svc = svm.SVC(kernel='sigmoid', C=(2**OptimalC_sigmoid), gamma=(2**Optimalgamma_sigmoid)).fit(X_train,y_train)
predicted = svc.predict(X_test)

# get the accuracy
print("Accuracy for sigmoid kernel on validation set = ",(accuracy_score(y_test, predicted))*100)
print("training time=",time.time()-start_time)

#OneVsRestClassifier
accuracies_linear=[]
C_linear=[]
gamma_linear=[]

#k-fold validation to find optimal hyperparameters C and Gamma-linear kernel
for i in range (-2,3):
    for j in range (-14,15):
        svc = OneVsRestClassifier(svm.SVC(kernel='linear', C=(2**i), gamma=(2**j)))
        #5fold validation
        scores = cross_val_score(svc, X_train, y_train, cv=5)
        accuracies_linear.append(scores.mean())
        C_linear.append(i)
        gamma_linear.append(j)
        
index=accuracies_linear.index(max(accuracies_linear))
OptimalC_linear=C_linear[index]
Optimalgamma_linear=gamma_linear[index]
print(accuracies_linear)
print("Optimal hyperparamteres for linear kernel: C=2^",OptimalC_linear," gamma=2^",Optimalgamma_linear)

accuracies_rbf=[]
C_rbf=[]
gamma_rbf=[]

#k-fold validation to find optimal hyperparameters C and Gamma-rbf kernel
for i in range (-2,3):
    for j in range (-14,15):
        svc = OneVsRestClassifier(svm.SVC(kernel='rbf', C=(2**i), gamma=(2**j)))
        #5fold validation
        scores = cross_val_score(svc, X_train, y_train, cv=5)
        accuracies_rbf.append(scores.mean())
        C_rbf.append(i)
        gamma_rbf.append(j)

index=accuracies_rbf.index(max(accuracies_rbf))
OptimalC_rbf=C_rbf[index]
Optimalgamma_rbf=gamma_rbf[index]
print(accuracies_rbf)
print("Optimal hyperparamteres for rbf kernel: C=2^",OptimalC_rbf," gamma=2^",Optimalgamma_rbf)

accuracies_poly=[]
C_poly=[]
gamma_poly=[]
degree_poly=[]
#k-fold validation to find optimal hyperparameters C and Gamma-poly kernel
for i in range (-2,3):
    for j in range (-14,15):
        for k in range (1,4):
            svc = OneVsRestClassifier(svm.SVC(kernel='poly', C=(2**i), gamma=(2**j),degree=k))
            #5fold validation
            scores = cross_val_score(svc, X_train, y_train, cv=5)
            accuracies_poly.append(scores.mean())
            C_poly.append(i)
            gamma_poly.append(j)
            degree_poly.append(k)
index=accuracies_poly.index(max(accuracies_poly))
OptimalC_poly=C_poly[index]
Optimalgamma_poly=gamma_poly[index]
Optimaldegree_poly=degree_poly[index]
print(accuracies_poly)
print("Optimal hyperparamteres for poly kernel: C=2^",OptimalC_poly," gamma=2^",Optimalgamma_poly," degree=",Optimaldegree_poly)


accuracies_sigmoid=[]
C_sigmoid=[]
gamma_sigmoid=[]

#k-fold validation to find optimal hyperparameters C and Gamma-sigmoid kernel
for i in range (-2,3):
    for j in range (-14,15):
        svc = OneVsRestClassifier(svm.SVC(kernel='sigmoid', C=(2**i), gamma=(2**j)))
        #5fold validation
        scores = cross_val_score(svc, X_train, y_train, cv=5)
        accuracies_sigmoid.append(scores.mean())
        C_sigmoid.append(i)
        gamma_sigmoid.append(j)

index=accuracies_sigmoid.index(max(accuracies_sigmoid))
OptimalC_sigmoid=C_sigmoid[index]
Optimalgamma_sigmoid=gamma_sigmoid[index]
print(accuracies_sigmoid)
print("Optimal hyperparamteres for sigmoid kernel: C=2^",OptimalC_sigmoid," gamma=2^",Optimalgamma_sigmoid)


#applying above optimal hyperparameters to the dataset
start_time = time.time()

svc = OneVsRestClassifier(svm.SVC(kernel='linear', C=(2**OptimalC_linear), gamma=(2**Optimalgamma_linear))).fit(X_train,y_train)
predicted = svc.predict(X_test)

# get the accuracy
print("Accuracy for linear kernel on validation set = ",(accuracy_score(y_test, predicted))*100)
print("training time=",time.time()-start_time)


start_time = time.time()

svc = OneVsRestClassifier(svm.SVC(kernel='rbf', C=(2**OptimalC_rbf), gamma=(2**Optimalgamma_rbf))).fit(X_train,y_train)
predicted = svc.predict(X_test)

# get the accuracy
print("Accuracy for rbf kernel on validation set = ",(accuracy_score(y_test, predicted))*100)
print("training time=",time.time()-start_time)

start_time = time.time()

svc = OneVsRestClassifier(svm.SVC(kernel='poly', C=(2**OptimalC_poly), gamma=(2**Optimalgamma_poly),degree=Optimaldegree_poly)).fit(X_train,y_train)
predicted = svc.predict(X_test)

# get the accuracy
print("Accuracy for poly kernel on validation set = ",(accuracy_score(y_test, predicted))*100)
print("training time=",time.time()-start_time)


start_time = time.time()

svc = OneVsRestClassifier(svm.SVC(kernel='sigmoid', C=(2**OptimalC_sigmoid), gamma=(2**Optimalgamma_sigmoid))).fit(X_train,y_train)
predicted = svc.predict(X_test)

# get the accuracy
print("Accuracy for sigmoid kernel on validation set = ",(accuracy_score(y_test, predicted))*100)
print("training time=",time.time()-start_time)



#class weight

accuracies_linear=[]
C_linear=[]
gamma_linear=[]

#k-fold validation to find optimal hyperparameters C and Gamma-linear kernel
for i in range (-2,3):
    for j in range (-14,15):
        svc = svm.SVC(kernel='linear', C=(2**i), gamma=(2**j),class_weight='balanced', decision_function_shape='ovo')
        #5fold validation
        scores = cross_val_score(svc, X_train, y_train, cv=5)
        accuracies_linear.append(scores.mean())
        C_linear.append(i)
        gamma_linear.append(j)
        
index=accuracies_linear.index(max(accuracies_linear))
OptimalC_linear=C_linear[index]
Optimalgamma_linear=gamma_linear[index]
print(accuracies_linear)
print("Optimal hyperparamteres for linear kernel: C=2^",OptimalC_linear," gamma=2^",Optimalgamma_linear)

accuracies_rbf=[]
C_rbf=[]
gamma_rbf=[]

#k-fold validation to find optimal hyperparameters C and Gamma-rbf kernel
for i in range (-2,3):
    for j in range (-14,15):
        svc = svm.SVC(kernel='rbf', C=(2**i), gamma=(2**j),class_weight='balanced',decision_function_shape='ovo')
        #5fold validation
        scores = cross_val_score(svc, X_train, y_train, cv=5)
        accuracies_rbf.append(scores.mean())
        C_rbf.append(i)
        gamma_rbf.append(j)

index=accuracies_rbf.index(max(accuracies_rbf))
OptimalC_rbf=C_rbf[index]
Optimalgamma_rbf=gamma_rbf[index]
print(accuracies_rbf)
print("Optimal hyperparamteres for rbf kernel: C=2^",OptimalC_rbf," gamma=2^",Optimalgamma_rbf)

accuracies_poly=[]
C_poly=[]
gamma_poly=[]
degree_poly=[]
#k-fold validation to find optimal hyperparameters C and Gamma-poly kernel
for i in range (-2,3):
    for j in range (-14,15):
        for k in range (1,4):
            svc = svm.SVC(kernel='poly', C=(2**i), gamma=(2**j),degree=k,class_weight='balanced',decision_function_shape='ovo')
            #5fold validation
            scores = cross_val_score(svc, X_train, y_train, cv=5)
            accuracies_poly.append(scores.mean())
            C_poly.append(i)
            gamma_poly.append(j)
            degree_poly.append(k)
index=accuracies_poly.index(max(accuracies_poly))
OptimalC_poly=C_poly[index]
Optimalgamma_poly=gamma_poly[index]
Optimaldegree_poly=degree_poly[index]
print(accuracies_poly)
print("Optimal hyperparamteres for poly kernel: C=2^",OptimalC_poly," gamma=2^",Optimalgamma_poly," degree=",Optimaldegree_poly)


accuracies_sigmoid=[]
C_sigmoid=[]
gamma_sigmoid=[]

#k-fold validation to find optimal hyperparameters C and Gamma-sigmoid kernel
for i in range (-2,3):
    for j in range (-14,15):
        svc = svm.SVC(kernel='sigmoid', C=(2**i), gamma=(2**j),coef0=1.0,class_weight='balanced',decision_function_shape='ovo')
        #5fold validation
        scores = cross_val_score(svc, X_train, y_train, cv=5)
        accuracies_sigmoid.append(scores.mean())
        C_sigmoid.append(i)
        gamma_sigmoid.append(j)

index=accuracies_sigmoid.index(max(accuracies_sigmoid))
OptimalC_sigmoid=C_sigmoid[index]
Optimalgamma_sigmoid=gamma_sigmoid[index]
print(accuracies_sigmoid)
print("Optimal hyperparamteres for sigmoid kernel: C=2^",OptimalC_sigmoid," gamma=2^",Optimalgamma_sigmoid)


#applying above optimal hyperparameters to the dataset
start_time = time.time()

svc = svm.SVC(kernel='linear', C=(2**OptimalC_linear), gamma=(2**Optimalgamma_linear),class_weight='balanced',decision_function_shape='ovo').fit(X_train,y_train)
predicted = svc.predict(X_test)

# get the accuracy
print("Accuracy for linear kernel on validation set = ",(accuracy_score(y_test, predicted))*100)
print("training time=",time.time()-start_time)


start_time = time.time()

svc = svm.SVC(kernel='rbf', C=(2**OptimalC_rbf), gamma=(2**Optimalgamma_rbf),class_weight='balanced',decision_function_shape='ovo').fit(X_train,y_train)
predicted = svc.predict(X_test)

# get the accuracy
print("Accuracy for rbf kernel on validation set = ",(accuracy_score(y_test, predicted))*100)
print("training time=",time.time()-start_time)

start_time = time.time()

svc = svm.SVC(kernel='poly', C=(2**OptimalC_poly), gamma=(2**Optimalgamma_poly),degree=Optimaldegree_poly,class_weight='balanced',decision_function_shape='ovo').fit(X_train,y_train)
predicted = svc.predict(X_test)

# get the accuracy
print("Accuracy for poly kernel on validation set = ",(accuracy_score(y_test, predicted))*100)
print("training time=",time.time()-start_time)


start_time = time.time()

svc = svm.SVC(kernel='sigmoid', C=(2**OptimalC_sigmoid), gamma=(2**Optimalgamma_sigmoid),class_weight='balanced',decision_function_shape='ovo').fit(X_train,y_train)
predicted = svc.predict(X_test)

# get the accuracy
print("Accuracy for sigmoid kernel on validation set = ",(accuracy_score(y_test, predicted))*100)
print("training time=",time.time()-start_time)

