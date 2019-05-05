# -*- coding: utf-8 -*-
"""
Created on Fri May  3 21:49:41 2019

@author: LENOVO
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# plotly
# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
data = pd.read_csv('D:\diabetes.csv')
#Split Data as M&B
p = data[data.Outcome == 1]
n = data[data.Outcome == 0]
#Visualization, Scatter Plot
plt.scatter(p.Pregnancies,p.Glucose,color = "brown",label="Diabet Positive",alpha=0.4)
plt.scatter(n.Pregnancies,n.Glucose,color = "Orange",label="Diabet Negative",alpha=0.2)
plt.xlabel("Pregnancies")
plt.ylabel("Glucose")
plt.legend()
plt.show()
#We appear that it is clear segregation
plt.scatter(p.Age,p.Pregnancies,color = "lime",label="Diabet Positive",alpha=0.4)
plt.scatter(n.Age,n.Pregnancies,color = "black",label="Diabet Negative",alpha=0.2)
plt.xlabel("Age")
plt.ylabel("Pregnancies")
plt.legend()
plt.show()

#We appear that it is clear segregation.


plt.scatter(p.Glucose,p.Insulin,color = "lime",label="Diabet Positive",alpha=0.4)
plt.scatter(n.Glucose,n.Insulin,color = "black",label="Diabet Negative",alpha=0.1)
plt.xlabel("Glucose")
plt.ylabel("Insulin")
plt.legend()
plt.show()

#We appear that it is clear segregation.
y= data.Outcome.values
x1= data.drop(["Outcome"],axis= 1) #we remowe diagnosis for predict

x = (x1-np.min(x1))/(np.max(x1)-np.min(x1))

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest =  train_test_split(x,y,test_size=0.3,random_state=42)

#Logistic Regression--------------------------------------------------------------------------
###################
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(xtrain,ytrain)
print("Test Accuracy {}".format(LR.score(xtest,ytest))) 

yprediciton1= LR.predict(xtest)
ytrue = ytest

from sklearn.metrics import confusion_matrix
CM = confusion_matrix(ytrue,yprediciton1)

#CM visualization

import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(CM,annot = True, linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Prediction(Ypred)")
plt.ylabel("Ytrue")
plt.show()

LRscore = LR.score(xtest,ytest)



#K-NN-------------------------------------------------------------------------------------------
#######
#Create-KNN-model
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors = 40) #n_neighbors = K value
KNN.fit(xtrain,ytrain) #learning model
prediction = KNN.predict(xtest)
#Prediction
print("{}-NN Score: {}".format(40,KNN.score(xtest,ytest)))

KNNscore = KNN.score(xtest,ytest)

#Find Optimum K value
scores = []
for each in range(1,100):
    KNNfind = KNeighborsClassifier(n_neighbors = each)
    KNNfind.fit(xtrain,ytrain)
    scores.append(KNNfind.score(xtest,ytest))
    
plt.plot(range(1,100),scores,color="black")
plt.title("Optimum K Value")
plt.xlabel("K Values")
plt.ylabel("Score(Accuracy)")
plt.show()

yprediciton2= KNN.predict(xtest)
ytrue = ytest

from sklearn.metrics import confusion_matrix
CM = confusion_matrix(ytrue,yprediciton2)

#CM visualization

import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(CM,annot = True, linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Prediction(Ypred)")
plt.ylabel("Ytrue")
plt.show()

#SVM with Sklearn---------------------------------------------------------------------
################
from sklearn.svm import SVC
SVM = SVC(random_state=42)
SVM.fit(xtrain,ytrain)  #learning 
#SVM Test 
print ("SVM Accuracy:", SVM.score(xtest,ytest))
SVMscore = SVM.score(xtest,ytest)

#Confusion Matrix
yprediciton3= SVM.predict(xtest)
ytrue = ytest
from sklearn.metrics import confusion_matrix
CM = confusion_matrix(ytrue,yprediciton3)
#CM visualization
import seaborn as sns
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(CM,annot = True, linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Prediction(Ypred)")
plt.ylabel("Ytrue")
plt.show()


#Naive Bayes--------------------------------------------------------------------------------
###########
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(xtrain,ytrain) #learning
#prediction
print("Accuracy of NB Score: ", NB.score(xtest,ytest))
NBscore= NB.score(xtest,ytest)
#Confusion Matrix
yprediciton4= NB.predict(xtest)
ytrue = ytest
from sklearn.metrics import confusion_matrix
CM = confusion_matrix(ytrue,yprediciton4)
#CM visualization
import seaborn as sns
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(CM,annot = True, linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Prediction(Ypred)")
plt.ylabel("Ytrue")
plt.show()


#Decision Tree Algorithm----------------------------------------------------------------------
#######################
from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier()
DTC.fit(xtrain,ytrain) #learning
#prediciton
print("Decision Tree Score: ",DTC.score(xtest,ytest))
DTCscore = DTC.score(xtest,ytest)

#Confusion Matrix
yprediciton5= DTC.predict(xtest)
ytrue = ytest
from sklearn.metrics import confusion_matrix
CM = confusion_matrix(ytrue,yprediciton5)
#CM visualization
import seaborn as sns
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(CM,annot = True, linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Prediction(Ypred)")
plt.ylabel("Ytrue")
plt.show()
#Decision Tree Algorithm

from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier()
DTC.fit(xtrain,ytrain) #learning
#prediciton
print("Decision Tree Score: ",DTC.score(xtest,ytest))




#Random Forest---------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
RFC= RandomForestClassifier(n_estimators = 24, random_state=42) #n_estimator = DT
RFC.fit(xtrain,ytrain) # learning
print("Random Forest Score: ",RFC.score(xtest,ytest))
RFCscore=RFC.score(xtest,ytest)
#Find Optimum K value
scores = []
for each in range(1,30):
    RFfind = RandomForestClassifier(n_estimators = each)
    RFfind.fit(xtrain,ytrain)
    scores.append(RFfind.score(xtest,ytest))
    
plt.plot(range(1,30),scores,color="black")
plt.title("Optimum N Estimator Value")
plt.xlabel("N Estimators")
plt.ylabel("Score(Accuracy)")
plt.show()#Find Optimum K value
scores = []
for each in range(1,30):
    RFfind = RandomForestClassifier(n_estimators = each)
    RFfind.fit(xtrain,ytrain)
    scores.append(RFfind.score(xtest,ytest))
    
plt.plot(range(1,30),scores,color="black")
plt.title("Optimum N Estimator Value")
plt.xlabel("N Estimators")
plt.ylabel("Score(Accuracy)")
plt.show()

#Confusion Matrix

yprediciton6= RFC.predict(xtest)
ytrue = ytest

from sklearn.metrics import confusion_matrix
CM = confusion_matrix(ytrue,yprediciton6)

#CM visualization

import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(CM,annot = True, linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Prediction(Ypred)")
plt.ylabel("Ytrue")
plt.show()
scores=[LRscore,KNNscore,SVMscore,NBscore,DTCscore,RFCscore]
AlgorthmsName=["Logistic Regression","K-NN","SVM","Naive Bayes","Decision Tree", "Random Forest"]
#create traces
trace1 = go.Bar(
    x = AlgorthmsName,
    y= scores,
    name='Algortms Name',
    marker =dict(color='rgba(0,255,0,0.5)',
               line =dict(color='rgb(0,0,0)',width=2)),
                text=AlgorthmsName
)
data = [trace1]

layout = go.Layout(barmode = "group",
                  xaxis= dict(title= 'ML Algorithms',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Prediction Scores',ticklen= 5,zeroline= False))
fig = go.Figure(data = data, layout = layout)
iplot(fig)


################################################################################################3
from sklearn.metrics import f1_score
LRf1 = f1_score(ytrue, yprediciton1, average='weighted') 
LRf1
#K-NN
KNNf1= f1_score(ytrue, yprediciton2, average='weighted') 
KNNf1
#SVM
SVMf1=f1_score(ytrue, yprediciton3, average='weighted') 
SVMf1
#naive bayes
NBf1 = f1_score(ytrue, yprediciton4, average='weighted') 
NBf1
#Decision Tree
DTf1=f1_score(ytrue, yprediciton5, average='weighted') 
DTf1
#RandomForest
RFf1=f1_score(ytrue, yprediciton6, average='weighted') 
RFf1
scoresf1=[LRf1,KNNf1,SVMf1,NBf1,DTf1,RFf1]
#create traces

trace1 = go.Scatter(
    x = AlgorthmsName,
    y= scoresf1,
    name='Algortms Name',
    marker =dict(color='rgba(225,126,0,0.5)',
               line =dict(color='rgb(0,0,0)',width=2)),
                text=AlgorthmsName
)
data = [trace1]

layout = go.Layout(barmode = "group", 
                  xaxis= dict(title= 'ML Algorithms',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Prediction Scores(F1)',ticklen= 5,zeroline= False))
fig = go.Figure(data = data, layout = layout)
iplot(fig)