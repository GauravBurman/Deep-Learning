# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:31:32 2019

@author: gaura
"""

#classes for neural network
from AdvancedAnalytics import NeuralNetwork
from sklearn.neural_network import MLPClassifier

#other needed classes
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from AdvancedAnalytics import ReplaceImputeEncode
import math
import pandas as pd
import numpy as np


df = pd.read_excel ("CreditHistory_Clean.xlsx") 
print("Authentication Data with %i observations & %i attributes.\n" \
%df.shape, df[0:2])

#dropping target variable
df_input = df.drop('good_bad', axis = 1)

#data dictionary
data_map = {'age':['I',(19,120), [0,0]], \
            'amount':['I', (0,20000),[0,0]],
            'checking':['N',(1,2,3,4),[0,0] ],
            'coapp':['N',(1,2,3),[0,0]],
            'depends':['B',(1,2),[0,0]],
            'duration':['I',(1,72), [0,0]],
            'employed':['N',(1,2,3,4,5),[0,0]],
            'existcr':['N',(1,2,3,4),[0,0]],
            'foreign':['B',(1,2),[0,0]],
            'history':['N',(0,1,2,3,4),[0,0]],
            'installp':['N',(1,2,3,4),[0,0]],
            'job':['N',(1,2,3,4),[0,0]],
            'marital':['N',(1,2,3,4),[0,0]],
            'other':['N',(1,2,3),[0,0]],
            'property':['N', (1,2,3,4),[0,0]],
            'purpose':['N', ('0','1','2','3','4','5','6','8','9','X'),[0,0]],
            'resident':['N',(1,2,3,4), [0,0]],
            'savings':['N',(1,2,3,4,5),[0,0]],
            'telephon':['B',(1,2),[0,0]],
            'good_bad': ['B',('good','bad'), (0,0)]}

#we need to encode the categorical variables
rie = ReplaceImputeEncode(data_map=data_map, nominal_encoding='one-hot', \
interval_scale='std', display=True, drop= False)

df_rie = rie.fit_transform(df)
print(df_rie)

#good_bad is the name of the binary target

X = df_rie.drop('good_bad', axis = 1)
Y= df['good_bad']
Y = Y.map({'good':1,'bad':0})
y = np.ravel(Y)

features = ['age', 'amount', 'duration', 'depends', 'foreign', 'telephon',
       'checking0', 'checking1', 'checking2', 'checking3', 'coapp0', 'coapp1',
       'coapp2', 'employed0', 'employed1', 'employed2', 'employed3',
       'employed4', 'existcr0', 'existcr1', 'existcr2', 'existcr3', 'history0',
       'history1', 'history2', 'history3', 'history4', 'installp0',
       'installp1', 'installp2', 'installp3', 'job0', 'job1', 'job2', 'job3',
       'marital0', 'marital1', 'marital2', 'marital3', 'other0', 'other1',
       'other2', 'property0', 'property1', 'property2', 'property3',
       'purpose0', 'purpose1', 'purpose2', 'purpose3', 'purpose4', 'purpose5',
       'purpose6', 'purpose7', 'purpose8', 'purpose9', 'resident0',
       'resident1', 'resident2', 'resident3', 'savings0', 'savings1',
       'savings2', 'savings3', 'savings4']
classes = ['good','bad']

#Fitting neural networks to different hidden layers
print("\n******** NEURAL NETWORK ********")
#Neural Network
network_list = [(3), (11), (6,5), (7,6), (8,7), (5,4)]
max_recall = 0
best_network = (0)
score_list = ['accuracy','precision', 'f1', 'recall']

for nn in network_list:
    print("\nNetwork: ", nn)
    fnn = MLPClassifier(hidden_layer_sizes=nn,solver='lbfgs', max_iter=1000, tol= 1e-64, random_state=12345)
    mean_score = []
    std_score = []
    print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean","Std. Dev."))
    for s in score_list:
        fnn_4 = cross_val_score(fnn,x,y,scoring=s,cv=4)
        mean=fnn_4.mean()
        std=fnn_4.std()
        mean_score.append(mean)
        std_score.append(std)
        print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
    
    
print("The highest f1 score is 0.8358 for hidden_layer_sizes= (6,5)")
x_train, x_validate, y_train, y_validate =train_test_split(x,y,test_size = 0.3, random_state=7)
fnn = MLPClassifier(hidden_layer_sizes=(6,5),activation='tanh',\
                    solver='lbfgs', max_iter=2000, random_state=12345)
fnn = fnn.fit(x_train, y_train)
NeuralNetwork.display_binary_split_metrics(fnn, x_train, y_train, x_validate,y_validate)