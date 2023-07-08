#Question 5 fix false positive thing!!!!!!!!!!
import numpy as np
import sympy
import csv
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import sklearn
import random
import math 

#_______________Dictionary_________________________

Label = {"Iris-setosa" : 1, "Iris-versicolor" : 2, "Iris-virginica" : 3}

#_________________Data Extraction___________________
with open("C:\\Abhishek\\IIITD\\Academics\\CSE\\CSE342_SML\\Iris\\Iris.csv") as file:
    predata = csv.reader(file)
    data,data2 = [],[]
    anskey = []
    dans = {}
    skip1 = 0
    A,An,Al = [],[],[]
    B,Bn,Bl = [],[],[]
    C,Cn,Cl = [],[],[]
    for i in predata:
        if skip1 == 0:
            skip1 = 1
            continue
        j = [float(i[x]) for x in range(1,len(i)-1)]
        l = Label[i[-1]]

        data.append(j)
        dans[tuple(j)] = l
        data2.append([1] + j)
        anskey.append(l)

        #Dividing into 6 buckets
        if l == 1 :
            A.append([1] + j)
            Al.append(1)
        else:
            An.append([1] + j)
            Al.append(0)
        if l == 2 :
            B.append([1] + j)
            Bl.append(1)
        else:
            Bn.append([1] + j)
            Bl.append(0)
        if l == 3 :
            C.append([1] + j)
            Cl.append(1)
        else:
            Cn.append([1] + j)
            Cl.append(0)
        
# X_train, X_test , Y_train , Y_test = train_test_split(data2, anskey , test_size = 0.2, random_state = 0)


#____________Logistic Regression_____________

def z(x,t): # x is one datapoint of data
    a = np.dot(np.array([x]),np.array([t]).T)
    a = a[0][0]
    return(float(a))

def sigmoid(x,t): # given data point and weights it returns the sigmoid function
    ans  = 1/(1 + np.exp(-1 * z(x,t)))
    return ans

def pred(x,t): # given one sample datapoint x and the weights it returns the predicted value
    y = sigmoid(x,t)
    return y

def derivative(data,t,anskey): # returns an array where jth index is the derivative of cost wrt to t[j]
    der_list = [0.0]*len(data[0])
    for j in range(len(der_list)):
        for i in range(len(data)):
            x = data[i]
            der_list[j] += ((pred(x,t) - anskey[i]) * x[j])
    for i in der_list:
        i = i/len(data)
    return der_list

def update_weight(data,t,anskey,alpha = 0.001): # updates every index of t to return the new t at once
    der_list = derivative(data,t,anskey)
    for j in range(0,len(t)):
        t[j] -= alpha * der_list[j]
    return t

def final_t(data,t,anskey,alpha= 0.001): # function that keeps updating t until we fall below a certain threshold and then returns final t
    for i in range(0,100):
        t = update_weight(data,t,anskey,alpha)
    return t

#___________________Testing______________

ta = [0.0] * len(data2[0])
tb = [0.0] * len(data2[0])
tc = [0.0] * len(data2[0])


ta = final_t(data2,ta,Al,0.0012)


fin_a = [pred(data2[i],ta) for i in range(0,len(data2))]
# fin_b = [pred(X_train[i],tb) for i in range(0,len(X_train))]

pred_a = []
# print(len(fin_a))
# print(len(Al))
train_b = [] #those that are not A acc to model
Bl = []
count = 0
B_CDict = {2:1,3:0}
for i in range(0,len(fin_a)):
    if fin_a[i]>0.5:
        pred_a.append(1)
    else:
        pred_a.append(0)
        train_b.append(data2[i])
        Bl.append(B_CDict[anskey[i]])

acca = 0
for i in range(len(Al)):
    if pred_a[i] == Al[i]:
        acca +=1
print("Accuracy classifying into A and Not A",100*acca/len(Al))

tb = final_t(train_b,tb,Bl,0.001)
#Those that are not A found, now classify those into B and not B i.e. C
#with al = 0.005 t = 0.99 for b and c
fin_b = [pred(train_b[i],tb) for i in range(0,len(train_b))]
pred_b = []
for i in fin_b:
    if i>0.5:
        pred_b.append(1)
    else:
        pred_b.append(0)
accb = 0
for i in range(len(Bl)):
    if Bl[i] == pred_b[i]:
        accb += 1

print("Accuracy classifying into B and Not B",100*accb/len(Bl))







