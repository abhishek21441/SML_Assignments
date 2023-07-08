#Question 3
import numpy as np
import sympy
import csv
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import sklearn
import random
import math 

#_________________Data Extraction___________________
with open("C:\\Abhishek\\IIITD\\Academics\\CSE\\CSE342_SML\\Assignment2\\Real Estate.csv") as file:
    predata = csv.reader(file)
    data = []
    anskey = []
    skip1 = 0
    for i in predata:
        if skip1 == 0:
            skip1 = 1
            continue
        j = [float(i[x]) for x in range(1,len(i)-1)]
        j.insert(0,1)
        # print(j)
        l = float(i[-1])
        data.append(j)
        anskey.append(l)

# print(np.array(data).shape)
# print(np.array(data).T.shape)

#__________________Calculating Theta_________________

Syminv = np.linalg.inv(np.dot(np.array(data).T,np.array(data)))
XTy = np.dot(np.array(data).T,np.array(anskey))
theta = np.dot(Syminv,XTy)

#_________________Expected Y values -- Hypothesis Function___________________

ExpY = np.dot(theta.T,np.array(data).T)
# print(ExpY.shape)

#_________________Errors______________________________________________________

RealMean = np.mean(anskey)
# print(RealMean)
TSS = sum([(anskey[i] - RealMean)**2 for i in range(len(anskey))])
RSS = sum([(anskey[i] - ExpY[i])**2 for i in range(len(anskey))])

R2 = 1- (RSS/TSS)

RMSE = math.sqrt(RSS/len(anskey))
print(R2)
print(RMSE)
