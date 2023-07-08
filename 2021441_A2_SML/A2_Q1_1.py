#Question 1
import numpy as np
import sympy
import csv
from sklearn.neighbors import NearestNeighbors
import random
import math 

#____________Data Retrival________________________________

file = open("C:\\Abhishek\\IIITD\\Academics\\CSE\\CSE342_SML\\Assignment2\\glass.csv", "r")
data = list(csv.reader(file, delimiter=","))
skip1 = 0
findata = []
for i in range(0,len(data)):
    if skip1 == 0:
        skip1 = 1
        continue
    # anskey.append(d[data[i][1]])
    findata.append([float(data[i][x]) for x in range(0,len(data[i])-1)])

# print(findata)


#__________________Mahalanobis Distances_____________________________

def Mahalanobis(findata,no_comp = 9): # given a data set of columns being attributes and data entries being rows return 

    data = np.array(findata).T # it is a numpy array of 27 features(rows, because numpy :P) and 50000 data points (columns)
    avg = np.array([0.0]*no_comp) # to compute the mean of the 50000 datapoints
    for j in range(0,no_comp):
        for i in range(len(findata)):
            avg[j] += findata[i][j]/len(findata)

    avg = np.array(avg)# mean of all data entries
    # print(avg[:5])
    C = np.cov(data) # 27 * 27 covariance matrix of data
    # print(C)
    temp = np.asarray(sympy.Matrix(C).inv()).astype(float)
    C_I = np.linalg.inv(C) # inverse of covariance matrix
    # print(C)


    mhnlbs = [] # stores mahalanobis distance of each point in our dataset
    for i in range(len(findata)):
        s1 = (np.array(findata[i]) - avg) # diff between datapoint and mean
        di = abs(float(np.dot(np.dot(s1.T,C_I),s1)))
        # if(di<0): 
        #     print(di)
        mhnlbs.append((di)**(1/2)) # calculating mahalanobis distance

    return mhnlbs

#_____________________________________Local Outlier Factor_______________________________

def EuDist(v1,v2):
    # v1,v2 are vectors
    l = len(v1)
    dist = 0
    for i in range(l):
        dist += (v1[i]-v2[i])**2
    dist = dist**(1/2)
    return dist


def knndist (v1, V, knndict,MinPts = 3): #finds the kdistance of a point
    index = V.index(v1)
    dist = EuDist(v1,knndict[index][-1])

    return dist

def Nbhood(v1,V,knndict,MinPts = 3):
    index = V.index(v1)
    point_idx = knndict[index]
    neighbours = [V[i] for i in range(1,len(point_idx))]

    # print(len(neighbours))

    return neighbours

def RchDist(v1,v2,V,knndict,MinPts = 3):
    index = V.index(v2)
    v3 = V[knndict[index][-1]] #Kth neighbour of v2
    return max(EuDist(v2,v3), EuDist(v1,v2))

def LRD(v1,V,knndict,MinPts = 3):
    sumRD = 0
    NBHD = Nbhood(v1,V,knndict,MinPts)
    for i in range(MinPts):
        sumRD += RchDist(v1,NBHD[i],V,knndict,MinPts) 
    sumRD = sumRD/MinPts

    lrd = 1/sumRD

    return lrd

def LOF(v1,V,knndict,MinPts = 3):
    lof = 0
    NBHD = Nbhood(v1,V,knndict,MinPts)
    for i in range(MinPts):
        lof += LRD(NBHD[i],V,knndict,MinPts)/LRD(v1,V,knndict,MinPts)
    lof = lof/(2*MinPts)

    return lof

#_______________Create Dictionary for MHNBLS Distance and LOF__________________

def MHDict(V,no_comp=9):
    M = Mahalanobis(V,no_comp)
    MHDict = {}
    for i in range(len(V)):
        MHDict[tuple(V[i])] = M[i] # key = point value = MHD
    
    return MHDict

def LOFDict(V,knndict):
    LOFDict = {}
    for i in range(len(V)):
        LOFDict[tuple(V[i])] = LOF(V[i],V,knndict) #key = point value = LOF
    
    return LOFDict

#________________Otsu Thresholding___________________
#Given a dictionary of Points and its value (Either MH Distance or LOF it returns OTSU) for a given t(hreshold) aswell
#How to choose a t?

def otsuCluster(OD,t): # performs clustering for a given threshold
    ansD = {}
    for i in OD:
        if OD[i]<t: # foreground
            ansD[i] = 0
        else: # background
            ansD[i] = 1
    return ansD

def Prob(D): # given the dictionary with clusters it returns probability of foreground
    countf,count = 0,0
    for i in D:
        count += 1
        if D[i] == 0:
            countf += 1
    return countf/count

def Means(D,OD): # returns array with 0th being mean value(MH Dist or LOF) of foreground and 1st being mean of background #OD Being orignal Dict IE one with MHD or LOF
    avgfg, avgbg = 0,0
    countf,countb = 0,0
    for i in D:
        if D[i] == 0:
            avgfg += OD[i]
            countf += 1
        else:
            avgbg += OD[i]
            countb += 1
    avgfg = avgfg/countf
    avgbg = avgbg/countb
    
    ansList = [avgfg,avgbg,countf,countb]
    return ansList

def Variance(D,OD): # return variance of foreground and background
    avgf,avgb,countf,countb = Means(D,OD)
    varfg,varbg = 0,0
    
    for i in OD:
        varfg += ((OD[i] - avgf)**2)
        varbg += ((OD[i] - avgb)**2)
    varfg = varfg/(countf-1)
    varbg = varbg/(countb-1)

    varList = [varfg,varbg]
    return varList

def Otsu(OD,t):
    D = otsuCluster(OD,t)
    p,v = Prob(D),Variance(D,OD)
    ans  = p*v[0] + (1-p)*v[1] # Otsu Equation
    return ans

#______________________________________MH Testing________________________


M2 = MHDict(findata)
t = 3.8
outl = 0
for i in M2:
    if M2[i]>t:
        outl+=1
print("Outliers in Mahalabonis: ",outl)
print(Otsu(M2,3.8))



#________________________________________LOF Testing_____________________


#___________Sets up KNN__________________________
neigh = NearestNeighbors(n_neighbors=4)
neigh.fit(findata) #fit created
knndict = []

for i in findata:
    dist,val = neigh.kneighbors([i]) 
    knndict.append(val[0]) #the ith index of knndict stores the indexes of the k nearest neighbours of findata[i] wrt findata

#_____LOF___________

LOFList = []
for i in range(len(findata)):
    LOFList.append(LOF(findata[i],findata,knndict)) # Calculates LOF of every point in findata
LOFList.sort()

L2 = LOFDict(findata,knndict)

min,max = LOFList[2],LOFList[-3]
t2 = min
minv,mint = 10000,t2
while(t2<max):
    if minv<Otsu(L2,t2):
        minv = Otsu(L2,t2)
        mint = t2
    t2 += 0.01

print("Min t : ",mint)
    


# t = 3.8
t = 0.44
outl = 0
for i in L2:
    if L2[i]>t:
        outl += 1

print("Outliers in LOF: ",outl)
print(Otsu(L2,t))




