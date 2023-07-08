#Question 3 
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
    data = []
    anskey = []
    skip1 = 0
    dicty = {}
    for i in predata:
        if skip1 == 0:
            skip1 = 1
            continue
        j = [float(i[x]) for x in range(1,len(i)-1)]
        l = Label[i[-1]]
        data.append(j)
        anskey.append(l)
        dicty[tuple(j)] = l


#_________________Test train Split__________________

DTr, DTe, LTr, LTe = train_test_split(data, anskey, test_size=0.15, random_state=42)

#________________LDA_________________________________

def m_of_list(l): # given a list of lists returns a list of the mean of the internal lists
    avg = np.zeros(len(l[0]))
    for i in range(len(l)):
        avg += np.array(l[i])
    avg = np.dot(1/len(l) , avg)
    return list(avg)

def s_of_list(l):
    avg = np.array(m_of_list(l))
    var = np.zeros((len(l[0]),len(l[0])))

    for i in l:
        n = np.array(i)
        # print((n-avg).T)
        nn = np.array([n-avg])
        var += np.array(np.multiply((n-avg) , nn.T ))
        # print(var)
    
    return var
#Dividing into clusters

def Divide(data,dicty,n=3): #data is full of all data points ; dicty is the dictionary that relates points to its labels ; n is number of labels
    fin_dict = {}
    #initialising
    for i in range(1,n+1):
        fin_dict[i] = []
    
    for i in data:
        lab = dicty[tuple(i)]
        fin_dict[lab].append(i)
    
    return fin_dict # returns dict with label as key and list of list(data points) as values

#_______________WithinClassOps_______________
def mean(data,dicty,n=3): # returns a dict where key is label and its value is the mean of that cluster
    fin_dict = Divide(data,dicty,n)
    dict_means = {}

    for i in range(1,n+1):
        dict_means[i] = m_of_list(fin_dict[i])

    return dict_means

def mean_all(data,dicty,n=3): # mean of all points of clusters
    fin_dict = Divide(data,dicty,n)
    M = mean(data,dicty,n=3)
    a = list(dicty.keys())[0]
    mu = np.zeros(len(a))
    for i in range(1,n+1):
        mu += M[i]
    mu = np.dot(1/n , mu)
    return list(mu)

def scatter(data,dicty,n=3): # returns a dict with label as key and numpy scatter of that cluster as value
    fin_dict = Divide(data,dicty,n)
    dict_scatter = {}
    for i in range(1,n+1):
        dict_scatter[i] = s_of_list(fin_dict[i])
    return dict_scatter

def SwInv(data,dicty,n=3): # Compute the inverse of Sw which in turn was found by adding all the values of scatter (previous function)
    fin_dict = Divide(data,dicty,n)
    S = scatter(data,dicty,n)
    a = list(dicty.keys())[0]
    sw = np.zeros((len(a),len(a)))
    for i in range(1,n+1):
        sw += S[i]
    swin = np.linalg.inv(sw)
    return swin

def Sb(data,dicty,n=3):
    fin_dict = Divide(data,dicty,n)
    mu_c = mean(data,dicty,n)
    mu0 = mean_all(data,dicty,n)
    a = list(dicty.keys())[0]
    sb = np.zeros((len(a),len(a)))
    for i in range(1,n+1):
        ni = fin_dict[i]
        del_mu = np.array(mu_c[i]) - np.array(mu0)
        nn = np.array([del_mu])
        sb += np.dot(len(fin_dict[i]), np.array(np.multiply((del_mu) , nn.T )))
    
    return sb

def EigenMatrix(data,dicty,n=3): #returns Eigenvector list(list of list) and list of Eigenvalues 
    SWI = SwInv(data,dicty,n=3)
    SBI = Sb(data,dicty,n=3)
    EGM = np.dot(SWI,SBI)
    w, v = np.linalg.eig(EGM)

    return w,v

def Reduced_Comps(data,dicty,num=3,n=3):
    w,v = EigenMatrix(data,dicty)
    red = v[:num]
    return red

def New_Dat(data,dicty,num=3,n=3):
    L_Mat = Reduced_Comps(data,dicty,num,n)

    Old_D = np.array(data)
    New_D = np.dot(Old_D, L_Mat.T)
    # print(New_D.shape)
    return New_D


#________________KNN_________________________________
def most_frequent(List): #returns the most frequent element in a list
    return max(set(List), key = List.count)

#___Globally Define KNN for each data____
neigh = NearestNeighbors(n_neighbors=5)
neigh.fit(data)
knn_full = []
for i in data:
    dist,val = neigh.kneighbors([i])
    knn_full.append(val)

#_____LDA KNN______



def KNNFull(x,data,k=5):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(data)
    dist,val = neigh.kneighbors([x])
    # pot_labels = [anskey[tuple(val[i])] for i in range(len(val))]
    dist = list(dist[0])
    val = list(val[0])
    ans = [ data[val[i]] for i in range(1,len(val)) ]
    return val

def GiveLabel(x,data,anskey): #x is datapoint val is the index of the k nearest neighbours
    anslabels = []
    val = KNNFull(x,data)
    for i in val:
        anslabels.append(anskey[i])
        fin = most_frequent(anslabels)
    return fin

#________________Testing_______________________________
DTr, DTe, LTr, LTe = train_test_split(data, anskey, test_size=0.15, random_state=2)
labels = []
for i in DTe:
    labels.append(GiveLabel(i,DTr,LTr))

p = 0
for i in range(23):
    if labels[i] == LTe[i]:
        p+=1
print("Accuracy of non LDA : ", 100*p/len(labels))


#______________LDA Testing

New_Data = New_Dat(data,dicty)
DTr, DTe, LTr, LTe = train_test_split(New_Data, anskey, test_size=0.15, random_state=2)
neigh2 = NearestNeighbors(n_neighbors=5)
neigh2.fit(New_Data)
knn_lda = []
for i in New_Data:
    dist,val = neigh2.kneighbors([i])
    knn_lda.append(val)

def KNNLDA(x,New_Data,k=5):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(New_Data)
    dist,val = neigh.kneighbors([x])
    # pot_labels = [anskey[tuple(val[i])] for i in range(len(val))]
    dist = list(dist[0])
    val = list(val[0])
    ans = [ data[val[i]] for i in range(1,len(val)) ]
    return val

def GiveLabelLDA(x,New_Data,anskey): #x is datapoint val is the index of the k nearest neighbours
    anslabels = []
    val = KNNLDA(x,New_Data)
    for i in val:
        anslabels.append(anskey[i])
        fin = most_frequent(anslabels)
    return fin

labels = []
for i in DTe:
    labels.append(GiveLabelLDA(i,DTr,LTr))

p = 0
for i in range(23):
    if labels[i] == LTe[i]:
        p+=1
print("Accuracy of LDA : ", 100*p/len(labels))






    