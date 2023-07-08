#Question2 attempt 2
import csv
import numpy as np
import copy
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

#___________Necessary Dictionaries + PreProcessing_________________

CPDict = {"asymptomatic" : 0, "nonanginal" : 1, "typical" : 2, "nontypical" : 3}
THALDict = {"normal" : 0,  "reversable" : 1, "fixed" : 2}
AHDDict = {"No" : 0, "Yes" : 1} #label columns

with open("C:\\Abhishek\\IIITD\\Academics\\CSE\\CSE342_SML\\Assignment2\\Heart.csv") as predata:
    data= []
    C1,C2 = [],[] #C1 = no ; C2 = yes
    anskey = []
    skip1 = 0
    s = csv.reader(predata)
    for i in s:
        if skip1 == 0:
            skip1 =1
            continue
        i = i[1:]
        if "NA" in i:
            continue
        i[2] = CPDict[i[2]]
        i[12] = THALDict[i[12]]
        i[13] = AHDDict[i[13]]

        stability = 0.479665515
        j = [float(i[x]) for x in range(0,len(i)-1)]
        data.append(j)
        anskey.append(i[13])
        # print(i[-1])
        if i[-1] == 1:
            C2.append(j)
        else:
            C1.append(j)

X_train, X_test , Y_train , Y_test = train_test_split(data, anskey , test_size = 0.2, random_state = 0)

#______________Principal Component Analysis____________\
pca = PCA(n_components=3)
pca.fit(data)
x = pca.transform(data)
# print(x.shape)
# print(list(x[0]))
with open("C:\\Abhishek\\IIITD\\Academics\\CSE\\CSE342_SML\\Assignment2\\Heart.csv") as predata:
    PC1 = []
    PC2 = [] #C1 = no ; C2 = yes
    skip1 = 0
    s = csv.reader(predata)
    count = -1
    for i in s:
        if skip1 == 0:
            skip1 =1
            continue
        
        i = i[1:]
        if "NA" in i:
            continue
        count += 1
        i[13] = AHDDict[i[13]]
        j = list(x[count])
        if i[-1] == 1:
            PC2.append(j)
        else:
            PC1.append(j)


#_______________Fischer Discriminant Analysis_______________

#______________________Mean and Variance of Data_________

def mean(data):
    avg = np.array([[0.0]*len(data[0])])
    for i in data:
        n = np.array(i)
        avg += n
    # for i in avg:
    #     i = i/len(data)   
    avg = np.dot(1/len(data) , avg)
    return avg

def scatter(data):
    avg = mean(data)
    var = np.zeros((len(data[0]),len(data[0])))
    # print(var.shape)

    for i in data:
        n = np.array(i)
        var += np.array(np.multiply((n-avg) , (n-avg).T ))
    np.dot(1/(len(data) -1) , var)
    # print(var.shape)
    return var

#______________Computing necessary terms_________________

#Means : M1(Mean of C1), M2(Mean of C2), S1(Scatter matrix between for C1), S2(Scatter matrix between for C2), N1(Number of datapoints for C1), N2(Number of datapoints for C2)

#Cost function

def Cost(C1,C2):
    w = Wmax(C1,C2)
    N1,N2 = len(C1),len(C2)
    M1,M2 = mean(C1),mean(C2)
    S1,S2 = scatter(C1),scatter(C2)
    
    wgtsc = N1*S1 + N2*S2 #calculates sigma
    num = np.dot( np.array([M1-M2]) , np.array([w]).T ) #calculates m.t . w
    num = float(num[0][0])
    num = num**2

    den = np.dot( np.dot( np.array([w]) , wgtsc ) , np.array([w]).T )
    den = float(den[0][0])
    J = num/den

    return J

#Max w Direction

def Wmax(C1,C2):
    N1,N2 = len(C1),len(C2)
    M1,M2 = mean(C1),mean(C2)
    S1,S2 = scatter(C1),scatter(C2)
    wgtsc = np.dot(N1,S1) + np.dot(N2,S2)
    # print(wgtsc)
    wgtscinv = np.linalg.inv(np.dot(N1,S1) + np.dot(N2,S2))
    # print("Wgtscinv : ", wgtscinv.shape)
    meandiff = (M1-M2)
    # print("meandiff: ",meandiff.shape)
    wmax = np.dot(wgtscinv,meandiff.T)
    # print(wmax.shape)
    return wmax

#___________________Projections____________

def Project(w,data): # project all x onto w and returns a list of float values
    proj_data = []
    # print(np.array([data[0]]).shape)
    # print(np.dot(w.T , np.array([data[0]]).T))
    for i in data:
        d = np.dot(w.T , np.array([i]).T)
        d = d[0][0]
        d = float(d)
        proj_data.append(d)
    return proj_data

def addingone(fin_dat):
    new_dat = []
    for i in range(0,len(fin_dat)):
        arr = [1,fin_dat[i]]
        new_dat.append(arr)
    return new_dat

#_________________Logistic Regression________

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
    der_list = [0]*len(data[0])
    for j in range(len(der_list)):
        for i in range(len(data)):
            x = data[i]
            der_list[j] += ((pred(x,t) - anskey[i]) * x[j])
    for i in der_list:
        i = i/len(data)
    return der_list

def update_weight(data,t,anskey,alpha = 0.027154565): # updates every index of t to return the new t at once
    der_list = derivative(data,t,anskey)
    for j in range(0,len(t)):
        t[j] -= alpha * der_list[j]
    return t

def final_t(data,t,anskey,alpha= 0.027154565): # function that keeps updating t until we fall below a certain threshold and then returns final t
    for i in range(0,1000):
        t = update_weight(data,t,anskey,alpha)
    return t

#_________________Testing________________

# Just FDA and Logistic
W = Wmax(C1,C2)

new_dat = Project(W,data)
fin_dat = addingone(new_dat)
ini_t = [0,0]

t_fin = final_t(fin_dat,ini_t,anskey,0.02715456)
z_fin = [pred(fin_dat[i],t_fin) for i in range(0,len(fin_dat))]


acc = 0
predi=[]

for i in range(0,len(z_fin)):
    if z_fin[i]<0.5:
        predi.append(0)
    else:
        predi.append(1)

for i in range(0,len(z_fin)):
    if anskey[i] == predi[i]:
        acc+=1

print( "Accuracy with FDA only : ",100*acc/297)

#_________PCA + FDA__________
W = Wmax(PC1,PC2)
new_dat = pca.transform(data)
new_dat = Project(W,new_dat)
fin_dat = addingone(new_dat)
ini_t = [0,0]

t_fin = final_t(fin_dat,ini_t,anskey,0.027154565)
z_fin = [pred(fin_dat[i],t_fin) for i in range(0,len(fin_dat))]


acc = 0
predi=[]
c = 0.4796655137
for i in range(0,len(z_fin)):
    if z_fin[i]<0.5:
        predi.append(0)
    else:
        predi.append(1)

for i in range(0,len(z_fin)):
    if anskey[i] == predi[i]:
        acc+=1

print( "Accuracy with PCA + FDA only : ",100*acc/297)

# print(len(C1))
# print(len(PC1))