# General libraries

import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pandas as pd


#%%
# to load CIFAR 10 :

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo: 
        dict = pickle.load(fo, encoding='latin1')
    return dict

# to display a single image :

def display(image,label,labels): 
    
    image = image.reshape(3,32,32)
    image = image.transpose(1,2,0)
    plt.imshow(image)

    plt.title(labels[label])
    plt.show()

#%%

# to create matrix of enumeration N^k_{i,j} in the theory model
# A is a dataframe

def create_matrix_N(A,I,J):
    N = np.zeros((I,J))
    for i in list(A['cifar10_test_test_idx']):
        i = int(i)
        j = int(A[A['cifar10_test_test_idx'] == i]['chosen_label'])
        N[i,j] = 1
    return(N)

#%%

# EM algorithm with each setp (step E, step M )

def EM_stepM(T,N):
    I = len(T[:,0])
    J = len(T[0,:])
    K = len(N)
    
    PI = []
    for k in range(K):
        PI += [ np.ones((J,J)) ]
    p = [1/J]*J
    
    for k in range(K):
        for l in range(J): 
            sum_j = 0.0
            
            for a in range(J):
                sum_j += sum(T[:,l]*N[k][:,a]) 
            
            for j in range(J): 
                sum_i = sum(T[:,l]*N[k][:,j])
                PI[k][l,j] = sum_i/sum_j
                print(PI[k][l,j])
            
            p[l] = sum(T[:,l])/I
    return([PI,p,N,T])


def EM_stepE(f):
    PI = f[0]
    p = f[1]
    N = f[2]
    T = f[3]
    
    I = len(N[0][:,0])
    J = len(N[0][0,:])
    K = len(N)
    
    for i in range(I):
        prod_2 = []
        for l in range(J):
            prod_1 = p[l]
            for k in range(K):
                for j in range(J):
                    prod_1 = prod_1 * (PI[k][l,j]**(N[k][i,j]))
            
            prod_2.append(prod_1)
        
        for l in range(J): 
            T[i,l] = prod_2[l]/sum(prod_2)
    return(T)


# warning : here we have taken label in the order, important for the link 
# between the chosen label and the number of the image 

def EM_algo(T,N,n):
    I = len(T[:,0])
    label = np.zeros((I,2))
    
    for i in range(n):
        T = EM_stepE(EM_stepM(T,N))
    
    for i in range(I):
        list_T = list(T[i,:])
        label[i,:] = [int(i),list_T.index(max(list_T))]
    
    return(label)
# fist column if label is the numerotation of the image 
# second column is the the chosen label in the model
