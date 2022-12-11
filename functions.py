# General libraries

import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pandas as pd

# to load the path of the working folder :

def paths(user):
  
    if user == "a" or user =="au" or user == "aurelien" : 
        
        path_to_documents = os.path.join('C:','\\Users','lyz50',"Documents")
        path_to_Github_folder = os.path.join(path_to_documents,"Github") 
      
        path_to_project = os.path.join(path_to_Github_folder,"Projet_Crowd_Sourcing")
        path_to_CIFAR10 = os.path.join(path_to_documents,"CIFAR10","cifar-10-python","cifar-10-batches-py")

    if user == "nicolas" :
  
        path_to_Github_folder = os.path.join('C:',"\\Users","Nicolas","OneDrive","Documents","Cours S9","AtelierProjet")

        path_to_project = os.path.join(path_to_Github_folder,"Projet_Crowd_Sourcing")
        path_to_CIFAR10 = os.path.join(path_to_Github_folder,"cifar-10-batches-py")

    return path_to_Github_folder, path_to_project, path_to_CIFAR10

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
