# General libraries

import numpy as np
import matplotlib.pyplot as plt
import os

# to load the path of the working folder :

def paths(user):
    if user == "a" or user =="au" or user == "aurelien" :
        path_to_Github_folder = os.path.join('C:','\\Users','lyz50',"Documents","Github")
        path_to_project = os.path.join(path_to_Github_folder,"Projet_Crowd_Sourcing")
    if user == "nicolas" :
        print("mets ton chemin ici")
    return path_to_Github_folder, path_to_project

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