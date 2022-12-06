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

        print("mets tes chemins ici")

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


# to generate a "false" observer with the ground truth :
# useless if we use cifar10h

def observer(X,Y):

    N = np.zeros(((len(X)),len(Y)))
    
    for i in range(len(X)):

        true_label = Y[i]

        N[i][true_label] = 10

        for l in range(len(Y)):
            if l!=true_label :
              N[i][l] = 1

    return N

# to get the counts matrix of an observer

def annotator_matrix(annotator_id,path_to_project):

    os.chdir(path_to_project)
    df = pd.read_csv("cifar10h-raw.csv")

    i = annotator_id

    if i<0 or i> 2570:
        raise ValueError(
            "annotator_id not in [0:2570] dear friend"
            )

    # print(df.loc[df['annotator_id'] == annotator_id,"chosen_label"])
    print(df.loc[df['image_filename'] == "cabin_cruiser_s_000814.png"])

    return i