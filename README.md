# Projet_Crowd_Sourcing

## Labels :

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| - | - | - | - | - | - | - | - | - | - |
| airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |

## Documentation :

#

CIFAR 10 :
https://www.cs.toronto.edu/~kriz/cifar.html

The CIFAR-10 dataset consists of 60 000 32x32 colour images in 10 classes,
with 6 000 images per class.

There are 50 000 training images and 10 000 test images :
- 5 training batches of 10 000 images each (total training : 5 000 imgs from each class)
(some training batches may contain more images from one class than another)
- test batch contains 10 000 imgs : 1 000 randomly-selected images from each class.

Tutorial to load CIFAR-10 :
https://www.binarystudy.com/2021/09/how-to-load-preprocess-visualize-CIFAR-10-and-CIFAR-100.html#download

#

CIFAR 10h : https://github.com/jcpeterson/cifar-10h

CIFAR-10h is a dataset of labels reflecting human perceptual uncertainty for the 10 000 images CIFAR-10 test set.

- data/cifar10h-counts.npy - 10 000 x 10 numpy matrix containing human classification counts (out of ~50) for each image and class

- data/cifar10h-probs.npy - 10000 x 10 numpy matrix containing normalized human classification counts (probabilities) for each image and class.

## Instructions for application in code_EM_DAWID-SKENE : 

 You have to download cifar10 for pyhton at the link : http://www.cs.toronto.edu/~kriz/cifar.html
  - Extrate files
  - Create a folder containing : cifar-10-batches-py , next to "Projet_Crowd_Sourcing" folder 

 Inter in the file functions.py in the path : code_EM_DAWID-SKENE 
  - On comment lines 
  - Inter on (if user == ".." :) your name user ( like : test ) 
  - Inter and your path to your GitHub file (before Projet_Crowd_Sourcing) in path_to_Github_folder = os.path.join( .. )
  - Uncomment lines 

 Inter in the file script.py in the path : code_EM_DAWID-SKENE 
  - Inter your name user at the beginning

 Run cells 
