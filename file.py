from functions import *

#

user = "au"

# 

path_to_Github_folder, path_to_project = paths(user)
path_to_data = os.path.join(path_to_project,"cifar-10-python","cifar-10-batches-py")
os.chdir(path_to_data)
# print(os.listdir())

# Let's work on data_batch_1

file = os.path.join(path_to_data,"data_batch_1")
data = unpickle(file)
meta_file = os.path.join(path_to_data,"batches.meta")
meta = unpickle(meta_file)

labels = meta['label_names']

########################################
# For one observer :
########################################

X = data['data']
# (10 000, 3072)
Y = data['labels']
# 10 000

N = np.zeros(((len(X)),len(labels)))
# print(np.shape(N))
# (10 000, 10)

# N[i][l] = number of times that the observer has given the label l to the i_th image

PI = np.zeros(( (len(labels)),len(labels) ))
# print(np.shape(PI))
# (10, 10)

# p[j][l] : in theory probability that the observer has given the label l
# where j is the right label

T = np.zeros(((len(X)),len(labels)))
# print(np.shape(T))
# (10 000, 10)

# T[i][j] : in theory 1 if j is the correct label for i

p = [0]*len(labels)
# print(len(p))
# 10

# p[j] : in theory probability that a random image has the label j

########################################

# Initialisation of T :

for i in range(len(T)):
    for j in range(len(T[0])):
        T[i][j] = 0

nb_iter = 100

for n in range(nb_iter):
    0
    # calculation of pi

    # calculation of p


