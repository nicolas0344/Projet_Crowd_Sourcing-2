#%%
from functions import *

#

user = "au"

# 

path_to_Github_folder, path_to_project, path_to_CIFAR10 = paths(user)
os.chdir(path_to_CIFAR10)
# print(os.listdir())

# Let's work on data_batch_1

file = os.path.join(path_to_CIFAR10,"test_batch")
data = unpickle(file)
meta_file = os.path.join(path_to_CIFAR10,"batches.meta")
meta = unpickle(meta_file)

labels = meta['label_names']

X = data['data']
print(X)
# (10 000, 3072)
Y = data['labels']
# 10 000

n = 9911 #random.randrange(0,len(X))
display(X[n],Y[n],labels)

#%%
os.chdir(path_to_project)

df = pd.read_csv("cifar10h-raw.csv",na_values="-99999")
df.dropna(inplace=True)

# print(len(df))
# print(len(df['cifar10_test_test_idx'].unique()))
# 514 200
# 10 000

idx = df['cifar10_test_test_idx'].unique()

K = len(df['annotator_id'].unique())
# 2571 annotators

PI = []
for i in range(K):
    PI += [np.zeros(((len(labels)),len(labels) ))]

# print(np.shape(PI[0]))
# (10, 10)

# PI_k[j][l] : in theory probability that the observer has given the label l
# where j is the right label

T = np.zeros(((len(X)),len(labels)))
# print(np.shape(T))
# (10 000, 10)
# T[i][j] : in theory 1 if j is the correct label for i

p = [0]*len(labels)
# print(len(p))
# 10

# p[j] : in theory probability that a random image has the label j

#%%
W = []
for k in range(K):

    W += [
        df.loc[abs(df['annotator_id'] - k)<10-3,
        ["cifar10_test_test_idx","chosen_label"]].astype("int") 
        ]

    if k % 100 == 0:
        print("k = ",k)

#%%
print(len(W))
print(W[0])
#
print(W[0]["cifar10_test_test_idx"][0])
#print(0 in W[0]["chosen_label"].unique())

#%%

# Initialisation of T :

for i in range(len(T)):
    for j in range(len(T[0])):
        T[i][j] = 0.5

nb_iter = 10
J = len(p) ; I = len(X)

for k in range(1):

    for n in range(nb_iter):

        # calculation of pi :
        for j in range(J):

            print("iter "+str(n)+" j "+str(j))

            for l in range(J):

                sum = 0

                for s in range(len(W[k])):

                    # si i a été labellé par l'annotateur k
                    i_0 = W[k]["cifar10_test_test_idx"][s]
                    sum += T[i_0][j]

                    # (T[i_0][j])*N_(i_0,l) avec N_(i_0,l) = 1

                #PI[k][j][l] = sum

        # calculation of p :

        # for j in range(J):
        #     sum = 0
        #     for i in range(I):
        #         sum += T[i][j]
        #     p[j]= sum / len(T)



# %%
