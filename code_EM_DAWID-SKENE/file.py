#%%
from functions import *
  
#
  
user = "nicolas"
  
# 
  
path_to_Github_folder, path_to_project, path_to_CIFAR10 = paths(user)
os.chdir(path_to_CIFAR10)
# print(os.listdir())
  
# Let's work on data_batch_1
  
file = os.path.join(path_to_CIFAR10,"test_batch")  
data = unpickle(file) #images 
meta_file = os.path.join(path_to_CIFAR10,"batches.meta")
meta = unpickle(meta_file)  #labels
  
labels = meta['label_names']  #nom du label 
  
X = data['data']   # images couleurs
print(X)
# (10 000, 3072)
Y = data['labels']   # numéro de labélisation 
# 10 000
  
n = 9912 #random.randrange(0,len(X))
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

PI_k = []
for i in range(K):
    PI_k += [np.zeros(((len(labels)),len(labels) ))]

# print(np.shape(PI_k[0]))
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
for i in range(len(idx)):

    W += [
        df.loc[abs(df['cifar10_test_test_idx'] - i)<10-3,
        ["annotator_id","chosen_label"]].astype("int") 
        ]

    if i % 100 == 0:
        print("i = ",i)

#%%
print(W[0])
print(8 in W[0]["chosen_label"].unique())

#%%

# Initialisation of T :

for i in range(len(T)):
    for j in range(len(T[0])):
        T[i][j] = 0.5

nb_iter = 10
J = len(p) ; I = len(X)

for n in range(nb_iter):

    for k in range(K):

        # calculation of pi :
        for j in range(J):

            print("iter "+str(n)+" j "+str(j))

            TN = 0
            for l in range(J):
                sum = 0
                for i in range(I):

                    # N_(i,l) = number of times that the observer has given the label l to the i_th image
                    # 0 or 1 in or case
                    if (l in W[i]["chosen_label"].unique()):
                        
                        sum += T[i][j] # (T[i][j])*N_(i,l) avec N_(i,l) = 1

                PI_k[k][j][l] = sum
                TN += sum

            for l in range(J):

                PI_k[k][j][l] = PI_k[k][j][l]/TN

        # calculation of p :

        for j in range(J):
            sum = 0
            for i in range(I):
                sum += T[i][j]
            p[j]= sum / len(T)



# %%
