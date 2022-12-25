#%%
from functions import *
  
#
  
user = "nicolas"
  
# 
  
path_to_Github_folder, path_to_project, path_to_CIFAR10 = paths(user)
os.chdir(path_to_CIFAR10)
# print(os.listdir())
  
# Let's work on data_batch_1
data = unpickle(os.path.join(path_to_CIFAR10,"test_batch")) #images 
meta = unpickle(os.path.join(path_to_CIFAR10,"batches.meta")) #labels
  
labels = meta['label_names']  #nom des labels

X = data['data']   # images couleurs (10 000, 3072)
Y = data['labels']   # numéro de labélisation  10 000
  
n = 9912 #random.randrange(0,len(X))
display(X[n],Y[n],labels)

#%%
os.chdir(path_to_project)

df = pd.read_csv("cifar10h-raw.csv",na_values="-99999")
df.dropna(inplace=True)  # delete annotators/spamers

# print(len(df))
# print(len(df['cifar10_test_test_idx'].unique()))
# 514 200
# 10 000

idx = df['cifar10_test_test_idx'].unique()  #numéro de l'image
K = len(df['annotator_id'].unique())
# 2571 annotators

PI_k = []
N_k = []
for i in range(K):
    PI_k += [ np.ones((len(labels),len(labels))) ]
    N_k += [ np.ones((len(X),len(labels))) ]

# print(np.shape(PI_k[0]))
# (10, 10)

# PI_k[j][l] : in theory probability that the observer has given the label l
# where j is the right label

T = np.ones((len(X),len(labels)))
# print(np.shape(T))
# (10 000, 10)
# T[i][j] : in theory 1 if j is the correct label for i

p = [1]*len(labels)
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

PI_test = []
N_test = []
for i in range(9):
    PI_test += [ np.ones((5,5)) ]
    N_test += [ np.ones((10,5)) ]

T_test = np.ones((10,5))
p_test = [1]*5

b = 0
for j in range(5):
    b += sum(T_test[:,0]*N_test[0][:,j])

a = sum(T_test[:,0]*N_test[0][:,0])



def produit_liste(A):
    a = 1
    for i in A: 
        a = a*i
    return(a)


def EM_stepE(T,N):
    I = len(T[:,0])
    J = len(T[0,:])
    K = len(N)
    
    PI = []
    for k in range(K):
        PI += [ np.ones((J,J)) ]
    p = [1]*J
    
    for k in range(K):
        for l in range(J): 
            
            sum_j = 0.0
            for a in range(J):
                sum_j += sum(T[:,l]*N[k][:,a]) #step E
            
            for j in range(J): 
                sum_i = sum(T[:,l]*N[k][:,j])
                PI[k][l,j] = sum_i/sum_j  #step E
            
            p[l] = sum(T[:,l])/I
    return([PI,p,N,T])

EM_stepE(T_test, N_test)


def EM_stepM(f):
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
            
            prod_1 = 1
            for k in range(K):
                for j in range(J):
                    prod_1 = prod_1 * p[l] * PI[k][l,j]**(N[k][i,j])
            
            prod_2.append(prod_1)
        
        for l in range(J): 
            T[i,l] = prod_2[l]/sum(prod_2)
    return(T)
        
EM_stepM(EM_stepE(T_test,N_test))

    
def EM_algo(T,N,n):
    
    for i in range(n):
        T = EM_stepM(EM_stepE(T,N))
    return(T)

EM_algo(T_test,N_test,10)














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
