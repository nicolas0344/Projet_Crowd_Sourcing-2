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
  
n = 9911 #random.randrange(0,len(X))
display(X[n],Y[n],labels)

#%%
os.chdir(path_to_project)

df = pd.read_csv("cifar10h-raw.csv",na_values="-99999")
df.dropna(inplace=True)  # delete annotators/spamers
<<<<<<< HEAD
=======

print(len(df))
print(len(df['cifar10_test_test_idx'].unique()))
print(len(df['annotator_id'].unique()))
print("")

>>>>>>> refs/remotes/origin/main
df_test = df[['annotator_id','true_label','chosen_label','cifar10_test_test_idx']]


#On réduit la taille du jeu de donnée pour des fins numériques

df_test = df_test[df_test['annotator_id']<150]
print(len(df_test['annotator_id'].unique()))
print(len(df_test['cifar10_test_test_idx'].unique()))
df_test = df_test[df_test['cifar10_test_test_idx']<250]
print(len(df_test['cifar10_test_test_idx'].unique()))
print(len(df_test['annotator_id'].unique()))

K = len(df_test['annotator_id'].unique())
I = len(df_test['cifar10_test_test_idx'].unique())
J = 10

T_test = np.ones((I,J))*(1/J)
N_test = []
for k in range(K):
    print(k)
    a = df_test[df_test['annotator_id'] == k]
    N_test += [ create_matrix_N(a) ]
    
def create_matrix_N(A):
    N = np.zeros((I,J))
    for i in list(A['cifar10_test_test_idx']):
        i = int(i)
        j = int(A[A['cifar10_test_test_idx'] == i]['chosen_label'])
        N[i,j] = 1
    return(N)


# PI[k][j,l] : in theory probability that the observer has given the label l
# where j is the right label

# T[i][j] : in theory 1 if j is the correct label for i

# p[j] : in theory probability that a random image has the label j

#%%

PI_test = []
N_test = []
for k in range(1000):
    PI_test += [ np.random.random((10,10)) ]
    N_test += [ np.random.random((10000,10)) ]

T_test = np.random.random((10000,10))
p_test = [1]*10


PI_test2 = []
N_test2 = []
for k in range(K):
    PI_test2 += [ np.random.random((J,J)) ]
    N_test2 += [ np.random.random((I,J)) ]

T_test2 = np.random.random((I,J))
p_test2 = [1]*J

#%%

# T 
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

EM_stepM(T_test, N_test)

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
                    prod_1 = prod_1 * ( PI[k][l,j] ) **(N[k][i,j])
            
            prod_2.append(prod_1)
        
        for l in range(J): 
            T[i,l] = prod_2[l]/sum(prod_2)
    return(T)
        
EM_stepE(EM_stepM(T_test,N_test))
EM_stepE(EM_stepM(T_test2,N_test2))

#%%
def EM_algo(T,N,n):
    t=EM_stepE(EM_stepM(T,N))
    for i in range(n-1):
        t = EM_stepE(EM_stepM(t,N))
    return(t)
#%%
max(EM_algo(T_test,N_test,3)[0,:])

# %%
